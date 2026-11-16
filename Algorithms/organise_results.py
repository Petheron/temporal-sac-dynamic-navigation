"""Reorganise the results folder into one run per cell, and nothing deleted.

Usage:
    python Algorithms\\organise_results.py ^
        --roots "C:\\Users\\Adam_\\Downloads" "<project>\\Results" ^
        --results-dir "<project>\\Results"
"""

import argparse
import csv
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# The repository root, found from this file rather than the shell
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Without this, -I would leave the import below with nowhere to look
sys.path.insert(0, str(SCRIPT_DIR))

from audit_results import catalogue, survey

# A flag containing any of these means the run is not production evidence
JUNK_FLAGS = ("EXCLUDED BY NAME RULE", "ON THE MISDISPATCH LIST", "SHORT RUN",
              "MERGED FILE", "NO CONFIG EVIDENCE", "UNKNOWN SEED VALUE",
              "ARCH MISLABELLED", "ENV MISMATCH", "WINDOW MISMATCH",
              "DURATION MISMATCH", "SEED MISMATCH", "BAD EXIT", "TEST EPS",
              "PHASE RANDOMISED", "UNREADABLE")

# A flag that only records a relationship, which does not disqualify a run
# The 6 September LSTM runs carry this because their Transformer twins exist
BENIGN_FLAGS = ("METRICS IDENTICAL TO A RUN CLAIMING",)

# The comparisons Plotting.py draws, each 1 folder of subfolders
# The numeric prefixes make alphabetical order match the intended legend order
GROUPS = [
    ("01_arms_horizontal_t030", [("1_stack4", "bl_h_t030"),
                                 ("2_lstm", "ls_h_t030"),
                                 ("3_transformer", "tf_h_t030")]),
    ("02_arms_horizontal_t070", [("1_stack4", "bl_h_t070"),
                                 ("2_lstm", "ls_h_t070"),
                                 ("3_transformer", "tf_h_t070")]),
    ("03_arms_horizontal_t100", [("1_stack4", "bl_h_t100"),
                                 ("2_lstm", "ls_h_t100"),
                                 ("3_transformer", "tf_h_t100")]),
    ("04_arms_horizontal_t120", [("1_stack4", "bl_h_t120"),
                                 ("2_lstm", "ls_h_t120"),
                                 ("3_transformer", "tf_h_t120")]),
    ("05_arms_depthwise_t030", [("1_stack4", "bl_d_t030"),
                                ("2_lstm", "ls_d_t030"),
                                ("3_transformer", "tf_d_t030")]),
    ("06_stack_curve_horizontal", [("1_stack02", "bl_h_w02"),
                                   ("2_stack04", "bl_h_t030"),
                                   ("3_stack08", "bl_h_w08"),
                                   ("4_stack10", "bl_h_w10"),
                                   ("5_stack16", "bl_h_w16")]),
    ("07_duration_curve_horizontal", [("1_t030", "bl_h_t030"),
                                      ("2_t040", "bl_h_t040"),
                                      ("3_t050", "bl_h_t050"),
                                      ("4_t070", "bl_h_t070"),
                                      ("5_t080", "bl_h_t080"),
                                      ("6_t090", "bl_h_t090"),
                                      ("7_t100", "bl_h_t100"),
                                      ("8_t110", "bl_h_t110"),
                                      ("9_t120", "bl_h_t120")]),
    # Section 3.5 compares the arms on a matched 10-decision input
    ("08_matched_context_t100", [("1_stack10", "bl_h_w10_t100"),
                                 ("2_lstm", "ls_h_t100"),
                                 ("3_transformer", "tf_h_t100")]),
    # The batch-size control, which is stack-4 at 10.0s against itself at 320
    ("09_batch_size_t100", [("1_batch_default", "bl_h_t100"),
                            ("2_batch_320", "bl_h_w04_t100_b320")]),
]


def label_for(run):
    """The name a clean run is filed under

    The batch size is part of the identity, because bl_h_t100 and its
    batch-320 twin differ in nothing else and are separate cells in the
    analysis. canonical() already carries it on the cell key.
    """
    label = run["true_label"]
    batch = re.search(r"_b(\d+)$", run["cell"])
    return f"{label}_b{batch.group(1)}" if batch else label


def unique_targets(actions):
    """Give every action a target no other action wants

    A collision is settled by appending part of the fingerprint, which is
    unique by construction. Two runs can share a folder name without being
    the same run, and 4 of the old aggregates are called Baseline_4Stack.
    """
    seen = {}
    clashes = []
    for action in actions:
        target = action["target"]
        if target in seen:
            action["target"] = f"{target}_{action['fingerprint'][:6]}"
            clashes.append((target, action["source"]))
        seen[action["target"]] = action["source"]
    return clashes


def classify(run):
    """Return the reason a run is quarantined, or an empty string if it is clean"""
    for flag in run["flags"].split("; "):
        if not flag:
            continue
        if any(flag.startswith(b) for b in BENIGN_FLAGS):
            continue
        for junk in JUNK_FLAGS:
            if junk in flag:
                # The reason becomes a folder name, so it loses its detail
                return junk.lower().replace(" ", "_")
    return ""


def run_folder(results_path, forbidden):
    """The folder that holds one run, or the file itself if there is none

    Walking up from results.npy finds the run folder in the normal layout.
    A loose results.npy sitting in a download folder has no run folder of its
    own, and walking up from it lands on a folder holding unrelated runs.
    Taking that folder would move or copy everything inside it, so a folder
    that is forbidden, or that holds other catalogued runs, is not used.
    """
    folder = results_path.parent
    if folder.name == "SAC_DRLNavigation":
        folder = folder.parent
    if folder in forbidden or not folder.is_dir():
        return results_path
    return folder


def forbidden_folders(records, roots):
    """Folders that must never be taken as a single run's folder

    Every search root, every ancestor of one, and any folder that more than 1
    run would resolve to.
    """
    blocked = set()
    for root in roots:
        blocked.add(root.resolve())
        blocked.update(root.resolve().parents)

    counts = defaultdict(int)
    for record in records:
        path = Path(record["path"])
        folder = path.parent
        if folder.name == "SAC_DRLNavigation":
            folder = folder.parent
        counts[folder] += 1
    blocked.update(folder for folder, n in counts.items() if n > 1)
    return blocked


def pick_source(run, results_dir, forbidden):
    """Which copy of a run to keep, preferring one already in the results folder"""
    folders = [run_folder(Path(p), forbidden)
               for p in run["all_paths"].split(" | ")]
    inside = [f for f in folders if results_dir in f.parents]
    chosen = inside[0] if inside else folders[0]
    return chosen, [f for f in folders if f != chosen]


def merge_arrays(paths):
    """Concatenate several results.npy into 1, seed by seed

    This repeats what merge_results.py does, so a merged file here is the same
    file that script would have produced from the same inputs.
    """
    datasets = [np.load(p, allow_pickle=True) for p in paths]
    merged = []
    for group in range(len(datasets[0])):
        rows = []
        for series in range(len(datasets[0][group])):
            rows.append(np.concatenate(
                [d[group][series] for d in datasets], axis=0))
        merged.append(rows)
    return np.array(merged, dtype=object)


def build_plan(runs, results_dir, destination, forbidden):
    """Work out every action without taking any of them"""
    actions = []
    by_cell = defaultdict(list)

    for run in runs:
        reason = classify(run)
        source, extras = pick_source(run, results_dir, forbidden)
        if reason:
            label = f"{source.parent.name}__{source.name}"
            target = destination / "quarantine" / reason / label
            if source.is_file():
                # A loose results.npy is filed into a folder named for it
                target = target.with_suffix("") / source.name
        else:
            target = destination / "runs" / label_for(run)
            by_cell[run["cell"]].append((run, target))

        actions.append({
            "kind": "move" if results_dir in source.parents else "copy",
            "label": label_for(run),
            "cell": run["cell"],
            "seed": run.get("seed", ""),
            "reason": reason or "clean",
            "fingerprint": run["fingerprint"],
            "source": str(source),
            "target": str(target),
            "redundant": " | ".join(str(e) for e in extras),
        })

    clashes = unique_targets(actions)

    targets = [a["target"] for a in actions]
    if len(set(targets)) != len(targets):
        raise SystemExit("Two actions still want the same target, aborting")

    return actions, by_cell, clashes


def report(actions, by_cell, clashes):
    """Print the plan, grouped the way a decision would be made from it"""
    moves = [a for a in actions if a["kind"] == "move"]
    copies = [a for a in actions if a["kind"] == "copy"]
    clean = [a for a in actions if a["reason"] == "clean"]
    held = [a for a in actions if a["reason"] != "clean"]
    redundant = sum(len(a["redundant"].split(" | ")) if a["redundant"] else 0
                    for a in actions)

    print("===== PLAN =====")
    print(f"  {len(clean)} clean runs to runs/")
    print(f"  {len(held)} flagged runs to quarantine/")
    print(f"  {len(moves)} moved from inside the results folder")
    print(f"  {len(copies)} copied in from elsewhere")
    print(f"  {redundant} redundant copies left where they are")

    if clashes:
        print(f"\n===== NAME CLASHES RESOLVED =====")
        print(f"  {len(clashes)} targets were wanted twice and got a")
        print(f"  fingerprint suffix to tell them apart")
        for target, source in clashes[:10]:
            print(f"      {Path(target).name}  <-  {source}")

    print("\n===== QUARANTINE =====")
    reasons = defaultdict(int)
    for action in held:
        reasons[action["reason"]] += 1
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:28s} {count}")

    print("\n===== PLOTTING GROUPS =====")
    for group, members in GROUPS:
        parts = []
        for arm, cell in members:
            parts.append(f"{arm}({len(by_cell.get(cell, []))})")
        print(f"  {group:30s} {'  '.join(parts)}")


def apply_plan(actions, destination, manifest_path):
    """Carry out the plan, never overwriting and never deleting

    Each action is written to the manifest as it completes rather than at the
    end. An interruption then leaves a manifest covering exactly what was
    done, which --undo can still reverse.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    done = 0

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        handle.flush()

        for index, action in enumerate(actions, 1):
            source, target = Path(action["source"]), Path(action["target"])
            if target.exists():
                raise SystemExit(f"Refusing to overwrite: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)

            if action["kind"] == "move":
                shutil.move(str(source), str(target))
            elif source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)

            writer.writerow(action)
            handle.flush()
            done += 1

            if index % 25 == 0 or index == len(actions):
                print(f"  {index}/{len(actions)}")

    return done


def build_plotting(by_cell, destination):
    """Write 1 merged results.npy per arm, under each comparison group"""
    written = 0
    for group, members in GROUPS:
        for arm, cell in members:
            entries = sorted(by_cell.get(cell, []), key=lambda e: e[0].get("seed", ""))
            if not entries:
                continue
            sources = []
            for _, target in entries:
                found = next(Path(target).rglob("results.npy"), None)
                if found is not None:
                    sources.append(found)
            if not sources:
                continue
            out = destination / "plotting" / group / arm
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "results.npy", merge_arrays(sources))
            written += 1
            print(f"  {group}/{arm}  {len(sources)} seeds")
    return written


def undo(manifest_path):
    """Put every moved folder back where the manifest says it came from"""
    rows = list(csv.DictReader(open(manifest_path, encoding="utf-8")))
    restored = 0
    for row in reversed(rows):
        if row["kind"] != "move":
            continue
        source, target = Path(row["source"]), Path(row["target"])
        if not target.exists():
            print(f"  already gone, skipped: {target}")
            continue
        if source.exists():
            print(f"  original is back already, skipped: {source}")
            continue
        source.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target), str(source))
        restored += 1
    print(f"restored {restored} folders")
    print("Copies made from outside the results folder were left in place.")


MANIFEST_COLUMNS = ["kind", "label", "cell", "seed", "reason", "fingerprint",
                    "source", "target", "redundant"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="+",
                        help="folders to catalogue, searched recursively")
    parser.add_argument("--results-dir", type=Path,
                        help="the results folder being reorganised")
    parser.add_argument("--dest", type=Path, default=None,
                        help="where to build, default <results-dir>\\organised")
    parser.add_argument("--apply", action="store_true",
                        help="carry out the plan, otherwise only print it")
    parser.add_argument("--undo", type=Path, default=None,
                        help="reverse a previous run from its manifest")
    args = parser.parse_args()

    if args.undo is not None:
        if not args.undo.is_file():
            raise SystemExit(f"Manifest not found: {args.undo}")
        undo(args.undo)
        return

    if not args.roots or not args.results_dir:
        raise SystemExit("--roots and --results-dir are both required")
    if not args.results_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.results_dir}")

    results_dir = args.results_dir.resolve()
    destination = (args.dest or results_dir / "organised").resolve()

    if destination.exists() and any(destination.iterdir()):
        raise SystemExit(f"Refusing to write into a folder that is not empty: "
                         f"{destination}")

    print(f"results dir: {results_dir}")
    print(f"destination: {destination}\n")

    records = survey(args.roots)
    runs = catalogue(records)
    print(f"catalogued {len(runs)} distinct runs from {len(records)} files\n")

    forbidden = forbidden_folders(records, args.roots)
    actions, by_cell, clashes = build_plan(runs, results_dir, destination,
                                           forbidden)
    report(actions, by_cell, clashes)

    if not args.apply:
        print("\nNothing was written. Pass --apply to carry this out.")
        return

    manifest = destination / "manifest.csv"
    print("\n===== APPLYING =====")
    print(f"  recording every action to {manifest} as it happens")
    done = apply_plan(actions, destination, manifest)
    print(f"  {done} actions recorded")

    print("\n===== PLOTTING GROUPS =====")
    written = build_plotting(by_cell, destination)
    print(f"wrote {written} merged series")

    print("\nNothing was deleted. Redundant copies are listed in the manifest.")
    print(f"To reverse every move: --undo \"{manifest}\"")


if __name__ == "__main__":
    main()
