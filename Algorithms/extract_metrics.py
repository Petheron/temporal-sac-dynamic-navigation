"""Extract every metric needed for the duration curve and the outcome tables.

Usage:
    python -I Algorithms\\extract_metrics.py Results
    python -I Algorithms\\extract_metrics.py "C:\\path\\to\\one download folder"
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

# Group and element indices, from data_to_save in OffPolicyMain.py
STEP_GROUP, STEP_INDEX = 0, 0
TIME_GROUP, TIME_INDEX = 0, 1
LENGTH_GROUP, LENGTH_INDEX = 4, 0
SIDE_GROUP = 5
SUCCESS, FAIL, EUCLIDEAN, TIMEOUT = 0, 1, 2, 3

CHECKPOINTS_PER_SEED = 100
FINAL_N = 10

# A run directory is <cell>_s<seed>_<jobid>, or <cell>_s<seed> for older runs
NAME_PATTERN = re.compile(r"^(?P<cell>.+?)_s(?P<seed>\d+)(?:_(?P<job>\d+))?$")


def series(data, group, index):
    """Return one series as a flat array, whatever nesting it arrived in."""
    return np.asarray(data[group][index]).ravel()


def tail_mean(values, written):
    """Mean of the last 10 written checkpoints."""
    live = values[:written]
    return float(live[-FINAL_N:].mean()) if written >= FINAL_N else float(live.mean())


def read_run(path):
    """Return every metric for 1 results file, or None if it is unreadable."""
    data = np.load(path, allow_pickle=True)
    steps = series(data, STEP_GROUP, STEP_INDEX)

    # A checkpoint that was never written leaves its step count at 0
    written = int((steps > 0).sum())
    if written == 0:
        return None

    row = {
        "written": written,
        "steps": int(steps[written - 1]),
        "hours": float(series(data, TIME_GROUP, TIME_INDEX)[:written].max()) / 3600.0,
        "length": tail_mean(series(data, LENGTH_GROUP, LENGTH_INDEX), written),
        "success": tail_mean(series(data, SIDE_GROUP, SUCCESS), written),
        "fail": tail_mean(series(data, SIDE_GROUP, FAIL), written),
        "euclidean": tail_mean(series(data, SIDE_GROUP, EUCLIDEAN), written),
        "timeout": tail_mean(series(data, SIDE_GROUP, TIMEOUT), written),
    }
    row["sum"] = row["success"] + row["fail"] + row["euclidean"] + row["timeout"]

    # More than 100 checkpoints means several seeds packed into 1 file
    row["seeds"] = written // CHECKPOINTS_PER_SEED if written >= CHECKPOINTS_PER_SEED else 1
    row["complete"] = written >= CHECKPOINTS_PER_SEED and row["seeds"] == 1
    return row


def marker_field(run_dir, key):
    """Return 1 field from the job marker, or an empty string."""
    marker = run_dir / "job_marker.txt"
    if not marker.is_file():
        return ""
    for line in marker.read_text(errors="replace").splitlines():
        if line.startswith(f"{key}="):
            return line.partition("=")[2].strip()
    return ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+")
    args = parser.parse_args()

    rows = []
    seen = set()

    for root in args.roots:
        root = Path(root)
        if not root.is_dir():
            print(f"SKIPPED, not a directory: {root}")
            continue

        for results in sorted(root.rglob("results.npy")):
            key = str(results.resolve())
            if key in seen:
                continue
            seen.add(key)

            name = results.parent.parent.name
            try:
                row = read_run(results)
            except Exception as error:
                print(f"READ FAILED  {name}  {error}")
                continue
            if row is None:
                continue

            match = NAME_PATTERN.match(name)
            row["name"] = name
            row["cell"] = match.group("cell") if match else name
            row["seed"] = match.group("seed") if match else "?"
            row["job"] = marker_field(results.parent, "job_id").split(".")[0]
            row["travel"] = marker_field(results.parent, "cfg_platform_travel_time")
            row["stacks"] = marker_field(results.parent, "cfg_observation_stacks")
            row["seqlen"] = marker_field(results.parent, "cfg_sequence_length")
            rows.append(row)

    if not rows:
        raise SystemExit("No results.npy found under the given roots.")

    rows.sort(key=lambda r: (r["cell"], r["seed"]))

    header = (
        f"{'run':<26} {'job':>7} {'ckpt':>5} {'trav':>5} {'stk':>4} {'seq':>4} "
        f"{'hours':>6} {'eplen':>7} {'succ':>6} {'fail':>6} {'eucl':>6} "
        f"{'tout':>6} {'sum':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        flag = "" if r["complete"] else "  NOT COMPLETE"
        print(
            f"{r['name']:<26} {r['job']:>7} {r['written']:>5} "
            f"{r['travel'] or '-':>5} {r['stacks'] or '?':>4} {r['seqlen'] or '?':>4} "
            f"{r['hours']:>6.2f} {r['length']:>7.1f} {r['success']:>6.3f} "
            f"{r['fail']:>6.3f} {r['euclidean']:>6.3f} {r['timeout']:>6.3f} "
            f"{r['sum']:>6.3f}{flag}"
        )

    # Per-cell means are what the decision criteria are written against
    print()
    print("PER-CELL MEANS, complete runs only")
    print()
    head = (f"{'cell':<22} {'n':>2} {'succ':>6} {'fail':>6} {'eucl':>6} "
            f"{'tout':>6} {'eplen':>7}   per-seed success")
    print(head)
    print("-" * len(head))

    cells = defaultdict(list)
    for r in rows:
        if r["complete"]:
            cells[r["cell"]].append(r)

    for cell in sorted(cells):
        group = sorted(cells[cell], key=lambda r: r["seed"])
        n = len(group)
        print(
            f"{cell:<22} {n:>2} "
            f"{np.mean([r['success'] for r in group]):>6.3f} "
            f"{np.mean([r['fail'] for r in group]):>6.3f} "
            f"{np.mean([r['euclidean'] for r in group]):>6.3f} "
            f"{np.mean([r['timeout'] for r in group]):>6.3f} "
            f"{np.mean([r['length'] for r in group]):>7.1f}   "
            + ", ".join(f"s{r['seed']}={r['success']:.3f}" for r in group)
        )

    # The 4 outcome rates are a partition, so any sum off 1.0 is a defect
    print()
    bad = [r for r in rows if abs(r["sum"] - 1.0) > 0.005]
    if bad:
        print("OUTCOME RATES DO NOT SUM TO 1.0, investigate before using:")
        for r in bad:
            print(f"  {r['name']:<26} sum {r['sum']:.4f}")
    else:
        print("All outcome rates sum to 1.000, so the 4 categories are a partition.")

    incomplete = [r for r in rows if not r["complete"]]
    if incomplete:
        print("\nNOT COMPLETE, exclude from analysis:")
        for r in incomplete:
            print(f"  {r['name']:<26} {r['written']:>3} ckpt  {r['steps']:>9} steps")


if __name__ == "__main__":
    main()
