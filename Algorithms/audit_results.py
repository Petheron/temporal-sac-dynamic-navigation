"""Catalogue every run found under any number of folders, and change nothing.

Usage:
    python Algorithms\\audit_results.py ^
        --roots "C:\\Users\\Adam_\\Downloads" "<project>\\Results" ^
        --csv "<project>\\results_audit.csv"
"""

import argparse
import csv
import hashlib
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# This script lives in Algorithms/, so the project root is one level up
# Resolving from __file__ means the script works from any working directory
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Running with -I drops this folder from the import path and breaks the next line
sys.path.insert(0, str(SCRIPT_DIR))

from generate_figures import (EXCLUDE, FINAL_N, MISDISPATCH, NAME, NOT_RUNS,
                              SIDE_GROUP, STEP_GROUP, canonical)

# Every run is meant to have trained for this many decision steps
EXPECTED_STEPS = 1000000

# The HPC writes results under a per-run folder named for the job
# Some download archives flatten that to a single file instead
NESTED_PARENT = "SAC_DRLNavigation"

# The submission script writes this where a config key was not present at all
ABSENT = {"", "<absent>", "<none>", "none", "None"}

# cfg_level_type as recorded by the marker, against the letter used in names
ENV_BY_LEVEL = {"5": "d", "6": "h"}

# Every condition in this thesis ran with phase randomisation off
EXPECTED_PHASE = {"", "<absent>", "0", "false", "False"}

# The config filename a job recorded for itself, which the job actually loaded
CONFIG_NAME = re.compile(r"^sac_(?P<arch>bl|ls|tf)_(?P<env>[hd])"
                         r"(?:_w(?P<window>\d+))?(?:_t(?P<duration>\d{3}))?"
                         r"_s(?P<seed>\d)\.yaml$")

# The run name as the submission scripts write it onto the output folder
RUN_NAME = re.compile(r"^(?P<arch>bl|ls|tf)_(?P<env>[hd])"
                      r"(?:_w(?P<window>\d+))?(?:_t(?P<duration>\d{3}))?"
                      r"_s(?P<seed>\d)(?:_b\d+)?(?:_\d+)?$")

# The seed value each index was run with, verified 1 to 1 against every run whose name carries one
# Recorded by the run itself, so it identifies a seed even on runs predating the index scheme
SEED_INDEX = {"4298347": "0", "3690292": "1", "3957050": "2",
              "2412994": "3", "3989801": "4"}

# Config keys read straight out of an archived .yaml
YAML_KEYS = ["temporal_architecture", "level_type", "observation_stacks",
             "sequence_length", "memory_size", "platform_travel_time",
             "randomise_platform_phase", "num_test_eps", "num_action_repeat",
             "seed"]


def present(*values):
    """First value that is actually a value, treating the absent marker as none"""
    for value in values:
        if value is not None and str(value) not in ABSENT:
            return str(value)
    return ""


def as_number(value):
    """Read a config value as a number, or return None if it is not one"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ===== EVIDENCE SOURCES =====

def read_marker(path):
    """Parse a job_marker.txt into a dict, keeping every field it holds"""
    if path is None or not path.is_file():
        return {}
    fields = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            fields[key.strip()] = value.strip()
    return fields


def read_yaml(path):
    """Pull the keys that identify a run out of a .yaml, without a yaml parser

    The configs are flat enough to read line by line, and avoiding the
    dependency means this runs in any environment that can run the training.
    """
    if path is None or not path.is_file():
        return {}
    found = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        for key in YAML_KEYS:
            if stripped.startswith(f"{key}:"):
                found.setdefault(key, stripped[len(key) + 1:].strip())
    return found


def onnx_architecture(path):
    """Name the architecture an .onnx export was produced by, from its bytes

    A memory-carrying export names recurrent_out among its outputs, and an
    LSTM export additionally carries LSTM operators. An export with neither
    has no temporal state, which is observation stacking.
    """
    if path is None or not path.is_file():
        return ""
    blob = path.read_bytes()
    if b"LSTM" in blob:
        return "ls"
    if b"recurrent_out" in blob:
        return "tf"
    return "bl"


def describe_results(path):
    """Read one results.npy and return the facts worth cataloguing

    The fingerprint hashes groups 1 to 6, which hold every recorded series.
    Two files with the same fingerprint are the same run saved twice, whatever
    they are named and wherever they sit.
    """
    data = np.load(path, allow_pickle=True)

    digest = hashlib.sha256(b"".join(
        np.asarray(data[i], dtype=float).tobytes() for i in range(1, 7)))

    steps = np.asarray(data[STEP_GROUP][0])
    side = np.asarray(data[SIDE_GROUP])

    return {
        "fingerprint": digest.hexdigest()[:16],
        "groups_present": int(len(data)),
        "seeds_in_file": int(steps.shape[0]),
        "final_step": int(steps[0][-1]),
        "checkpoints": int(steps.shape[1]),
        "final10_success": round(float(side[0][0][-FINAL_N:].mean()), 4),
        "final10_fall": round(float(side[1][0][-FINAL_N:].mean()), 4),
        "final10_euclidean": round(float(side[2][0][-FINAL_N:].mean()), 4),
        "final10_timeout": round(float(side[3][0][-FINAL_N:].mean()), 4),
    }


# ===== GATHERING =====

def artefacts_for(results_path, run_name):
    """Find the marker, config and export that belong to one results.npy

    A nested run keeps its artefacts inside its own folder. A flattened one
    keeps them beside it, sharing the run name as a filename prefix.
    """
    if results_path.name == "results.npy":
        folder = results_path.parent
        root = folder.parent if folder.name == NESTED_PARENT else folder
        return {
            "marker": next(root.rglob("job_marker.txt"), None),
            "yaml": next(root.rglob("*.yaml"), None),
            "onnx": next(root.rglob("*.onnx"), None),
            "run_dir": root,
        }

    folder = results_path.parent
    return {
        "marker": next(folder.glob(f"{run_name}*job_marker.txt"), None),
        "yaml": next(folder.glob(f"{run_name}*.yaml"), None),
        "onnx": next(folder.glob(f"{run_name}*.onnx"), None),
        "run_dir": results_path,
    }


def find_results(root):
    """Yield (run name, results path) for both layouts under one root"""
    for path in root.rglob("results.npy"):
        parent = path.parent
        name = parent.parent.name if parent.name == NESTED_PARENT else parent.name
        yield name, path

    for path in root.rglob("*_results.npy"):
        name = path.name
        for suffix in ("__results.npy", "_results.npy"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        yield name, path


def survey(roots):
    """Walk every root and return one record per results.npy found"""
    records = []

    for root in roots:
        if not root.is_dir():
            raise SystemExit(f"Not a directory: {root}")

        for name, path in sorted(find_results(root), key=lambda pair: str(pair[1])):
            found = artefacts_for(path, name)
            marker = read_marker(found["marker"])
            config = read_yaml(found["yaml"])

            record = {
                "root": str(root),
                "path": str(path),
                "run_dir": str(found["run_dir"]),
                "run_name": name,
                "layout": "nested" if path.name == "results.npy" else "flat",
                "bytes": path.stat().st_size,
                "has_marker": "yes" if marker else "no",
                "has_yaml": "yes" if config else "no",
                "has_onnx": "yes" if found["onnx"] else "no",
                "onnx_says": onnx_architecture(found["onnx"]),
            }

            try:
                record.update(describe_results(path))
            except Exception as problem:
                record.update({"fingerprint": f"UNREADABLE:{name}",
                               "groups_present": 0, "seeds_in_file": 0,
                               "final_step": 0, "checkpoints": 0,
                               "final10_success": 0.0, "final10_fall": 0.0,
                               "final10_euclidean": 0.0, "final10_timeout": 0.0,
                               "read_error": str(problem)[:120]})

            # Every marker field is carried, not a chosen few
            # Unused fields are kept in case a later question needs them
            record.update(marker)
            for key in YAML_KEYS:
                record[f"yaml_{key}"] = config.get(key, "")

            # Identity is settled per file, because 2 files can hold the same metrics while claiming to be different runs
            record.update(from_folder_name(record["run_name"]))
            record.update(from_config_name(record))
            record.update(from_loaded_config(record))
            record["claimed_arch"] = (record.get("arch_cfg")
                                      or record.get("arch_file")
                                      or record.get("arch_name") or "?")
            records.append(record)

    return records


# ===== IDENTITY =====

def architecture_of(record):
    """Which arm a run actually trained, from the values it loaded

    transformer in temporal_architecture is decisive. Failing that, a non-zero
    memory size or sequence length means the recurrent arm, because the
    baseline runs with both at 0. Everything else is observation stacking.
    """
    switch = present(record.get("cfg_temporal_architecture"),
                     record.get("yaml_temporal_architecture"))
    if switch:
        return "tf" if switch.lower() == "transformer" else switch

    memory = as_number(present(record.get("cfg_memory_size"),
                               record.get("yaml_memory_size")))
    sequence = as_number(present(record.get("cfg_sequence_length"),
                                 record.get("yaml_sequence_length")))
    if (memory or 0) > 0 or (sequence or 0) > 0:
        return "ls"

    stacks = as_number(present(record.get("cfg_observation_stacks"),
                               record.get("yaml_observation_stacks")))
    return "bl" if stacks is not None else ""


def from_loaded_config(record):
    """What the values the run loaded say about it, each kept separate"""
    said = {"arch_cfg": architecture_of(record)}

    level = present(record.get("cfg_level_type"), record.get("yaml_level_type"))
    said["env_cfg"] = ENV_BY_LEVEL.get(level, "")

    # The window lives under a different key for each arm
    if said["arch_cfg"] == "bl":
        window = present(record.get("cfg_observation_stacks"),
                         record.get("yaml_observation_stacks"))
    else:
        window = present(record.get("cfg_sequence_length"),
                         record.get("yaml_sequence_length"))
    said["window_cfg"] = window

    travel = present(record.get("cfg_platform_travel_time"),
                     record.get("yaml_platform_travel_time"))
    # A config with no travel time runs at the scene default, which is 3.0s
    said["duration_cfg"] = travel if travel else "3.0"
    said["duration_default"] = "no" if travel else "yes"

    said["seed_value"] = present(record.get("cfg_seed"), record.get("yaml_seed"))
    said["seed_cfg"] = SEED_INDEX.get(said["seed_value"], "")
    return said


def from_config_name(record):
    """What the config filename the job recorded for itself claims"""
    parsed = CONFIG_NAME.match(present(record.get("config")))
    if parsed is None:
        return {}
    return {f"{k}_file": v for k, v in parsed.groupdict().items() if v is not None}


def from_folder_name(name):
    """What the run's own folder name claims, which is never trusted alone"""
    parsed = RUN_NAME.match(name)
    if parsed is None:
        return {}
    return {f"{k}_name": v for k, v in parsed.groupdict().items() if v is not None}


def merge_group(copies):
    """Build one logical run from every copy of it that was found

    Evidence found on any copy applies to all of them, because they are the
    same run. This is what lets a copy with no marker still be identified.
    """
    best = max(copies, key=lambda r: (r["has_marker"] == "yes",
                                      r["has_yaml"] == "yes",
                                      r["has_onnx"] == "yes",
                                      -len(r["run_name"])))
    run = dict(best)
    run["copies"] = len(copies)
    run["all_paths"] = " | ".join(sorted(r["path"] for r in copies))
    run["all_names"] = " | ".join(sorted({r["run_name"] for r in copies}))

    # Take any field that is blank on the chosen copy from a copy that has it
    for other in copies:
        for field, value in other.items():
            if value not in ("", None) and run.get(field) in ("", None):
                run[field] = value

    return run


def verdict(run):
    """Flags for one logical run, as a list of short strings"""
    flags = []

    if str(run["fingerprint"]).startswith("UNREADABLE"):
        return ["UNREADABLE"]

    # Architecture, cross-checked against each source that records it
    loaded = run.get("arch_cfg", "")
    for label, claimed in (("folder name", run.get("arch_name", "")),
                           ("config filename", run.get("arch_file", "")),
                           ("onnx export", run.get("onnx_says", ""))):
        if claimed and loaded and claimed != loaded:
            flags.append(f"ARCH MISLABELLED ({label} says {claimed}, "
                         f"run loaded {loaded})")

    # Environment
    loaded_env = run.get("env_cfg", "")
    for label, claimed in (("folder name", run.get("env_name", "")),
                           ("config filename", run.get("env_file", ""))):
        if claimed and loaded_env and claimed != loaded_env:
            flags.append(f"ENV MISMATCH ({label} says {claimed}, "
                         f"run loaded {loaded_env})")

    # Window, compared numerically so that 04 and 4 agree
    loaded_window = as_number(run.get("window_cfg", ""))
    for label, claimed in (("folder name", run.get("window_name", "")),
                           ("config filename", run.get("window_file", ""))):
        claimed_number = as_number(claimed)
        if claimed_number is not None and loaded_window is not None \
                and claimed_number != loaded_window:
            flags.append(f"WINDOW MISMATCH ({label} says {int(claimed_number)}, "
                         f"run loaded {int(loaded_window)})")

    # Traverse time, where the name carries deciseconds and the config seconds
    loaded_duration = as_number(run.get("duration_cfg", ""))
    for label, claimed in (("folder name", run.get("duration_name", "")),
                           ("config filename", run.get("duration_file", ""))):
        claimed_number = as_number(claimed)
        if claimed_number is not None and loaded_duration is not None \
                and abs(claimed_number / 10.0 - loaded_duration) > 1e-6:
            flags.append(f"DURATION MISMATCH ({label} says "
                         f"{claimed_number / 10.0}s, run loaded "
                         f"{loaded_duration}s)")

    if run.get("identical_to_arch"):
        flags.append(f"METRICS IDENTICAL TO A RUN CLAIMING "
                     f"{run['identical_to_arch']}")

    index_from_name = run.get("seed_name") or run.get("seed_file")
    if index_from_name and run.get("seed_cfg") and index_from_name != run["seed_cfg"]:
        flags.append(f"SEED MISMATCH (name says s{index_from_name}, "
                     f"value {run.get('seed_value')} is s{run['seed_cfg']})")
    if run.get("seed_value") and not run.get("seed_cfg"):
        flags.append(f"UNKNOWN SEED VALUE ({run['seed_value']})")

    if present(run.get("cfg_randomise_platform_phase")) not in EXPECTED_PHASE:
        flags.append(f"PHASE RANDOMISED "
                     f"({run.get('cfg_randomise_platform_phase')})")

    if run["final_step"] != EXPECTED_STEPS:
        flags.append(f"SHORT RUN ({run['final_step']} steps)")
    if present(run.get("python_exit")) not in ("", "0"):
        flags.append(f"BAD EXIT ({run.get('python_exit')})")
    if present(run.get("cfg_num_test_eps")) not in ("", "0"):
        flags.append(f"TEST EPS ({run.get('cfg_num_test_eps')})")
    if run["seeds_in_file"] > 1:
        flags.append(f"MERGED FILE ({run['seeds_in_file']} seeds)")
    if run["has_marker"] == "no" and run["has_yaml"] == "no":
        flags.append("NO CONFIG EVIDENCE")

    if run["run_name"] in MISDISPATCH:
        flags.append("ON THE MISDISPATCH LIST")
    if NOT_RUNS.match(run["run_name"]) or EXCLUDE.search(run["run_name"]):
        flags.append("EXCLUDED BY NAME RULE")

    return flags


def catalogue(records):
    """Collapse copies into logical runs and give each one a verdict"""
    # 2 files with the same metrics but different claimed architectures are not 1 run saved twice
    # Merging them would hide the defect this audit exists to find, so the claim is part of the key
    groups = defaultdict(list)
    for record in records:
        groups[(record["fingerprint"], record["claimed_arch"])].append(record)

    collisions = defaultdict(set)
    for fingerprint, arch in groups:
        collisions[fingerprint].add(arch)

    runs = []
    for (fingerprint, arch), copies in groups.items():
        run = merge_group(copies)
        others = sorted(collisions[fingerprint] - {arch})
        if others:
            run["identical_to_arch"] = " ".join(others)
        run["flags"] = "; ".join(verdict(run))

        parsed = NAME.match(run["run_name"])
        run["cell"] = ("UNPARSED NAME" if parsed is None
                       else canonical(parsed.group("cell"), parsed.group("batch")))

        # The label the evidence supports, used for the rest of this run
        arch = run.get("arch_cfg") or run.get("arch_file") or run.get("arch_name", "?")
        env = run.get("env_cfg") or run.get("env_file") or run.get("env_name", "?")
        window = run.get("window_cfg", "?")
        duration = run.get("duration_cfg", "?")
        seed = (run.get("seed_cfg") or run.get("seed_name")
                or run.get("seed_file") or "?")
        run["seed"] = seed
        run["true_label"] = f"{arch}_{env}_w{window}_t{duration}s_s{seed}"

        runs.append(run)

    return sorted(runs, key=lambda r: (r["cell"], r.get("seed", "")))


# ===== REPORTING =====

def code_version(run):
    """The tuple of source checksums a run was trained by, as a short string"""
    keys = sorted(k for k in run if k.startswith("md5_") and k != "md5_assembly")
    if not keys:
        return present(run.get("md5_assembly")) or "<not recorded>"
    joined = "|".join(f"{k}={run[k]}" for k in keys)
    return hashlib.sha256(joined.encode()).hexdigest()[:12]


def report(records, runs):
    """Print the catalogue, grouped the way a decision would be made from it"""
    clean = [r for r in runs if not r["flags"]]
    flagged = [r for r in runs if r["flags"]]

    print("===== WHAT WAS FOUND =====")
    print(f"  {len(records)} result files on disk")
    print(f"  {len(runs)} distinct runs behind them")
    print(f"  {len(records) - len(runs)} are duplicate copies")
    print(f"  {len(clean)} runs are clean, {len(flagged)} carry at least 1 flag")

    print("\n===== EVIDENCE AVAILABLE =====")
    for label, field in (("job_marker.txt", "has_marker"),
                         ("archived .yaml", "has_yaml"),
                         (".onnx export", "has_onnx")):
        have = sum(1 for r in runs if r[field] == "yes")
        print(f"  {label:16s} {have} of {len(runs)} runs")

    print("\n===== CODE VERSION =====")
    versions = defaultdict(list)
    for run in runs:
        versions[code_version(run)].append(run)
    for digest, members in sorted(versions.items(), key=lambda kv: -len(kv[1])):
        print(f"  {digest}  ({len(members)} runs)")
        if len(versions) > 1:
            for run in sorted(members, key=lambda r: r["true_label"])[:40]:
                print(f"      {run['true_label']}")
    if len([d for d in versions if d != "<not recorded>"]) > 1:
        print("  more than 1 code version present across these runs")

    print("\n===== CELLS AND SEEDS =====")
    by_cell = defaultdict(list)
    for run in runs:
        by_cell[run["cell"]].append(run)
    for cell in sorted(by_cell):
        members = by_cell[cell]
        seeds = sorted({r.get("seed", "?") for r in members})
        bad = sum(1 for r in members if r["flags"])
        note = f"   {bad} flagged" if bad else ""
        print(f"  {cell:24s} {len(seeds)} seeds   "
              f"{' '.join('s' + s for s in seeds)}{note}")

    print("\n===== DUPLICATES =====")
    dupes = [r for r in runs if r["copies"] > 1]
    if not dupes:
        print("  none")
    for run in dupes:
        print(f"  {run['true_label']}  ({run['copies']} copies)")
        for path in run["all_paths"].split(" | "):
            print(f"      {path}")

    print("\n===== FLAGGED RUNS =====")
    if not flagged:
        print("  none")
    for run in flagged:
        print(f"  {run['true_label']}   folder: {run['run_name']}")
        for flag in run["flags"].split("; "):
            print(f"      {flag}")
        print(f"      {run['path']}")


# ===== BACKUP =====

def sha256(path):
    """Hash a whole file in chunks, so a large checkpoint does not load at once"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# What a run needs to stay identifiable and re-analysable
# Checkpoints and replay buffers are the bulk of a run folder and none of this
LIGHT_PATTERNS = ("results*.npy", "seeds.npy", "job_marker.txt", "*.yaml",
                  "*.onnx", "*.txt")


def light_files(folder):
    """The identifying files inside one run folder"""
    found = []
    for pattern in LIGHT_PATTERNS:
        found.extend(p for p in folder.rglob(pattern) if p.is_file())
    return sorted(set(found))


def folder_size(path):
    """Bytes a run folder holds, or the size of a single file"""
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return path.stat().st_size


def backup(records, destination, light):
    """Copy every catalogued run under destination and verify each copy

    Sources are only ever read. The destination must be empty or absent, so
    an existing backup can never be written over.
    """
    if destination.exists() and any(destination.iterdir()):
        raise SystemExit(f"Refusing to write into a folder that is not empty: "
                         f"{destination}")

    sources = sorted({Path(r["run_dir"]) for r in records})
    copied, total, failed = 0, 0, []

    for index, source in enumerate(sources, 1):
        # A run folder keeps its own name, prefixed by the folder it sat in, so 2 downloads holding the same run name cannot collide
        target = destination / f"{source.parent.name}__{source.name}"

        if source.is_dir() and light:
            pairs = []
            for original in light_files(source):
                copy = target / original.relative_to(source)
                copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, copy)
                pairs.append((original, copy))
        elif source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=False)
            pairs = [(p, target / p.relative_to(source))
                     for p in source.rglob("*") if p.is_file()]
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            pairs = [(source, target)]

        for original, copy in pairs:
            if sha256(original) != sha256(copy):
                failed.append(str(original))
            else:
                copied += 1
                total += original.stat().st_size

        print(f"  [{index}/{len(sources)}] {target.name}")

    print(f"\ncopied {copied} files, {total / 1e9:.2f} GB")
    if failed:
        print(f"{len(failed)} files did not verify:")
        for path in failed:
            print(f"  {path}")
        raise SystemExit("Backup is NOT verified, do not delete anything")
    print("every copied file matches its original by hash")


# ===== DRIVER =====

# Identity and verdict first, then the raw evidence, then where it all sits
LEAD_COLUMNS = ["true_label", "flags", "cell", "copies", "run_name", "all_names",
                "fingerprint", "arch_cfg", "arch_file", "arch_name", "onnx_says",
                "env_cfg", "env_file", "env_name", "window_cfg", "window_file",
                "window_name", "duration_cfg", "duration_default", "duration_file",
                "duration_name", "seed", "seed_cfg", "seed_value", "seed_file",
                "seed_name", "identical_to_arch",
                "seeds_in_file", "final_step", "checkpoints", "groups_present",
                "final10_success", "final10_fall", "final10_euclidean",
                "final10_timeout", "has_marker", "has_yaml", "has_onnx", "layout"]

TAIL_COLUMNS = ["root", "run_dir", "path", "all_paths"]

FILE_COLUMNS = ["fingerprint", "run_name", "layout", "bytes", "has_marker",
                "has_yaml", "has_onnx", "root", "path"]


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="+", required=True,
                        help="folders to search, searched recursively")
    parser.add_argument("--csv", type=Path, default=None,
                        help="write the catalogue here, 1 row per distinct run")
    parser.add_argument("--backup", type=Path, default=None,
                        help="copy every catalogued run here and verify by hash")
    parser.add_argument("--light", action="store_true",
                        help="back up only the files that identify a run")
    args = parser.parse_args()

    for root in args.roots:
        print(f"searching: {root}")
    print()

    records = survey(args.roots)
    if not records:
        raise SystemExit("No results.npy found under any of those roots")

    runs = catalogue(records)
    report(records, runs)

    folders = sorted({Path(r["run_dir"]) for r in records})
    full = sum(folder_size(f) for f in folders)
    small = sum(sum(p.stat().st_size for p in light_files(f)) if f.is_dir()
                else f.stat().st_size for f in folders)
    print("\n===== BACKUP SIZE =====")
    print(f"  {len(folders)} run folders")
    print(f"  everything          {full / 1e9:7.2f} GB")
    print(f"  --light             {small / 1e9:7.2f} GB  "
          f"(results, markers, configs, exports)")

    if args.csv is not None:
        # Every marker field that any run carried becomes a column
        extra = sorted({k for run in runs for k in run
                        if k.startswith(("cfg_", "md5_", "yaml_"))})
        extra += [k for k in ("config", "python_exit", "walltime_used")
                  if any(k in run for run in runs)]
        write_csv(args.csv, LEAD_COLUMNS + extra + TAIL_COLUMNS, runs)

        files_csv = args.csv.with_name(args.csv.stem + "_files" + args.csv.suffix)
        write_csv(files_csv, FILE_COLUMNS, records)

        print(f"\nwrote {args.csv}  ({len(LEAD_COLUMNS + extra + TAIL_COLUMNS)} columns)")
        print(f"wrote {files_csv}")

    if args.backup is not None:
        print("\n===== BACKUP =====")
        backup(records, args.backup, args.light)

    if args.csv is None and args.backup is None:
        print("\nNothing was written. Pass --csv to save the catalogue.")
    else:
        print("\nNothing was moved, renamed or deleted.")


if __name__ == "__main__":
    main()
