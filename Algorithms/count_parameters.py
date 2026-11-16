"""Count trainable parameters per architecture from saved weights.

Usage:
    python Algorithms\\count_parameters.py Results "C:\\Users\\Adam_\\Downloads"
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import torch

# Files to read, and the label each one reports under
# log_alpha is optional because a smoke test may not have written it
WEIGHT_FILES = (
    ("actor_dict_0", "actor", True),
    ("critic_dict_0", "critic (twins)", True),
    ("log_alpha_dict_0", "log_alpha", False),
)

# A key matching any of these holds a buffer rather than a trainable weight
# Buffers are reported separately so the count can be checked
BUFFER_PATTERNS = (
    "running_mean",
    "running_var",
    "running_std",
    "num_batches_tracked",
    "normalizer",
    "normalization_steps",
)

ARCH_LABEL = {
    "bl": "Frame stacking",
    "ls": "Long Short-Term Memory",
    "tf": "Transformer",
}

# Older production runs predate the <arch>_<env> naming scheme
LEGACY_ARCH = (
    ("lstm_", "ls"),
    ("bl_hz", "bl"),
    ("bl_dw", "bl"),
    ("tf_hz", "tf"),
    ("tf_dw", "tf"),
    ("tf_smoke", "tf"),
)


def is_buffer(key):
    """Return True when a state dict key names a buffer, not a weight."""
    lowered = key.lower()
    return any(pattern in lowered for pattern in BUFFER_PATTERNS)


def torch_load(path):
    """Load a checkpoint across PyTorch versions.

    weights_only was added to torch.load in 1.13 and defaults to True from
    2.6 onward. Passing it to an older build raises TypeError, and omitting
    it on a newer build would refuse a plain pickle, so both are tried.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def read_weights(path):
    """Return trainable count, buffer count and buffer key names.

    Handles 3 shapes: a bare tensor, a plain state dict, and a dict that
    wraps a state dict under a "state_dict" key.
    """
    obj = torch_load(path)

    # log_alpha is a bare tensor rather than a mapping
    if torch.is_tensor(obj):
        return obj.numel(), 0, []

    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]

    if not isinstance(obj, dict):
        raise TypeError(f"unsupported object of type {type(obj).__name__}")

    trainable = 0
    buffered = 0
    buffer_keys = []
    for key, value in obj.items():
        if not torch.is_tensor(value):
            continue
        count = value.numel()
        if is_buffer(key):
            buffered += count
            buffer_keys.append(f"{key} ({count:,})")
        else:
            trainable += count
    return trainable, buffered, buffer_keys


def arch_of(run_name):
    """Return the 2 letter architecture code for a run directory name."""
    for prefix, code in LEGACY_ARCH:
        if run_name.startswith(prefix):
            return code
    match = re.match(r"^(bl|ls|tf)_", run_name)
    return match.group(1) if match else None


def marker_field(folder, key):
    """Return 1 field from the job marker, or an empty string."""
    marker = folder / "job_marker.txt"
    if not marker.is_file():
        return ""
    for line in marker.read_text(errors="replace").splitlines():
        if line.startswith(f"{key}="):
            return line.partition("=")[2].strip()
    return ""


def window_of(folder, run_name, code):
    """Return the temporal window, from the marker or else the run name."""
    key = "cfg_observation_stacks" if code == "bl" else "cfg_sequence_length"
    found = marker_field(folder, key)
    if found and found.isdigit():
        return int(found)

    # Older runs predate the provenance block, so fall back to the name
    match = re.search(r"_w(\d{2})", run_name)
    if match:
        return int(match.group(1))

    # The production runs carry no window in their names
    # Frame stacking used 4 and the other 2 used sequence 10
    return 4 if code == "bl" else 10


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+")
    parser.add_argument(
        "--max-per-group",
        type=int,
        default=2,
        help="Runs to read per architecture and window. 0 reads them all.",
    )
    args = parser.parse_args()

    # Printed because the load path differs across PyTorch versions
    print(f"torch {torch.__version__}")
    print()

    # Group candidates first so only a few weight files are ever opened
    groups = defaultdict(list)
    seen = set()

    for root in args.roots:
        root = Path(root)
        if not root.is_dir():
            print(f"SKIPPED, not a directory: {root}")
            continue
        for actor in sorted(root.rglob("actor_dict_0")):
            key = str(actor.resolve())
            if key in seen:
                continue
            seen.add(key)
            folder = actor.parent
            run_name = folder.parent.name
            code = arch_of(run_name)
            if code is None:
                continue
            groups[(code, window_of(folder, run_name, code))].append((run_name, folder))

    if not groups:
        raise SystemExit(
            "Found no run directory holding actor_dict_0 whose name starts "
            "with bl, ls, tf or lstm. Check the paths given."
        )

    results = {}
    failures = []

    for key in sorted(groups, key=lambda k: (k[0], k[1])):
        candidates = groups[key]
        take = candidates if args.max_per_group == 0 else candidates[: args.max_per_group]
        for run_name, folder in take:
            totals = {}
            buffers = 0
            buffer_keys = []
            fatal = False
            for filename, label, required in WEIGHT_FILES:
                path = folder / filename
                if not path.is_file():
                    if required:
                        failures.append(f"{run_name}: {filename} missing")
                        fatal = True
                        break
                    totals[label] = 0
                    continue
                try:
                    trainable, buffered, keys = read_weights(path)
                except Exception as error:
                    failures.append(f"{run_name}: {filename}: {error}")
                    if required:
                        fatal = True
                        break
                    totals[label] = 0
                    continue
                totals[label] = trainable
                buffers += buffered
                buffer_keys.extend(f"{filename}: {k}" for k in keys)
            if fatal:
                continue
            totals["_buffers"] = buffers
            totals["_keys"] = buffer_keys
            totals["_total"] = sum(
                v for k, v in totals.items() if not k.startswith("_")
            )
            results.setdefault(key, {})[run_name] = totals

    if not results:
        print()
        print("No run could be read. Failures:")
        for note in dict.fromkeys(failures):
            print(f"  {note}")
        raise SystemExit(1)

    labels = [label for _, label, _ in WEIGHT_FILES]
    header = (
        f"{'architecture':<24} {'win':>4} {'run':<26} "
        + " ".join(f"{lab:>16}" for lab in labels)
        + f" {'TRAINABLE':>12} {'buffers':>8}"
    )
    print(header)
    print("-" * len(header))

    for key in sorted(results, key=lambda k: (k[0], k[1])):
        code, window = key
        for run_name, t in sorted(results[key].items()):
            print(
                f"{ARCH_LABEL[code]:<24} {window:>4} {run_name:<26} "
                + " ".join(f"{t[lab]:>16,}" for lab in labels)
                + f" {t['_total']:>12,} {t['_buffers']:>8,}"
            )

    # A disagreement inside 1 group means a config difference worth knowing
    disagree = []
    for key, runs in results.items():
        distinct = sorted({t["_total"] for t in runs.values()})
        if len(distinct) > 1:
            disagree.append(
                f"{ARCH_LABEL[key[0]]} at window {key[1]}: "
                + ", ".join(f"{d:,}" for d in distinct)
            )
    if disagree:
        print()
        print("WARNING, runs of the same configuration disagree:")
        for note in disagree:
            print(f"  {note}")

    print()
    print("FOR THE METHODOLOGY TABLE, production configuration only")
    print("  bl at 4 stacks, ls and tf at sequence 10.")
    print()
    for code in ("bl", "ls", "tf"):
        want = 4 if code == "bl" else 10
        key = (code, want)
        if key not in results:
            print(f"  {ARCH_LABEL[code]:<24} no run found at window {want}")
            continue
        totals = [t["_total"] for t in results[key].values()]
        best = max(set(totals), key=totals.count)
        print(f"  {ARCH_LABEL[code]:<24} {best:>12,}  = {best / 1e6:.2f}M")

    print()
    print("Buffers excluded as non-trainable:")
    shown = set()
    for key in sorted(results, key=lambda k: (k[0], k[1])):
        for t in results[key].values():
            for name in t["_keys"]:
                if name not in shown:
                    shown.add(name)
                    print(f"  {name}")
    if not shown:
        print("  none detected, so every saved tensor was counted as trainable")

    if failures:
        print()
        print("Non-fatal read problems, deduplicated:")
        for note in dict.fromkeys(failures):
            print(f"  {note}")


if __name__ == "__main__":
    main()
