"""Check that every result directory agrees with its own recorded config.

Lives in Algorithms/ and defaults to the folder the window sweep downloads to.
"""

import argparse
import re
from pathlib import Path

# Root of the repository, which is the parent of Algorithms/
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results_check" / "wsweep_upload"

# Matches the run name as it appears in the marker's config= field
CONFIG_PATTERN = re.compile(
    r"^sac_(?P<arch>bl|ls|tf)_(?P<env>[hd])_w(?P<window>\d{2})"
    r"_t(?P<duration>\d{3})_s(?P<seed>\d)\.yaml$"
)

WINDOW_KEY = {
    "bl": "observation_stacks",
    "ls": "sequence_length",
    "tf": "sequence_length",
}

LEVEL_BY_ENV = {"h": "6", "d": "5"}


def read_marker(path):
    """Parse a job_marker.txt into a dict."""
    fields = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            fields[key.strip()] = value.strip()
    return fields


def check(result_dir):
    """Problems found in one result directory, as a list of strings."""
    problems = []

    marker_path = next(result_dir.glob("**/job_marker.txt"), None)
    if marker_path is None:
        return [f"{result_dir.name}: no job_marker.txt"]

    marker = read_marker(marker_path)
    label = result_dir.name

    match = CONFIG_PATTERN.match(marker.get("config", ""))
    if match is None:
        # Runs from before the naming scheme get reported, not failed
        return [f"{label}: config '{marker.get('config')}' predates the naming "
                f"scheme, skipped"]

    claimed = match.groupdict()
    arch = claimed["arch"]

    # Window in the name must match the key this architecture actually reads
    window_key = f"cfg_{WINDOW_KEY[arch]}"
    recorded = marker.get(window_key)
    if recorded != str(int(claimed["window"])):
        problems.append(
            f"{label}: name says window {int(claimed['window'])}, "
            f"{window_key} says {recorded}"
        )

    # Environment in the name must match the level type
    if marker.get("cfg_level_type") != LEVEL_BY_ENV[claimed["env"]]:
        problems.append(
            f"{label}: name says env {claimed['env']}, cfg_level_type says "
            f"{marker.get('cfg_level_type')}"
        )

    # A tf run must carry the architecture switch, and bl and ls must not
    # This is the check that would have caught the September mislabelling
    arch_field = marker.get("cfg_temporal_architecture", "<absent>")
    if arch == "tf" and arch_field != "transformer":
        problems.append(
            f"{label}: name says tf, cfg_temporal_architecture says {arch_field}"
        )
    if arch != "tf" and arch_field == "transformer":
        problems.append(
            f"{label}: name says {arch}, cfg_temporal_architecture says "
            f"transformer"
        )

    # num_test_eps above 0 crashes at the first periodic eval on test_lengths
    if marker.get("cfg_num_test_eps") != "0":
        problems.append(
            f"{label}: cfg_num_test_eps is {marker.get('cfg_num_test_eps')}, "
            f"expected 0"
        )

    if marker.get("python_exit") != "0":
        problems.append(f"{label}: python_exit={marker.get('python_exit')}")

    # results.npy is the file the analysis actually needs
    if not list(result_dir.glob("**/results.npy")):
        problems.append(f"{label}: no results.npy")

    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_dir", type=Path, nargs="?", default=DEFAULT_RESULTS_DIR
    )
    args = parser.parse_args()

    if not args.results_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.results_dir}")

    print(f"results dir: {args.results_dir}\n")

    all_problems = []
    assemblies = {}
    checked = 0

    for result_dir in sorted(args.results_dir.iterdir()):
        if not result_dir.is_dir():
            continue
        checked += 1
        all_problems.extend(check(result_dir))

        marker_path = next(result_dir.glob("**/job_marker.txt"), None)
        if marker_path is not None:
            digest = read_marker(marker_path).get("md5_assembly", "<absent>")
            assemblies.setdefault(digest, []).append(result_dir.name)

    print(f"checked {checked} directories\n")

    # Grouping by build makes a comparison that spans two of them obvious
    print("Unity script assembly checksums:")
    for digest, names in sorted(assemblies.items()):
        print(f"  {digest}  ({len(names)} runs)")
        for name in sorted(names):
            print(f"    {name}")
    if len([d for d in assemblies if d != "<absent>"]) > 1:
        print("\n  NOTE: more than one build across these results.")

    if all_problems:
        print(f"\n{len(all_problems)} problems:")
        for problem in all_problems:
            print(f"  {problem}")
    else:
        print("\nno problems found")


if __name__ == "__main__":
    main()