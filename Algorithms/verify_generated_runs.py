"""Verify a generated batch against its manifest before anything is uploaded.

Usage:
    python Algorithms\\verify_generated_runs.py Submissions\\duration_t120\\manifest_duration_ls_h.csv
    python Algorithms\\verify_generated_runs.py Submissions\\duration_t120\\manifest_*.csv
"""

import csv
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_DIR = PROJECT_ROOT / "Environments" / "DRLNavigation" / "Config"
SUBMISSIONS_DIR = PROJECT_ROOT / "Submissions"

# Torque caps the -N job name at 15 characters and PBS Pro allows more
# The generator guards at 15, so this repeats the guard independently
MAX_JOB_NAME = 15

# The duration at which generate_runs.py deliberately omits the key
# Omitting it reproduces every run made before the parameter existed
# A config at this duration must NOT carry platform_travel_time
DEFAULT_DURATION_S = 3.0

# Maps the environment code to the level_type the config must carry
LEVEL_TYPE = {"h": 6, "d": 5}

# Maps the architecture code to the YAML key holding its temporal window
# Frame stacking widens an observation stack
# The other 2 lengthen a sequence instead
WINDOW_KEY = {
    "bl": "observation_stacks",
    "ls": "sequence_length",
    "tf": "sequence_length",
}

# Unity rounds an odd worker id, so 2 odd ids can collide after rounding
# generate_runs.py steps by 2 to avoid that, so every id must be even
REQUIRE_EVEN_ENV_IDS = True


def read_text(path):
    """Read a file as text without translating its line endings."""
    return path.read_text(encoding="utf-8", errors="replace")


def yaml_value(text, key):
    """Return the value of a top-level-ish YAML key, or None if absent."""
    match = re.search(rf"^[ \t]*{re.escape(key)}:[ \t]*(\S+)[ \t]*$", text, re.M)
    return match.group(1) if match else None


def shell_value(text, pattern):
    """Return the first capture of a pattern in a shell script, or None."""
    match = re.search(pattern, text, re.M)
    return match.group(1) if match else None


def check_config(row, failures):
    """Check 1 config against its manifest row."""
    path = CONFIG_DIR / row["config"]
    label = row["config"]

    if not path.is_file():
        failures.append(f"{label}: config file does not exist")
        return

    text = read_text(path)
    arch = row["arch"]
    env = row["env"]

    # The temporal window, under whichever key this architecture uses
    key = WINDOW_KEY.get(arch)
    if key is None:
        failures.append(f"{label}: manifest arch '{arch}' is not bl, ls or tf")
    else:
        if row["window_key"] != key:
            failures.append(
                f"{label}: manifest window_key is '{row['window_key']}' "
                f"but arch {arch} uses '{key}'"
            )
        found = yaml_value(text, key)
        if found != row["window"]:
            failures.append(
                f"{label}: {key} is {found} but the manifest says {row['window']}"
            )

    # The duration, which the generator omits at the default and writes otherwise
    duration = float(row["duration_s"])
    found = yaml_value(text, "platform_travel_time")
    if abs(duration - DEFAULT_DURATION_S) < 1e-9:
        if found is not None:
            failures.append(
                f"{label}: platform_travel_time is {found} but must be absent "
                f"at the {DEFAULT_DURATION_S} second default"
            )
    else:
        if found is None:
            failures.append(
                f"{label}: platform_travel_time is absent but the manifest "
                f"says {duration}"
            )
        elif abs(float(found) - duration) > 1e-9:
            failures.append(
                f"{label}: platform_travel_time is {found} but the manifest "
                f"says {duration}"
            )

    # The environment, which decides which level the run loads
    expected = LEVEL_TYPE.get(env)
    if expected is None:
        failures.append(f"{label}: manifest env '{env}' is not h or d")
    else:
        found = yaml_value(text, "level_type")
        if found != str(expected):
            failures.append(
                f"{label}: level_type is {found} but env {env} needs {expected}"
            )
        if row["level_type"] != str(expected):
            failures.append(
                f"{label}: manifest level_type is {row['level_type']} "
                f"but env {env} needs {expected}"
            )

    # Evaluation must stay off, because enabling it crashes on test_lengths
    found = yaml_value(text, "num_test_eps")
    if found != "0":
        failures.append(f"{label}: num_test_eps is {found} and must be 0")

    # A missing seed means every seed index trains the same run
    if yaml_value(text, "seed") is None:
        failures.append(f"{label}: no seed key found")

    # Euclidean failure is disabled by design across this whole project
    found = yaml_value(text, "euclidean_step_limit")
    if found != "0":
        failures.append(
            f"{label}: euclidean_step_limit is {found} and every other run "
            f"in this project uses 0"
        )


def check_submit(row, failures, warnings):
    """Check 1 submission script against its manifest row."""
    path = Path(row["_submit_path"])
    label = path.name

    if not path.is_file():
        failures.append(f"{label}: submission script does not exist")
        return

    text = read_text(path)

    # Unix line endings, because PBS will not run a script with CRLF
    if b"\r\n" in path.read_bytes():
        failures.append(f"{label}: contains CRLF line endings")

    # The job name, which is how every result directory is named
    found = shell_value(text, r"^#PBS -N[ \t]+(\S+)[ \t]*$")
    if found != row["job_name"]:
        failures.append(
            f"{label}: -N is '{found}' but the manifest says '{row['job_name']}'"
        )
    if len(row["job_name"]) > MAX_JOB_NAME:
        failures.append(
            f"{label}: job name '{row['job_name']}' is "
            f"{len(row['job_name'])} characters and the limit is {MAX_JOB_NAME}"
        )

    # The config this script will actually train with
    found = shell_value(text, r'^CONFIG="([^"]+)"')
    if found != row["config"]:
        failures.append(
            f"{label}: CONFIG is '{found}' but the manifest says '{row['config']}'"
        )

    # The Unity worker offset, which 2 concurrent jobs must never share
    found = shell_value(text, r"^ENV_ID=(\d+)")
    if found != row["env_id"]:
        failures.append(
            f"{label}: ENV_ID is {found} but the manifest says {row['env_id']}"
        )
    if REQUIRE_EVEN_ENV_IDS and found is not None and int(found) % 2 != 0:
        failures.append(
            f"{label}: ENV_ID {found} is odd and Unity rounds odd ids, "
            f"so 2 odd ids can collide after rounding"
        )

    # Walltime, where a cell killed at the cap leaves no results.npy
    found = shell_value(text, r"^#PBS -l walltime=(\S+)[ \t]*$")
    if found != row["walltime"]:
        failures.append(
            f"{label}: walltime is {found} but the manifest says {row['walltime']}"
        )

    # Memory, matched case-insensitively because the unit suffix varies
    found = shell_value(text, r"^#PBS -l mem=(\S+)[ \t]*$")
    if found is None:
        failures.append(f"{label}: no mem request found")
    else:
        digits = re.sub(r"[^0-9]", "", found)
        if digits != row["mem_gb"]:
            failures.append(
                f"{label}: mem is {found} but the manifest says {row['mem_gb']}GB"
            )

    # Every script in this tree has always requested 12 cpus
    # A different value would be a regression rather than a choice
    found = shell_value(text, r"^#PBS -l ncpus=(\d+)[ \t]*$")
    if found is not None and found != "12":
        warnings.append(
            f"{label}: ncpus is {found} and every other script requests 12"
        )

    # The provenance block is what lets a result be traced to its config
    if "record_field()" not in text:
        failures.append(f"{label}: missing the record_field provenance block")


def check_batch_internals(rows, failures):
    """Check the batch against itself for duplicate identifiers."""
    for field, human in (
        ("env_id", "environment id"),
        ("job_name", "job name"),
        ("config", "config name"),
    ):
        seen = {}
        for row in rows:
            seen.setdefault(row[field], []).append(row["job_name"])
        for value, owners in seen.items():
            if len(owners) > 1:
                failures.append(
                    f"duplicate {human} {value} shared by: {', '.join(owners)}"
                )


def check_other_batches(rows, warnings):
    """Warn when an env id in this batch is also held outside it."""
    mine = {Path(row["_submit_path"]).resolve() for row in rows}
    wanted = {row["env_id"]: row["job_name"] for row in rows}

    if not SUBMISSIONS_DIR.is_dir():
        warnings.append(f"{SUBMISSIONS_DIR} not found, cross-batch scan skipped")
        return

    for path in sorted(SUBMISSIONS_DIR.rglob("*.sh")):
        if path.resolve() in mine:
            continue
        try:
            found = shell_value(read_text(path), r"^ENV_ID=(\d+)")
        except OSError:
            continue
        if found in wanted:
            warnings.append(
                f"env id {found} is also in {path.name}, which this batch "
                f"assigns to {wanted[found]}. Safe only if that job will "
                f"never run at the same time"
            )


def main():
    paths = [Path(a) for a in sys.argv[1:]]
    if not paths:
        raise SystemExit(
            "Pass 1 or more manifest CSV paths written by generate_runs.py."
        )

    rows = []
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"Manifest not found: {path}")
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                # The scripts sit beside their own manifest
                row["_submit_path"] = str(path.parent / f"submit_{row['job_name']}.sh")
                row["_manifest"] = path.name
                rows.append(row)

    if not rows:
        raise SystemExit("Manifests hold no rows.")

    failures = []
    warnings = []

    for row in rows:
        check_config(row, failures)
        check_submit(row, failures, warnings)

    check_batch_internals(rows, failures)
    check_other_batches(rows, warnings)

    print(f"{len(rows)} cells from {len(paths)} manifest(s)")
    print()
    header = f"{'job name':<16} {'config':<30} {'env':>5} {'mem':>5} {'walltime':>9}"
    print(header)
    print("-" * len(header))
    for row in sorted(rows, key=lambda r: (r["arch"], r["job_name"])):
        print(
            f"{row['job_name']:<16} {row['config']:<30} {row['env_id']:>5} "
            f"{row['mem_gb']:>5} {row['walltime']:>9}"
        )

    if warnings:
        print()
        print("WARNINGS, which do not block submission:")
        for note in warnings:
            print(f"  {note}")

    print()
    if failures:
        print(f"FAILED, {len(failures)} problem(s). Do not upload:")
        for note in failures:
            print(f"  {note}")
        return 1

    print("VERIFIED. Every config and script matches its manifest row.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
