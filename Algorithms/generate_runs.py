"""Generate configs, submission scripts and a manifest for one sweep.

Each config differs from its source only in the lines the sweep varies.
Each submission script differs only in its PBS header and per-run settings.
Lives in Algorithms/ and resolves project paths from its own location.
Runs on Windows; the submission scripts come out with Unix line endings.
"""

import argparse
import csv
from pathlib import Path

# ===== SWEEP DEFINITION =====
# SWEEP: "window" or "duration"
# ARCH: "bl" (baseline), "ls" (LSTM) or "tf" (Transformer)
# ENV: "h" for level_type 6, "d" for level_type 5
# SEED_INDICES: Seed indices to generate configs for
# WINDOWS: Window sizes in decision steps
# DURATIONS: Cycle durations (in deciseconds)
# ENV_ID_BASE: First Unity worker offset; concurrent jobs need distinct values to avoid port conflicts
# JOB_NAME_BOTH_FIELDS: Put both window and duration in PBS job name; needed to avoid collisions when a batch varies both


SWEEP = "duration"
ARCH = "bl"
ENV = "h"
SEED_INDICES = (3, 4)
WINDOWS = (2, 8, 10, 16) if SWEEP == "window" else ((4,) if ARCH == "bl" else (10,))
DURATIONS = (70,)
ENV_ID_BASE = 920
JOB_NAME_BOTH_FIELDS = False
# ===== END SWEEP DEFINITION =====

# Everything below is relative to the repository root, one level up
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_CONFIG_DIR = PROJECT_ROOT / "Environments" / "DRLNavigation" / "Config"
DEFAULT_SUBMIT_DIR = PROJECT_ROOT / "Submissions" / "duration_sweep"
DEFAULT_BODY_FILE = SCRIPT_DIR / "submit_body.sh"

# Source config each architecture is generated from
SOURCE_BY_ARCH = {
    "bl": "sac_baseline_moving_platform_{env_word}_seed{seed}.yaml",
    "ls": "sac_lstm_moving_platform_{env_word}_seed{seed}.yaml",
    "tf": "sac_transformer_moving_platform_{env_word}_seed{seed}.yaml",
}

ENV_WORD = {"h": "horizontal", "d": "depthwise"}

# The key holding the temporal window differs by architecture
# bl stacks observations; ls and tf use a sequence length instead
WINDOW_KEY = {
    "bl": "observation_stacks",
    "ls": "sequence_length",
    "tf": "sequence_length",
}

# Indentation differs because sequence_length sits inside the memory block
WINDOW_INDENT = {"bl": "    ", "ls": "        ", "tf": "        "}

# Walltime per architecture, matching the existing submit scripts
# Measured: bl 3.2-7.3h, ls 39.4-42.7h, tf 43.5-46.9h
WALLTIME = {"bl": "08:00:00", "ls": "48:00:00", "tf": "48:00:00"}

# Windows at or above this get more walltime and more memory
# Stacking adds channels to the first conv of both encoders
# It also scales every stored observation, so 16 stacks is ~4x the buffer of 4
LARGE_WINDOW_THRESHOLD = 16
LARGE_WINDOW_WALLTIME = "09:00:00"

BASE_MEM_GB = {"bl": 32, "ls": 32, "tf": 32}
LARGE_WINDOW_EXTRA_GB = 16

NOTIFY_EMAIL = "adam.cameron@student.uts.edu.au"

DURATION_KEY = "platform_travel_time"


SUBMIT_TEMPLATE = """#!/bin/bash

#PBS -N {job}
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem={mem}GB
#PBS -l walltime={walltime}
#PBS -q small_gpuq
#PBS -m abe
#PBS -M {email}

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# ===== PER-RUN SETTINGS =====
# These two lines and "#PBS -N" above are all that differ between submission scripts
# CONFIG drives both the run and the marker file, so the two cannot drift
CONFIG="{config}"
ENV_ID={env_id}

"""


def read_keeping_newlines(path):
    """Read a file without translating its line endings."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def write_keeping_newlines(path, text):
    """Write a file without translating its line endings.

    Used for configs, so a generated one stays byte-identical to its source
    apart from the lines this script deliberately changes.
    """
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def write_unix_endings(path, text):
    """Write a file with Unix line endings, whatever the input had.

    Used for submission scripts, which run on Linux.
    A stray carriage return there breaks every command in the file.
    This is what removes the need to run dos2unix after transferring.
    """
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(normalised)


def run_name(arch, env, window, duration, seed_index):
    """Full run name, used for config filenames and the manifest."""
    # Fixed-width fields, so the name can be parsed by split or by slice
    return f"{arch}_{env}_w{window:02d}_t{duration:03d}_s{seed_index}"


def job_name(arch, env, window, duration, seed_index):
    """PBS job name, kept short.

    PBS_JOBNAME ends up in the results directory name and Torque caps -N at 15.
    Only the field this sweep varies goes in here.
    The rest is in the config filename and in job_marker.txt.
    """
    if JOB_NAME_BOTH_FIELDS:
        varying = f"w{window:02d}t{duration:03d}"
    else:
        varying = f"w{window:02d}" if SWEEP == "window" else f"t{duration:03d}"
    name = f"{arch}_{env}_{varying}_s{seed_index}"
    if len(name) > 15:
        raise SystemExit(f"Job name too long for PBS: {name}")
    return name


def config_name(name):
    """Config filename for a full run name."""
    return f"sac_{name}.yaml"


def memory_gb(arch, window):
    """Memory request, allowing for the replay buffer growing with the window."""
    # Only frame stacking is affected
    # ls and tf store one observation per transition whatever the window
    if arch == "bl" and window >= LARGE_WINDOW_THRESHOLD:
        return BASE_MEM_GB[arch] + LARGE_WINDOW_EXTRA_GB
    return BASE_MEM_GB[arch]


def walltime_for(arch, window):
    """Walltime request, allowing for wider windows running slower."""
    if arch == "bl" and window >= LARGE_WINDOW_THRESHOLD:
        return LARGE_WINDOW_WALLTIME
    return WALLTIME[arch]


def current_value(text, indent, key, source_label):
    """Read a key's current value, so the line to replace matches exactly."""
    needle = f"{indent}{key}:"
    for line in text.splitlines():
        if line.startswith(needle):
            return line[len(needle):].strip()
    raise SystemExit(
        f"{source_label}: key {key!r} not found at indent {len(indent)}"
    )


def replace_unique(text, old, new, label, source_label):
    """Replace a line that must appear exactly once."""
    occurrences = text.count(old)
    if occurrences != 1:
        raise SystemExit(
            f"{source_label}: expected 1 {label} line matching {old!r}, "
            f"found {occurrences}"
        )
    return text.replace(old, new)


def build_config(config_dir, arch, env, window, duration, seed_index,
                 name, dry_run):
    """Write one config, changing only the window and duration lines."""
    source_name = SOURCE_BY_ARCH[arch].format(
        env_word=ENV_WORD[env], seed=seed_index
    )
    source_path = config_dir / source_name

    if not source_path.is_file():
        raise SystemExit(f"Source config not found: {source_path}")

    text = read_keeping_newlines(source_path)

    # Rewrite the window under whichever key this architecture uses
    indent = WINDOW_INDENT[arch]
    key = WINDOW_KEY[arch]
    existing = current_value(text, indent, key, source_name)
    text = replace_unique(
        text,
        f"{indent}{key}: {existing}",
        f"{indent}{key}: {window}",
        key,
        source_name,
    )

    # Set the cycle duration
    # The source configs have no platform_travel_time key at all
    # It was only exposed in the new Unity build, and the old build ignores it
    # So it is appended only when a non-default duration is asked for
    duration_seconds = duration / 10.0
    if f"    {DURATION_KEY}:" in text:
        existing_duration = current_value(
            text, "    ", DURATION_KEY, source_name
        )
        text = replace_unique(
            text,
            f"    {DURATION_KEY}: {existing_duration}",
            f"    {DURATION_KEY}: {duration_seconds}",
            DURATION_KEY,
            source_name,
        )
    elif duration != 30:
        newline = "\r\n" if "\r\n" in text else "\n"
        text = text.rstrip() + newline + newline
        text += f"    {DURATION_KEY}: {duration_seconds}{newline}"

    target_path = config_dir / config_name(name)

    # Never clobber an existing config
    # A collision means this script has already been run
    if target_path.exists():
        raise SystemExit(f"Refusing to overwrite: {target_path}")

    if dry_run:
        print(f"would write {target_path.name}")
    else:
        write_keeping_newlines(target_path, text)
        print(f"wrote {target_path.name}")

    return target_path


def build_submit(submit_dir, job, config, env_id, mem, walltime, body, dry_run):
    """Write one submission script: generated header plus the shared body."""
    text = SUBMIT_TEMPLATE.format(
        job=job,
        mem=mem,
        walltime=walltime,
        email=NOTIFY_EMAIL,
        config=config,
        env_id=env_id,
    ) + body

    target_path = submit_dir / f"submit_{job}.sh"

    if target_path.exists():
        raise SystemExit(f"Refusing to overwrite: {target_path}")

    if dry_run:
        print(f"would write {target_path.name} "
              f"(env_id {env_id}, mem {mem}GB, walltime {walltime})")
    else:
        write_unix_endings(target_path, text)
        print(f"wrote {target_path.name} "
              f"(env_id {env_id}, mem {mem}GB, walltime {walltime})")

    return target_path


def verify_configs(paths, arch):
    """Re-read every written config and check the fields that matter."""
    key = WINDOW_KEY[arch]
    indent = WINDOW_INDENT[arch]

    for path in paths:
        text = read_keeping_newlines(path)

        # Take the window size back out of the filename to check against
        expected = int(path.name.split("_w")[1].split("_")[0])

        checks = {
            f"{indent}{key}: {expected}": key,
            "    num_test_eps: 0": "num_test_eps",
        }

        for line, label in checks.items():
            if text.count(line) != 1:
                raise SystemExit(f"{path.name}: {label} check failed")

        print(f"verified {path.name}")


def verify_submits(paths):
    """Confirm every submission script came out with Unix line endings."""
    # A carriage return in a bash script on Linux breaks every line it touches
    for path in paths:
        if b"\r" in path.read_bytes():
            raise SystemExit(f"{path.name}: contains carriage returns")
        print(f"verified {path.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--submit-dir", type=Path, default=DEFAULT_SUBMIT_DIR)
    parser.add_argument("--body", type=Path, default=DEFAULT_BODY_FILE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.config_dir.is_dir():
        raise SystemExit(f"Config directory not found: {args.config_dir}")

    if not args.body.is_file():
        raise SystemExit(f"Body file not found: {args.body}")

    # The output directory is created rather than demanded
    # Nothing is lost if it already holds a previous batch, because every individual write below refuses to overwrite an existing file
    if not args.dry_run:
        args.submit_dir.mkdir(parents=True, exist_ok=True)

    print(f"config dir : {args.config_dir}")
    print(f"submit dir : {args.submit_dir}")
    print(f"body file  : {args.body}\n")

    body = read_keeping_newlines(args.body)

    # The body has to start where the generated header stops
    if not body.lstrip().startswith("SCRATCH="):
        raise SystemExit("Body file does not begin with the SCRATCH= line")

    # Without the provenance block the markers come out too thin to verify
    if "record_field()" not in body:
        raise SystemExit("Body file is missing the provenance block")

    rows = []
    config_paths = []
    submit_paths = []
    index = 0

    for window in WINDOWS:
        for duration in DURATIONS:
            for seed_index in SEED_INDICES:
                full = run_name(ARCH, ENV, window, duration, seed_index)
                job = job_name(ARCH, ENV, window, duration, seed_index)


                env_id = ENV_ID_BASE + index * 2
                index += 1

                config_paths.append(build_config(
                    args.config_dir, ARCH, ENV, window, duration,
                    seed_index, full, args.dry_run,
                ))

                mem = memory_gb(ARCH, window)
                walltime = walltime_for(ARCH, window)

                submit_paths.append(build_submit(
                    args.submit_dir, job, config_name(full), env_id,
                    mem, walltime, body, args.dry_run,
                ))

                rows.append({
                    "job_name": job,
                    "run_name": full,
                    "config": config_name(full),
                    "arch": ARCH,
                    "env": ENV,
                    "level_type": 6 if ENV == "h" else 5,
                    "window": window,
                    "window_key": WINDOW_KEY[ARCH],
                    "duration_s": duration / 10.0,
                    "seed_index": seed_index,
                    "env_id": env_id,
                    "mem_gb": mem,
                    "walltime": walltime,
                })

    if not args.dry_run:
        verify_configs(config_paths, ARCH)
        verify_submits(submit_paths)

    manifest_path = args.submit_dir / f"manifest_{SWEEP}_{ARCH}_{ENV}.csv"

    if args.dry_run:
        print(f"\nwould write {manifest_path.name}")
    else:
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {manifest_path.name}")

    print(f"{len(rows)} cells")
    print(f"env_id assigned: {ENV_ID_BASE} to {ENV_ID_BASE + (len(rows) - 1) * 2}")
    print(f"env_id reserved: {ENV_ID_BASE} to {ENV_ID_BASE + len(rows) * 2 - 1}")


if __name__ == "__main__":
    main()