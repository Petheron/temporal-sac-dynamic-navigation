"""Check the local smoke test logs against what each config should produce.

Usage:
    python Algorithms\\check_smoke_logs.py smoke_logs
"""
import os
import re
import sys

# Keyed by the number that each log filename starts with
# Values are the [POLICY SELECTION] fields that run must report
# travel_time and randomise_phase are optional and only round 3 sets them
EXPECTED = {
    "01": {"label": "baseline dispatch probe, moving platform",
           "Policy class": "SAC", "use_transformer": "False",
           "use_lstm": "False", "use_temporal": "False",
           "sequence_length": "0", "memory_size": "0",
           "burn_in_length": "0", "level_type": "5"},
    "02": {"label": "LSTM ablation, burn_in_length 10",
           "Policy class": "SAC_LSTM", "use_transformer": "False",
           "use_lstm": "True", "use_temporal": "True",
           "sequence_length": "10", "memory_size": "512",
           "burn_in_length": "10", "level_type": "5"},
    "03": {"label": "LSTM ablation, memory_size 128",
           "Policy class": "SAC_LSTM", "use_transformer": "False",
           "use_lstm": "True", "use_temporal": "True",
           "sequence_length": "10", "memory_size": "128",
           "burn_in_length": "0", "level_type": "5"},
    "04": {"label": "LSTM ablation, sequence_length 20",
           "Policy class": "SAC_LSTM", "use_transformer": "False",
           "use_lstm": "True", "use_temporal": "True",
           "sequence_length": "20", "memory_size": "512",
           "burn_in_length": "0", "level_type": "5"},
    "05": {"label": "LSTM on NavigationLevel",
           "Policy class": "SAC_LSTM", "use_transformer": "False",
           "use_lstm": "True", "use_temporal": "True",
           "sequence_length": "10", "memory_size": "512",
           "burn_in_length": "0", "level_type": "0"},
    "06": {"label": "LSTM on WindingLevel",
           "Policy class": "SAC_LSTM", "use_transformer": "False",
           "use_lstm": "True", "use_temporal": "True",
           "sequence_length": "10", "memory_size": "512",
           "burn_in_length": "0", "level_type": "1"},
    "07": {"label": "LSTM on TowerLevel",
           "Policy class": "SAC_LSTM", "use_transformer": "False",
           "use_lstm": "True", "use_temporal": "True",
           "sequence_length": "10", "memory_size": "512",
           "burn_in_length": "0", "level_type": "2"},
    "08": {"label": "baseline on NavigationLevel",
           "Policy class": "SAC", "use_transformer": "False",
           "use_lstm": "False", "use_temporal": "False",
           "sequence_length": "0", "memory_size": "0",
           "burn_in_length": "0", "level_type": "0"},
    "09": {"label": "Transformer on NavigationLevel",
           "Policy class": "SAC_GTrXL", "use_transformer": "True",
           "use_lstm": "False", "use_temporal": "True",
           "sequence_length": "10", "memory_size": "512",
           "burn_in_length": "0", "level_type": "0"},

    # Round 2 times every ablation value against an unmodified reference
    # Round 1 showed the recurrent update dominates run time on level 5
    "10": {"label": "LSTM reference, unmodified production config",
           "Policy class": "SAC_LSTM", "use_transformer": "False",
           "use_lstm": "True", "use_temporal": "True",
           "sequence_length": "10", "memory_size": "512",
           "burn_in_length": "0", "level_type": "5"},
    "11": {"label": "LSTM ablation, sequence_length 5",
           "Policy class": "SAC_LSTM", "use_transformer": "False",
           "use_lstm": "True", "use_temporal": "True",
           "sequence_length": "5", "memory_size": "512",
           "burn_in_length": "0", "level_type": "5"},
    "12": {"label": "LSTM ablation, sequence_length 15",
           "Policy class": "SAC_LSTM", "use_transformer": "False",
           "use_lstm": "True", "use_temporal": "True",
           "sequence_length": "15", "memory_size": "512",
           "burn_in_length": "0", "level_type": "5"},
    "13": {"label": "LSTM ablation, memory_size 256",
           "Policy class": "SAC_LSTM", "use_transformer": "False",
           "use_lstm": "True", "use_temporal": "True",
           "sequence_length": "10", "memory_size": "256",
           "burn_in_length": "0", "level_type": "5"},
    "14": {"label": "LSTM ablation, burn_in_length 5",
           "Policy class": "SAC_LSTM", "use_transformer": "False",
           "use_lstm": "True", "use_temporal": "True",
           "sequence_length": "10", "memory_size": "512",
           "burn_in_length": "5", "level_type": "5"},

    # Round 3 runs the baseline on the horizontal moving platform at level_type 6
    # The policy fields are identical across all 7 and only the platform duration differs
    # travel_time is the value the [PLATFORM] line must report, and it is the only evidence that Unity applied the parameter rather than ignoring it
    "20": {"label": "platform default, no parameter present",
           "Policy class": "SAC", "use_transformer": "False",
           "use_lstm": "False", "use_temporal": "False",
           "sequence_length": "0", "memory_size": "0",
           "burn_in_length": "0", "level_type": "6",
           "travel_time": 3.0, "randomise_phase": "False"},
    "21": {"label": "platform 3.0 s set explicitly",
           "Policy class": "SAC", "use_transformer": "False",
           "use_lstm": "False", "use_temporal": "False",
           "sequence_length": "0", "memory_size": "0",
           "burn_in_length": "0", "level_type": "6",
           "travel_time": 3.0, "randomise_phase": "False"},
    "22": {"label": "platform 5.0 s",
           "Policy class": "SAC", "use_transformer": "False",
           "use_lstm": "False", "use_temporal": "False",
           "sequence_length": "0", "memory_size": "0",
           "burn_in_length": "0", "level_type": "6",
           "travel_time": 5.0, "randomise_phase": "False"},
    "23": {"label": "platform 10.0 s",
           "Policy class": "SAC", "use_transformer": "False",
           "use_lstm": "False", "use_temporal": "False",
           "sequence_length": "0", "memory_size": "0",
           "burn_in_length": "0", "level_type": "6",
           "travel_time": 10.0, "randomise_phase": "False"},
    "24": {"label": "platform 3.0 s with phase randomisation on",
           "Policy class": "SAC", "use_transformer": "False",
           "use_lstm": "False", "use_temporal": "False",
           "sequence_length": "0", "memory_size": "0",
           "burn_in_length": "0", "level_type": "6",
           "travel_time": 3.0, "randomise_phase": "True"},
    "25": {"label": "platform default with num_test_eps 2",
           "Policy class": "SAC", "use_transformer": "False",
           "use_lstm": "False", "use_temporal": "False",
           "sequence_length": "0", "memory_size": "0",
           "burn_in_length": "0", "level_type": "6",
           "travel_time": 3.0, "randomise_phase": "False"},
    "26": {"label": "platform default on the preserved old Windows build",
           "Policy class": "SAC", "use_transformer": "False",
           "use_lstm": "False", "use_temporal": "False",
           "sequence_length": "0", "memory_size": "0",
           "burn_in_length": "0", "level_type": "6"},
}

# Round one is logs 01 to 09, round two is 10 to 14, round three is 20 to 26
# A folder holding only one round is complete in its own right, so the summary counts whichever rounds are present rather than demanding all of them
ROUNDS = {"round 1, code paths": [f"{n:02d}" for n in range(1, 10)],
          "round 2, throughput": [f"{n:02d}" for n in range(10, 15)],
          "round 3, platform duration": [f"{n:02d}" for n in range(20, 27)]}

# level_type is not printed on its own line, only inside the full env_params dictionary, so it is pulled out of that line with a regex
LEVEL_RE = re.compile(r"'level_type':\s*([0-9]+)")

# The [PLATFORM] line is printed once per process by MovingPlatform.cs
# Its absence means the build does not contain the parameter read at all
PLATFORM_RE = re.compile(r"\[PLATFORM\][^\n]*travelTime=([0-9.]+)")
TIMESTEP_RE = re.compile(r"\[PLATFORM\][^\n]*fixedDeltaTime=([0-9.]+)")
STEPS_RE = re.compile(r"\[PLATFORM\][^\n]*physicsStepsPerTraverse=([0-9]+)")
PHASE_RE = re.compile(r"\[PLATFORM\][^\n]*randomisePhase=(\w+)")


def banner_value(text, field):
    """Read one [POLICY SELECTION] field, or None if the banner never printed.

    Some fields are printed with a type annotation after the value, as in
    "sequence_length : 10 (type int)", so anything from " (type" onward is
    dropped before the value is compared.
    """
    pattern = re.compile(r"\[POLICY SELECTION\]\s+" + re.escape(field) + r"\s*:\s*(.+)")
    match = pattern.search(text)
    if match is None:
        return None
    return match.group(1).split(" (type")[0].strip().strip("'\"")


def check_platform(text, want, out):
    """Check the [PLATFORM] line for a run whose expectation sets a duration."""
    # Only round 3 carries a travel_time, so earlier rounds skip this entirely
    if "travel_time" not in want:
        return

    match = PLATFORM_RE.search(text)
    if match is None:
        out.append(("FAIL", "no [PLATFORM] line, so the parameter was never "
                            "read and this build does not contain the edit"))
        return

    got = float(match.group(1))
    if abs(got - want["travel_time"]) > 1e-6:
        out.append(("FAIL", f"travelTime: got {got:.3f}, expected "
                            f"{want['travel_time']:.3f}, so the parameter was "
                            f"sent but not applied"))
    else:
        out.append(("ok", f"travelTime: {got:.3f}"))

    # The expected step count depends on the fixed timestep, so it is set here
    timestep = TIMESTEP_RE.search(text)
    steps = STEPS_RE.search(text)
    if timestep and steps:
        out.append(("ok", f"fixedDeltaTime: {timestep.group(1)} s, "
                          f"physicsStepsPerTraverse: {steps.group(1)}"))
    else:
        out.append(("INFO", "the [PLATFORM] line is missing its timestep or "
                            "step count fields"))

    if "randomise_phase" in want:
        phase = PHASE_RE.search(text)
        if phase is None:
            out.append(("INFO", "randomisePhase not found on the [PLATFORM] line"))
        elif phase.group(1) != want["randomise_phase"]:
            out.append(("FAIL", f"randomisePhase: got {phase.group(1)}, "
                                f"expected {want['randomise_phase']}"))
        else:
            out.append(("ok", f"randomisePhase: {phase.group(1)}"))


def check(path, want):
    """Return a list of (level, message) for one log."""
    text = open(path, encoding="utf-8", errors="replace").read()
    out = []

    if "[POLICY SELECTION]" not in text:
        out.append(("FAIL", "no [POLICY SELECTION] banner, so the run died before "
                            "the policy was chosen"))

    for field in ("Policy class", "use_transformer", "use_lstm", "use_temporal",
                  "sequence_length", "memory_size", "burn_in_length"):
        got = banner_value(text, field)
        if got is None:
            out.append(("FAIL", f"{field}: not found in the log"))
        elif got != want[field]:
            out.append(("FAIL", f"{field}: got {got}, expected {want[field]}"))
        else:
            out.append(("ok", f"{field}: {got}"))

    level = LEVEL_RE.search(text)
    if level is None:
        out.append(("INFO", "level_type not found in the env_params line"))
    elif level.group(1) != want["level_type"]:
        out.append(("FAIL", f"level_type: got {level.group(1)}, "
                            f"expected {want['level_type']}"))
    else:
        out.append(("ok", f"level_type: {level.group(1)}"))

    check_platform(text, want, out)

    if "Traceback" in text:
        first = text.split("Traceback", 1)[1].strip().splitlines()
        out.append(("FAIL", "Traceback present: " +
                    (first[-1] if first else "see the log")))
    else:
        out.append(("ok", "no traceback"))

    saves = len(re.findall(r"model saved", text, re.IGNORECASE))
    out.append(("ok" if saves >= 3 else "INFO",
                f"checkpoint saves found: {saves} (3 expected at 3000 steps)"))

    closed = re.search(r"environment closing", text, re.IGNORECASE) is not None
    out.append(("ok" if closed else "INFO",
                "environment closed cleanly" if closed
                else "no environment-closing message found"))

    return out


def main(folder):
    folder = os.path.abspath(folder)
    logs = sorted(f for f in os.listdir(folder) if f.lower().endswith(".log"))
    if not logs:
        sys.exit(f"no .log files in {folder}")

    present = {n[:2] for n in logs}
    rounds = [r for r, keys in ROUNDS.items() if present & set(keys)]

    print(f"smoke logs folder : {folder}")
    print(f"logs found        : {len(logs)}")
    print(f"rounds present    : {', '.join(rounds) if rounds else 'none recognised'}\n")

    fails = 0
    skipped = 0
    for name in logs:
        key = name[:2]
        want = EXPECTED.get(key)
        print("=" * 72)
        if want is None:
            # Counted as a problem below rather than silently skipped
            print(f"{name}  -  NOT CHECKED, no EXPECTED entry for prefix {key!r}")
            print()
            skipped += 1
            continue
        print(f"{name}  -  {want['label']}")
        print("=" * 72)
        for level, message in check(os.path.join(folder, name), want):
            print(f"  {level:<4}  {message}")
            fails += (level == "FAIL")
        print()

    # Only complain about logs missing from a round that has started
    missing = []
    for keys in ROUNDS.values():
        if present & set(keys):
            missing += [k for k in keys if k not in present]
    missing.sort()
    if missing:
        print(f"logs not present for: {', '.join(missing)}")

    print("-" * 72)

    problems = []
    if fails:
        problems.append(f"{fails} check(s) FAILED")
    if skipped:
        problems.append(f"{skipped} log(s) not checked, no EXPECTED entry")
    if missing:
        problems.append(f"{len(missing)} log(s) missing from a round that has started")

    if not problems:
        print(f"  ALL {len(logs)} SMOKE TESTS PASSED")
    else:
        print("  " + "; ".join(problems) + ".")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "smoke_logs")
