"""Compare results.npy files produced by Utils.save_data.

Two runs with the same seed, config and architecture produce bit-identical
metric arrays. Only wall-clock timing differs. So any metric difference means
the computation differed, and metric identity means it did not.

Usage:
    python3 -I compare_results.py LABEL=path.npy LABEL=path.npy [...]

Exit code 0 always; read the report.
"""
import itertools
import sys
import numpy as np

# Layout of save_data's list, from OffPolicyMain.main()
# (group index, group name, [series names])
LAYOUT = [
    (0, "step",          ["steps_so_far", "time_so_far"]),
    (1, "return",        ["avg_ep_return", "test_returns"]),
    (2, "loss",          ["actor_loss", "critic_loss", "alpha_loss"]),
    (3, "value",         ["v", "q", "log_p", "entropy", "alpha"]),
    (4, "length",        ["avg_ep_length", "test_lengths"]),
    (5, "side_stats",    ["success_rate", "fail_rate",
                          "euclidean_fail_rate", "timeout_fail_rate"]),
    (6, "goal_distance", ["max_goal_distance", "avg_goal_distance"]),
]

# Wall-clock time is not comparable between runs, so it is excluded
TIMING = {("step", "time_so_far")}


def flatten(path):
    """Return {(group, series): ndarray} for one results.npy."""
    data = np.load(path, allow_pickle=True)
    out = {}
    for gi, gname, series in LAYOUT:
        if gi >= len(data):
            continue
        group = data[gi]
        for si, sname in enumerate(series):
            if si < len(group):
                out[(gname, sname)] = np.asarray(group[si])
    return out


def compare(a, b):
    """Return (verdict, detail) for two arrays."""
    if a.shape != b.shape:
        return "SHAPE", f"{a.shape} vs {b.shape}"
    if np.array_equal(a, b):
        return "same", ""
    if a.dtype.kind == "f" or b.dtype.kind == "f":
        af, bf = a.astype(float), b.astype(float)
        both_nan = np.isnan(af) & np.isnan(bf)
        diff = np.abs(af - bf)
        diff[both_nan] = 0.0
        if not np.any(diff > 0):
            return "same", "(differs only in NaN placement)"
        n = int(np.count_nonzero(diff > 0))
        first = int(np.argmax(diff > 0))
        return "DIFFER", (f"max |d| {np.nanmax(diff):.6g}, "
                          f"{n}/{diff.size} points, first at index {first}")
    n = int(np.count_nonzero(a != b))
    return "DIFFER", f"{n}/{a.size} points"


def main(argv):
    runs = {}
    for arg in argv:
        if "=" not in arg:
            sys.exit(f"expected LABEL=path, got {arg!r}")
        label, path = arg.split("=", 1)
        runs[label] = flatten(path)
        print(f"loaded {label:<10} {path.rsplit('/', 1)[-1]}  "
              f"({len(runs[label])} series)")
    print()

    keys = [k for k in runs[next(iter(runs))] if k not in TIMING]

    for a, b in itertools.combinations(runs, 2):
        print("=" * 78)
        print(f"{a}  vs  {b}")
        print("=" * 78)
        same = differ = 0
        for key in keys:
            if key not in runs[a] or key not in runs[b]:
                print(f"  MISSING  {key[0]}.{key[1]}")
                continue
            verdict, detail = compare(runs[a][key], runs[b][key])
            if verdict == "same":
                same += 1
            else:
                differ += 1
            if verdict != "same" or detail:
                print(f"  {verdict:<7} {key[0]}.{key[1]:<22} {detail}")
        print(f"\n  {same} series identical, {differ} series differ, "
              f"out of {same + differ} compared (timing excluded)")
        if differ == 0:
            print("  VERDICT: these two runs performed the SAME computation.")
        else:
            print("  VERDICT: these two runs performed DIFFERENT computations.")
        print()

    # timing, reported separately and never part of a verdict
    print("=" * 78)
    print("wall-clock timing (informational only)")
    print("=" * 78)
    for label, run in runs.items():
        t = run.get(("step", "time_so_far"))
        s = run.get(("step", "steps_so_far"))
        if t is None:
            continue
        total = float(np.nanmax(t))
        steps = int(np.nanmax(s)) if s is not None else 0
        rate = steps / total if total else float("nan")
        print(f"  {label:<10} {total:>12.0f} s  ({total / 3600:5.2f} h)  "
              f"{steps:>9} steps  {rate:7.1f} steps/s")


if __name__ == "__main__":
    main(sys.argv[1:])
