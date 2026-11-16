"""Recompute every statistic reported in Chapter 3 from the collected metric series.

Usage:
    python Algorithms\\analyse_statistics.py --results Metrics
    python Algorithms\\analyse_statistics.py --only stack
"""

import argparse
import hashlib
import itertools
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

# ===== SELECTION RULES, HELD IDENTICAL TO GENERATE_FIGURES =====

# The headline metric averages the last 10 of the 100 checkpoints written
FINAL_N = 10

# Group 5 of results.npy is side_stats_data and holds the 4 mutually exclusive episode outcomes
# Index 0 is success, 1 is falling, 2 is unused and always 0, 3 is timeout
# The indices come from data_to_save in OffPolicyMain.py and the 4 sum to 1 at every checkpoint
SIDE_GROUP = 5
METRIC_INDEX = {"success": 0, "falling": 1, "timeout": 3}

# Folder names that are download wrappers rather than runs
NOT_RUNS = re.compile(
    r"^(Adam_|Desktop|Downloads|Results|Winding|Depthwise|Horizontal|baseline_)")

# Voided runs, ablations and smoke tests never enter a reported statistic
EXCLUDE = re.compile(r"^(VOID_|abl_|tf_smoke)|- Copy$")

# These 6 jobs were dispatched against the wrong configuration
# They are excluded by name rather than by pattern, because their names look valid
MISDISPATCH = {"tf_hz_s0", "tf_hz_s1", "tf_hz_s2",
               "tf_dw_s0", "tf_dw_s1", "tf_dw_s2"}

# A run file is named <cell>_s<seed> with an optional batch and job suffix
NAME = re.compile(r"^(?P<cell>.+?)_s(?P<seed>\d)(?:_b(?P<batch>\d+))?(?:_(?P<job>\d+))?$")

def canonical(cell, batch):
    """Map a run's folder name onto the cell key the tables use.

    Naming changed twice during the project.
    Three historical forms are folded onto the current one before grouping.
    This function is held identical to the one in generate_figures.py, because a figure
    and a statistic that disagree about which runs form a cell are worse than either alone.
    """
    # The recurrent arm was written out in full before the 2-letter scheme
    if cell.startswith("lstm_"):
        cell = "ls_" + cell[len("lstm_"):]

    # The earliest runs carried no duration key and all of them were at 3.0 s
    cell = re.sub(r"^(bl|ls|tf)_hz$", r"\1_h_t030", cell)
    cell = re.sub(r"^(bl|ls|tf)_dw$", r"\1_d_t030", cell)

    # A window cell once carried its duration inline, as w04t070 not w04_t070
    inline = re.match(r"^(?P<head>.+?_w\d+)t(?P<dur>\d{3})$", cell)
    if inline is not None:
        cell = (inline.group("head") if inline.group("dur") == "030"
                else f"{inline.group('head')}_t{inline.group('dur')}")

    # Window 4 at the baseline duration is the production baseline cell
    if cell == "bl_h_w04":
        cell = "bl_h_t030"

    return f"{cell}_b{batch}" if batch else cell


# ===== THE COMPARISONS CHAPTER 3 AND APPENDIX c REPORT =====

# Each entry is (family, label, left cell, right cell, metric, published p, published difference in pp)
# The published values are transcribed from Appendix c and are what the recomputation is checked against
COMPARISONS = [
    ("architecture", "Stack-4 vs LSTM (3.0 s)",        "bl_h_t030", "ls_h_t030", "success", 0.0079, +12.3),
    ("architecture", "Stack-4 vs Transformer (3.0 s)", "bl_h_t030", "tf_h_t030", "success", 0.0079, +21.4),
    ("architecture", "Stack-4 vs LSTM (7.0 s)",        "bl_h_t070", "ls_h_t070", "success", 0.0079,  +7.9),
    ("architecture", "Stack-4 vs Transformer (7.0 s)", "bl_h_t070", "tf_h_t070", "success", 0.0952, +16.0),
    ("architecture", "Stack-4 vs LSTM (10.0 s)",       "bl_h_t100", "ls_h_t100", "success", 0.0079,  +8.5),
    ("architecture", "Stack-4 vs Transformer (10.0 s)", "bl_h_t100", "tf_h_t100", "success", 0.0556, +11.5),
    ("architecture", "Stack-4 vs LSTM (12.0 s)",       "bl_h_t120", "ls_h_t120", "success", 0.0079,  +9.3),
    ("architecture", "Stack-4 vs Transformer (12.0 s)", "bl_h_t120", "tf_h_t120", "success", 0.0952,  +7.8),

    ("stack", "Stack-4 vs stack-2",  "bl_h_t030", "bl_h_w02", "success", 0.2222,  +1.3),
    ("stack", "Stack-4 vs stack-8",  "bl_h_t030", "bl_h_w08", "success", 0.0079, +30.3),
    ("stack", "Stack-4 vs stack-10", "bl_h_t030", "bl_h_w10", "success", 0.0079, +44.0),
    ("stack", "Stack-4 vs stack-16", "bl_h_t030", "bl_h_w16", "success", 0.0079, +74.8),
    ("stack", "Stack-10 vs stack-16", "bl_h_w10", "bl_h_w16", "success", 0.0159, None),

    ("controls", "Stack-10 vs LSTM (3.0 s)",            "bl_h_w10", "ls_h_t030", "success", 0.0079, -31.8),
    ("controls", "Batch 320 vs batch 32 (10.0 s)", "bl_h_w04_t100_b320", "bl_h_t100", "success", 0.1508, -7.9),
    ("controls", "Stack-4 vs LSTM (7.0 s)",             "bl_h_t070", "ls_h_t070", "falling", 0.0079,  -7.8),
    ("controls", "Stack-4 vs Transformer (7.0 s)",      "bl_h_t070", "tf_h_t070", "timeout", 0.0079, -12.5),
    ("controls", "Stack-4 vs LSTM (10.0 s)",            "bl_h_t100", "ls_h_t100", "falling", 0.0079,  -8.5),
]

# The 5 test families declared in Section 3.1 before any result was reported
# A family's declared size is what Benjamini-Hochberg divides by, and it exceeds the
# number of comparisons tabulated in Appendix C, so the correction is reported here
# against the comparisons this script holds rather than against the full declared family
DECLARED_FAMILY_SIZE = {
    "stack": 8,
    "horizontal_duration": 10,
    "architecture": 9,
    "behavioural": 6,
    "depthwise_duration": 2,
}

# The controls row group spans more than 1 declared family, so it carries no family size
# Its p-values are recomputed and checked like every other, but its correction is left to the thesis
UNFAMILIED = {"controls"}

ALPHA = 0.05


# ===== THE TESTS =====

def mannwhitney_exact(a, b):
    """Return (U, p, arrangements) from the exact two-tailed test by full enumeration.

    Ties are half-weighted, which is standard for Mann-Whitney and differs from SciPy's
    exact mode, so a tied comparison recomputed with SciPy will not always agree.
    """
    a, b = list(a), list(b)
    n, m = len(a), len(b)

    def u_stat(x, y):
        total = 0.0
        for xi in x:
            for yi in y:
                if xi > yi:
                    total += 1.0
                elif xi == yi:
                    total += 0.5
        return total

    observed = u_stat(a, b)
    pooled = a + b
    centre = n * m / 2.0
    target = abs(observed - centre)

    # The null distribution is every way of splitting the pooled values between the 2 conditions
    count = 0
    at_least_as_extreme = 0
    for combo in itertools.combinations(range(n + m), n):
        chosen = set(combo)
        left = [pooled[i] for i in combo]
        right = [pooled[i] for i in range(n + m) if i not in chosen]
        if abs(u_stat(left, right) - centre) >= target - 1e-12:
            at_least_as_extreme += 1
        count += 1

    return observed, at_least_as_extreme / count, count


def design_floor(n, m):
    """Return the smallest two-tailed p the design can produce, which is 2 of its arrangements."""
    return 2.0 / math.comb(n + m, n)


def hedges_g(a, b):
    """Return the bias-corrected standardised mean difference."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n, m = len(a), len(b)
    pooled_sd = math.sqrt(((n - 1) * a.var(ddof=1) + (m - 1) * b.var(ddof=1)) / (n + m - 2))
    if pooled_sd == 0:
        return float("inf") if a.mean() != b.mean() else 0.0
    d = (a.mean() - b.mean()) / pooled_sd
    # The small-sample correction matters at 5 seeds, where it removes about 9 per cent of the bias
    return d * (1.0 - 3.0 / (4.0 * (n + m) - 9.0))


def benjamini_hochberg(p_values, family_size, alpha=ALPHA):
    """Return 1 rejection flag per p-value, controlling the false discovery rate.

    The threshold for the i-th smallest p-value is i divided by the family size, times alpha.
    Rejection is monotone, so everything ranked below the largest rejected p-value is rejected too.
    """
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    rejected = [False] * len(p_values)
    largest = -1
    for rank, index in enumerate(order, start=1):
        if p_values[index] <= rank / family_size * alpha:
            largest = rank
    for rank, index in enumerate(order, start=1):
        if rank <= largest:
            rejected[index] = True
    return rejected


# ===== LOADING =====

def load_cells(results_dir):
    """Return every selected run grouped by cell key, carrying all 3 reported metrics.

    Two copies of 1 run collapse to 1 entry, because the download history overlaps.
    The copy kept is the shortest name, which is the original rather than a re-download.
    """
    fingerprints = defaultdict(list)

    for path in sorted(results_dir.glob("*__results.npy")):
        stem = path.name[: -len("__results.npy")]

        if NOT_RUNS.match(stem) or EXCLUDE.search(stem) or stem in MISDISPATCH:
            continue

        parsed = NAME.match(stem)
        if parsed is None:
            continue

        data = np.load(path, allow_pickle=True)
        side = np.asarray(data[SIDE_GROUP])

        # Groups 1 to 6 hold every recorded series, so 2 files with the same hash are 1 run saved twice
        digest = hashlib.sha256(b"".join(
            np.asarray(data[i], dtype=float).tobytes() for i in range(1, 7)))

        fingerprints[digest.hexdigest()[:16]].append({
            "stem": stem,
            "cell": canonical(parsed.group("cell"), parsed.group("batch")),
            "seed": int(parsed.group("seed")),
            "metrics": {name: float(np.asarray(side[index][0], dtype=float)[-FINAL_N:].mean())
                        for name, index in METRIC_INDEX.items()},
        })

    kept = [min(group, key=lambda run: (len(run["stem"]), run["stem"]))
            for group in fingerprints.values()]

    cells = defaultdict(list)
    for run in kept:
        cells[run["cell"]].append(run)
    return cells


# ===== REPORTING =====

def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(here.parent / "Metrics"),
                        help="The folder of collected metric series, defaulting to Metrics beside Algorithms")
    parser.add_argument("--only", default=None,
                        help="Restrict output to 1 family, which is architecture, stack or controls")
    args = parser.parse_args()

    results_dir = Path(args.results)
    if not results_dir.is_dir():
        raise SystemExit(f"not a directory: {results_dir}")

    cells = load_cells(results_dir)
    print("=" * 112)
    print("RECOMPUTED STATISTICS, EXACT TWO-TAILED MANN-WHITNEY U BY FULL ENUMERATION")
    print("=" * 112)
    print(f"read {sum(len(v) for v in cells.values())} runs across {len(cells)} cells from {results_dir}")
    print()

    selected = [c for c in COMPARISONS if args.only is None or c[0] == args.only]
    if not selected:
        raise SystemExit(f"no comparisons in family {args.only!r}")

    computed = defaultdict(list)
    rows = []
    missing = []

    for family, label, left_key, right_key, metric, published_p, published_diff in selected:
        left = cells.get(left_key, [])
        right = cells.get(right_key, [])
        if not left or not right:
            missing.append((label, left_key if not left else right_key))
            continue

        a = [run["metrics"][metric] for run in sorted(left, key=lambda r: r["seed"])]
        b = [run["metrics"][metric] for run in sorted(right, key=lambda r: r["seed"])]

        _, p, arrangements = mannwhitney_exact(a, b)
        diff = (np.mean(a) - np.mean(b)) * 100.0
        rows.append({
            "family": family, "label": label, "metric": metric,
            "n": len(a), "m": len(b), "p": p, "published_p": published_p,
            "diff": diff, "published_diff": published_diff,
            "floor": design_floor(len(a), len(b)), "g": hedges_g(a, b),
            "arrangements": arrangements,
        })
        computed[family].append(p)

    # Benjamini-Hochberg runs per family, over the comparisons this script holds
    outcomes = {}
    for family, p_values in computed.items():
        if family in UNFAMILIED:
            continue
        flags = benjamini_hochberg(p_values, DECLARED_FAMILY_SIZE[family])
        for p, flag in zip(p_values, flags):
            outcomes[(family, p)] = flag

    header = (f"  {'comparison':34s} {'metric':8s} {'n v m':>6s} {'diff pp':>8s} "
              f"{'pub':>7s} {'exact p':>9s} {'pub p':>7s} {'floor':>7s} {'g':>7s} {'BH':>4s}")
    for family in ("architecture", "stack", "controls"):
        family_rows = [r for r in rows if r["family"] == family]
        if not family_rows:
            continue
        size = "spans several families" if family in UNFAMILIED else f"declared family size {DECLARED_FAMILY_SIZE[family]}"
        print(f"  {family.upper()}   {size}, {len(family_rows)} comparisons recomputed here")
        print(header)
        for r in family_rows:
            flag = "n/a" if r["family"] in UNFAMILIED else ("yes" if outcomes.get((r["family"], r["p"])) else "no")
            agree = "" if abs(r["p"] - r["published_p"]) < 5e-5 else "  <-- DIFFERS"
            # A dash means the thesis publishes this comparison's p-value without tabulating its difference
            published_diff = "-" if r["published_diff"] is None else f"{r['published_diff']:+.1f}"
            print(f"  {r['label'][:34]:34s} {r['metric']:8s} {r['n']:2d} v {r['m']:<2d} "
                  f"{r['diff']:+8.1f} {published_diff:>7s} {r['p']:9.4f} {r['published_p']:7.4f} "
                  f"{r['floor']:7.4f} {r['g']:+7.2f} {flag:>4s}{agree}")
        print()

    agreed = sum(1 for r in rows if abs(r["p"] - r["published_p"]) < 5e-5)
    print("=" * 112)
    print(f"  matched {agreed} of {len(rows)}")
    for label, key in missing:
        print(f"  skipped {label}, cell {key} absent")
    print("=" * 112)


if __name__ == "__main__":
    main()
