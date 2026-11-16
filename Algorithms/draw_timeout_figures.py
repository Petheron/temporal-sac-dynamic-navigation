"""Draw the 2 timeout figures of Chapter 3 in the thesis's own style.

Usage:
    python Algorithms\\draw_timeout_figures.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths are resolved from this file, which sits in Algorithms/
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_PLOTTING_DIR = PROJECT_ROOT / "Results" / "plotting"
DEFAULT_OUT_DIR = PROJECT_ROOT.parent / "honours-thesis" / "figures"

# Anything not destined for the project itself goes to the downloads folder
# Built from the home directory so no username is hard-coded here
CHECK_DIR = Path.home() / "Downloads"

# results.npy group and element indices, from data_to_save in OffPolicyMain.py
# Group 5 is side_stats_data and its element 3 is timeout_fail_rate
# Group 0 element 0 is steps_so_far
SIDE_GROUP, TIMEOUT_INDEX = 5, 3
STEP_GROUP, STEP_INDEX = 0, 0

# The palette the rest of the thesis uses, so a colour keeps its meaning
INK, MUTE = "#1b1b1b", "#6b6b6b"
ACCENT, WARM, GREEN, PURPLE = "#0072ba", "#e16824", "#248e43", "#77448f"

# Figure 3.8 already colours the stack sizes, so these follow it exactly
# Stack-10 has no counterpart there because the sensitivity probe skipped it, so it takes a 5th colour distinct from the other 4
CRIMSON = "#bb264e"

# The text block is 130mm wide, so 5.1 inches reproduces close to natural size
WIDTH = 5.1

STYLE = {
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.labelsize": 9.5, "legend.fontsize": 8,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.dpi": 200, "savefig.bbox": "tight",
}

# Each figure: its output name, its group folder, and the series to draw
# A series is its subfolder, its legend entry, its colour and its dash pattern
FIGURES = [
    {
        "name": "stack_timeout_curve.pdf",
        "group": "06_stack_curve_horizontal",
        "series": [
            ("1_stack02", "Stack-2", ACCENT, "-"),
            ("2_stack04", "Stack-4", WARM, (0, (5, 2))),
            ("3_stack08", "Stack-8", GREEN, (0, (1.6, 1.6))),
            ("4_stack10", "Stack-10", CRIMSON, (0, (4, 1.4, 1, 1.4))),
            ("5_stack16", "Stack-16", PURPLE, "-"),
        ],
    },
    {
        "name": "matched_context_timeout_curve.pdf",
        "group": "08_matched_context_t100",
        "series": [
            ("1_stack10", "Observation Stacking (Stack-10)", ACCENT, "-"),
            ("2_lstm", "LSTM", WARM, (0, (5, 2))),
            ("3_transformer", "Transformer", GREEN, (0, (1.6, 1.6))),
        ],
    },
]


def read_series(folder):
    """Return the step axis and the per-seed timeout rates of 1 merged file

    The merged file holds 1 row per seed, so the timeout array comes back as
    (seeds, checkpoints) and the step axis is taken from the first row.
    """
    path = folder / "results.npy"
    if not path.is_file():
        raise SystemExit(f"Not found: {path}")

    data = np.load(path, allow_pickle=True)
    timeout = np.asarray(data[SIDE_GROUP][TIMEOUT_INDEX], dtype=float)
    steps = np.asarray(data[STEP_GROUP][STEP_INDEX], dtype=float)

    if timeout.ndim != 2 or steps.ndim != 2:
        raise SystemExit(f"{path} is not a merged multi-seed file")

    keep = steps[0] >= 0
    return steps[0][keep], timeout[:, keep]


def draw(spec, plotting_dir, out_dir):
    """Draw 1 figure and return the numbers worth quoting from it"""
    group = plotting_dir / spec["group"]
    if not group.is_dir():
        raise SystemExit(f"Not a directory: {group}")

    figure, axis = plt.subplots(figsize=(WIDTH, 2.9))
    facts = []

    for arm, label, colour, dashes in spec["series"]:
        steps, timeout = read_series(group / arm)
        x = steps / 1e6
        mean = timeout.mean(axis=0)

        # The band spans every seed, which is the convention of this chapter
        axis.fill_between(x, timeout.min(axis=0), timeout.max(axis=0),
                          color=colour, alpha=0.14, linewidth=0)
        axis.plot(x, mean, color=colour, linestyle=dashes, linewidth=1.2,
                  label=label)

        # The second half of training, which is what the prose describes
        # A label wrapped for the key is flattened again for the printed table
        half = x >= 0.5
        facts.append({
            "arm": " ".join(label.split()),
            "seeds": timeout.shape[0],
            "final": float(mean[-1]),
            "plateau": float(mean[half].mean()),
            "band_low": float(timeout[:, half].min()),
            "band_high": float(timeout[:, half].max()),
            "peak": float(mean.max()),
            "peak_at": float(x[int(mean.argmax())]),
            "zero_at": next((float(v) for v, m in zip(x, mean) if m <= 0.01), None),
        })

    axis.set_xlabel("Environment Decision Steps (Millions)")
    axis.set_ylabel("Timeout Rate")
    axis.set_xlim(0, 1.0)
    axis.set_ylim(0, 1.0)
    axis.set_xticks(np.arange(0, 1.01, 0.2))
    axis.set_yticks(np.arange(0, 1.01, 0.2))

    # The key sits above the axes rather than beside them, because a key to the right takes its width from the plotted field
    # Every series also runs through the upper right of this field at some point
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False,
                ncol=len(spec["series"]), handlelength=2.0, borderaxespad=0,
                columnspacing=1.8)

    out_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_dir / spec["name"])
    plt.close(figure)
    return facts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plotting-dir", type=Path, default=DEFAULT_PLOTTING_DIR,
                        help="the folder of merged comparison groups")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR,
                        help="where to write the figures")
    parser.add_argument("--stats", action="store_true",
                        help="print the numbers the prose quotes")
    parser.add_argument("--check", action="store_true",
                        help="write to the downloads folder and print the "
                             "numbers, leaving the thesis figures alone")
    args = parser.parse_args()

    # A check run must not be able to touch what is committed
    if args.check:
        args.out = CHECK_DIR
        args.stats = True

    plt.rcParams.update(STYLE)
    print(f"reading from: {args.plotting_dir}")
    print(f"writing to  : {args.out}\n")

    for spec in FIGURES:
        facts = draw(spec, args.plotting_dir, args.out)
        print(f"{spec['name']}")
        for fact in facts:
            zero = (f"{fact['zero_at']:.2f}M" if fact["zero_at"] is not None
                    else "never")
            print(f"  {fact['arm']:32s} {fact['seeds']} seeds  "
                  f"final {fact['final']:.3f}  "
                  f"second-half mean {fact['plateau']:.3f}")
            if args.stats:
                print(f"  {'':32s} peak {fact['peak']:.3f} at "
                      f"{fact['peak_at']:.2f}M, reaches 0.01 at {zero}, "
                      f"second-half seed range "
                      f"{fact['band_low']:.3f} to {fact['band_high']:.3f}")
        print()



if __name__ == "__main__":
    sys.exit(main())
