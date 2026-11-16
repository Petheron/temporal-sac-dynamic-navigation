"""Rebuild every generated figure in the thesis from source data.

Usage:
    python Algorithms\\generate_figures.py ^
        --results "<folder of *__results.npy files>" ^
        --screenshots "<folder holding the 2 environment captures>"
"""

import argparse
import hashlib
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Project root, the parent of the Algorithms/ folder this file is in
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_OUT_DIR = PROJECT_ROOT / "ThesisFigures"


# ===== PALETTE AND STYLE =====
# One palette across every figure
# A reader who learns that blue means observation stacking keeps that meaning
INK = "#1b1b1b"     # text drawn inside a figure
MUTE = "#6b6b6b"    # annotations and secondary rules
ACCENT = "#0072ba"  # observation stacking
WARM = "#e16824"    # Long Short-Term Memory
GREEN = "#248e43"   # Transformer
PURPLE = "#77448f"  # stack-16, which appears in Figure 3.8 only

# The thesis text block is 130mm wide
# A figure drawn at 5.1 inches is reproduced close to its natural size
WIDTH = 5.1

STYLE = {
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.labelsize": 9.5, "axes.titlesize": 9.5, "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "lines.linewidth": 1.4, "figure.dpi": 200, "savefig.bbox": "tight",
}


# ===== RUN SELECTION =====
# The headline metric averages the last 10 of the 100 checkpoints written
FINAL_N = 10

# Group 5 of results.npy is side_stats_data and its element 0 is success_rate
# Group 0 element 0 is steps_so_far
# Both indices come from data_to_save in OffPolicyMain.py
SIDE_GROUP = 5
STEP_GROUP = 0

# Folder names that are download wrappers rather than runs
NOT_RUNS = re.compile(
    r"^(Adam_|Desktop|Downloads|Results|Winding|Depthwise|Horizontal|baseline_)")

# Voided runs, ablations and smoke tests never enter a reported figure
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


def load_cells(results_dir):
    """Return every selected run, grouped by cell key.

    Two copies of 1 run collapse to 1 entry, because downloads overlap.
    The copy kept is the shortest name, which is the original not a re-download.
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

        # Groups 1 to 6 hold every recorded series
        # Two files with the same hash are the same run saved twice
        digest = hashlib.sha256(b"".join(
            np.asarray(data[i], dtype=float).tobytes() for i in range(1, 7)))

        fingerprints[digest.hexdigest()[:16]].append({
            "stem": stem,
            "cell": canonical(parsed.group("cell"), parsed.group("batch")),
            "seed": int(parsed.group("seed")),
            "success": float(side[0][0][-FINAL_N:].mean()),
            "series": np.asarray(side[0][0], dtype=float),
            "steps": np.asarray(data[STEP_GROUP][0][0], dtype=float),
        })

    kept = [min(group, key=lambda run: (len(run["stem"]), run["stem"]))
            for group in fingerprints.values()]

    cells = defaultdict(list)
    for run in kept:
        cells[run["cell"]].append(run)
    return cells


def band(axis, xs, series, colour, marker="o"):
    """Draw a mean line with a shaded band spanning the full seed range."""
    means = [float(np.mean(values)) for values in series]
    axis.fill_between(xs, [min(v) for v in series], [max(v) for v in series],
                      color=colour, alpha=0.16, linewidth=0)
    axis.plot(xs, means, color=colour, marker=marker, markersize=4,
              markerfacecolor="white", markeredgewidth=1.1)
    return means


def seeds(cells, key):
    """Return one cell's runs in seed order."""
    return sorted(cells.get(key, []), key=lambda run: run["seed"])


# ===== FIGURE 1.1 (TEMPORAL COVERAGE) =====
# Every value is derived rather than asserted
# 1 decision is the physics timestep of 0.0200s times num_action_repeat of 4
# Stack-N coverage is N decisions and the attention window is 10 decisions
# A platform cycle is twice its traverse time and the episode limit is 250 decisions
DECISION = 0.0200 * 4

COVERAGE_ROWS = [
    ("Observation Stack (4 Frames)", 4 * DECISION, "mechanism"),
    ("Attention Window (10 Decisions)", 10 * DECISION, "mechanism"),
    ("Observation Stack (16 Frames)", 16 * DECISION, "mechanism"),
    ("Platform Cycle (3.0 s Traverse)", 2 * 3.0, "task"),
    ("Platform Cycle (10.0 s Traverse)", 2 * 10.0, "task"),
    ("Platform Cycle (12.0 s Traverse)", 2 * 12.0, "task"),
    ("Episode Limit (250 Decisions)", 250 * DECISION, "task"),
]

COVERAGE_FULL = 25.0
COVERAGE_ZOOM = 1.45


def coverage_panel(axis, rows, limit, ticks, pad):
    """Draw one panel of horizontal bars on a linear time axis."""
    ys = np.arange(len(rows))[::-1]

    for y, (label, value, kind) in zip(ys, rows):
        axis.barh(y, value, height=0.58, linewidth=0,
                  color=ACCENT if kind == "mechanism" else WARM)
        axis.text(value + pad, y, f"{value:.2f} s" if value < 10 else f"{value:.1f} s",
                  va="center", fontsize=8, color=INK)

    axis.set_yticks(ys)
    axis.set_yticklabels([row[0] for row in rows])
    axis.set_xlim(0, limit)
    axis.set_ylim(-0.7, len(rows) - 0.3)
    axis.set_xticks(ticks)
    axis.set_xlabel("Simulated Time Spanned (Seconds)")
    axis.grid(axis="y", visible=False)


def figure_1_1(out_dir):
    """Temporal coverage of each mechanism against the task's own timescales.

    Both axes are linear.
    The step from 0.1 s to 0.3 s must not be drawn as wide as 10 s to 30 s.
    The gap between the 2 groups is the single thing this figure exists to show.
    At full scale the 3 mechanism bars are slivers, so panel (b) magnifies them.
    """
    # Panel titles sit a size below the shared default
    # They label a panel, not the figure
    plt.rcParams["axes.titlesize"] = 8.5

    figure, (full, zoom) = plt.subplots(
        2, 1, figsize=(5.4, 3.7),
        gridspec_kw={"height_ratios": [7, 3.1], "hspace": 0.72})

    coverage_panel(full, COVERAGE_ROWS, COVERAGE_FULL,
                   np.arange(0, COVERAGE_FULL + 1, 5), 0.45)
    full.axhline(3.5, color=INK, linewidth=0.5, alpha=0.35)
    full.set_title("(a) All Seven Quantities at One Scale", loc="left", color=INK)
    full.legend(handles=[Patch(color=ACCENT, label="Mechanism Coverage"),
                         Patch(color=WARM, label="Task Timescale")],
                loc="upper right", ncol=1, frameon=False, fontsize=8.5,
                handlelength=1.5, handletextpad=0.6, borderaxespad=0.4)

    coverage_panel(zoom, COVERAGE_ROWS[:3], COVERAGE_ZOOM,
                   [0, 0.25, 0.5, 0.75, 1.0, 1.25], 0.026)
    zoom.set_title("(b) The Three Mechanism Quantities (Magnified)",
                   loc="left", color=INK)

    figure.savefig(out_dir / "fig_temporal_coverage.pdf")
    plt.close(figure)
    plt.rcParams["axes.titlesize"] = STYLE["axes.titlesize"]

    for label, value, _ in COVERAGE_ROWS:
        print(f"  {label:34s} {value:7.2f} s")


# ===== FIGURE 2.1 (MOVING PLATFORM ENVIRONMENTS) =====
# Each capture, the figure it becomes, and the geometry of its platforms
# Both travel distances come from Section 2.2, which takes them from Grant (2025)
# The last field is the ground axis the platform travels along
ENVIRONMENTS = [
    ("Depthwise Environment.png", "env_depthwise.png", 5.0, 8.0, "along"),
    ("Horizontal Environment.png", "env_horizontal.png", 7.0, 14.0, "across"),
]

# The Unity game view inside the captured window, as left, top, right, bottom
GAME_VIEW = (320, 176, 2240, 1256)

# Pure green, which is the colour the source work uses for a travel arrow
ARROW_GREEN = (0, 255, 0)

# Arrows are drawn at 3 times size and reduced, which smooths their edges
SUPERSAMPLE = 3


def platform_blobs(mask, floor):
    """Return the centre and both top-face edge vectors of every platform.

    A platform is a square, so its 2 top-face edges give the screen projection
    of its side length along each ground axis at that platform's own depth.
    """
    from scipy import ndimage

    labelled, count = ndimage.label(mask)
    found = []

    for index in range(1, count + 1):
        ys, xs = np.where(labelled == index)
        if len(xs) < floor:
            continue
        points = np.stack([xs, ys], 1).astype(float)
        found.append({
            "centre": points.mean(0),
            "along": points[np.argmin(points[:, 1])] - points[np.argmin(points[:, 0])],
            "across": points[np.argmax(points[:, 0])] - points[np.argmin(points[:, 1])],
        })

    return found


def draw_arrow(draw, start, vector, width):
    """Draw 1 double-headed arrow from start to start plus vector."""
    start = np.asarray(start, dtype=float) * SUPERSAMPLE
    vector = np.asarray(vector, dtype=float) * SUPERSAMPLE
    width = width * SUPERSAMPLE

    unit = vector / np.linalg.norm(vector)
    normal = np.array([-unit[1], unit[0]])
    head_length, head_half = width * 2.6, width * 1.7

    tips = (start, start + vector)
    bases = (tips[0] + unit * head_length, tips[1] - unit * head_length)

    draw.polygon([tuple(bases[0] + normal * width / 2),
                  tuple(bases[1] + normal * width / 2),
                  tuple(bases[1] - normal * width / 2),
                  tuple(bases[0] - normal * width / 2)], fill=ARROW_GREEN)

    for tip, base in zip(tips, bases):
        draw.polygon([tuple(tip), tuple(base + normal * head_half),
                      tuple(base - normal * head_half)], fill=ARROW_GREEN)


def travel_vector(platform, axis, side_m, travel_m, anchors):
    """Return the arrow for one platform, in pixels, pointing the way it travels.

    A depth-wise platform starts beside 1 static platform and crosses to the next.
    Its arrow therefore points away from the static platform it is nearest.
    A horizontal platform sweeps across the line of static platforms.
    Its arrow therefore points to the far side of that line.
    The 2 depth-wise platforms start at opposite ends, so a fixed direction is wrong.
    """
    edge = platform[axis]
    vector = edge * (travel_m / side_m)

    if axis == "along":
        nearest = anchors[np.argmin(np.linalg.norm(anchors - platform["centre"], axis=1))]
        if np.dot(vector, platform["centre"] - nearest) < 0:
            vector = -vector
    else:
        chain = platform["along"] / np.linalg.norm(platform["along"])
        normal = np.array([-chain[1], chain[0]])
        if np.dot(platform["centre"] - anchors.mean(0), normal) * np.dot(vector, normal) > 0:
            vector = -vector

    return vector, edge


def figure_2_1(out_dir, screenshots_dir):
    """The 2 environments, each moving platform carrying its own travel arrow.

    Every arrow is derived rather than placed by eye.
    Measuring against each platform's own edges absorbs the camera's perspective.
    """
    from PIL import Image, ImageDraw

    for source, target, side_m, travel_m, axis in ENVIRONMENTS:
        capture = screenshots_dir / source
        if not capture.is_file():
            raise SystemExit(f"Capture not found: {capture}")

        image = Image.open(capture).convert("RGB")
        pixels = np.asarray(image).astype(int)
        red, green, blue = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]

        # The bright top face only
        # Including the darker side walls would move the corners being measured
        moving = platform_blobs((red > 200) & (green < 80) & (blue < 80), 6000)
        static = platform_blobs(
            (red > 200) & (green > 150) & (green < 230) & (blue < 140), 6000)

        if not moving or not static:
            raise SystemExit(f"{source}: no platforms found, check the capture")

        anchors = np.array([platform["centre"] for platform in static])
        print(f"  {source}: {len(moving)} moving and {len(static)} static platforms")

        big = image.resize((image.width * SUPERSAMPLE, image.height * SUPERSAMPLE),
                           Image.LANCZOS)
        draw = ImageDraw.Draw(big)

        for platform in sorted(moving, key=lambda p: p["centre"][0]):
            vector, edge = travel_vector(platform, axis, side_m, travel_m, anchors)
            draw_arrow(draw, platform["centre"], vector, 0.09 * np.linalg.norm(edge))
            print(f"    centre ({platform['centre'][0]:6.0f},{platform['centre'][1]:6.0f})"
                  f"  edge {np.linalg.norm(edge):6.1f} px = {side_m:.1f} m"
                  f"  arrow {np.linalg.norm(vector):6.1f} px = {travel_m:.1f} m")

        big.resize(image.size, Image.LANCZOS).crop(GAME_VIEW).save(out_dir / target)


# ===== FIGURE 3.1 (OBSERVATION-STACK CURVE) =====
# The stack sizes run, paired with the cell key each one is stored under
STACK_CELLS = [(2, "bl_h_w02"), (4, "bl_h_t030"), (8, "bl_h_w08"),
               (10, "bl_h_w10"), (16, "bl_h_w16")]


def figure_3_1(out_dir, cells):
    """Final-10 success rate against observation-stack size.

    The horizontal axis is linear, because the variable is a count of frames.
    Equal differences in that count must be drawn as equal distances.
    """
    xs, series, counts = [], [], []
    for size, key in STACK_CELLS:
        group = cells.get(key, [])
        if not group:
            continue
        xs.append(size)
        series.append([run["success"] for run in group])
        counts.append(len(group))

    figure, axis = plt.subplots(figsize=(WIDTH, 3.0))
    means = band(axis, xs, series, ACCENT)

    for x, mean, count in zip(xs, means, counts):
        axis.annotate(f"n={count}", (x, mean), textcoords="offset points",
                      xytext=(0, 9), ha="center", fontsize=7.5, color=MUTE)

    # Ticks step by 2 rather than landing only on the sampled sizes
    # Sampled-only ticks would put uneven gaps on an axis that is itself even
    axis.set_xticks(np.arange(2, 17, 2))
    axis.set_xlim(1, 17)
    axis.set_xlabel("Observation-Stack Size (Frames)")
    axis.set_ylabel("Final-10 Success Rate")
    axis.set_ylim(0, 1.0)

    figure.savefig(out_dir / "fig_stack_curve.pdf")
    plt.close(figure)
    print("  " + "  ".join(f"stack-{x}: {m:.3f} (n={c})"
                           for x, m, c in zip(xs, means, counts)))


# ===== FIGURE 3.2 (DURATION CURVE) =====
# Traverse times run, in deciseconds, matching the cell keys
DURATIONS = [30, 40, 50, 70, 80, 90, 100, 110, 120]


def figure_3_2(out_dir, cells):
    """Final-10 success rate against platform traverse time."""
    xs, series, counts = [], [], []
    for tenths in DURATIONS:
        group = cells.get(f"bl_h_t{tenths:03d}", [])
        if not group:
            continue
        xs.append(tenths / 10.0)
        series.append([run["success"] for run in group])
        counts.append(len(group))

    figure, axis = plt.subplots(figsize=(WIDTH, 3.0))
    means = band(axis, xs, series, ACCENT, marker="s")

    axis.axvline(10.0, color=WARM, linestyle=(0, (4, 3)), linewidth=1.0, zorder=1)
    axis.annotate("Knee at 10.0 s", (10.0, 0.22), textcoords="offset points",
                  xytext=(-6, 0), ha="right", fontsize=8, color=WARM)

    # The counts sit on a clear strip above the curve rather than beside each point, because a patch next to a point covers the band behind it
    top = max(max(col) for col in series)
    for x, count in zip(xs, counts):
        axis.annotate(f"n={count}", (x, top + 0.07), ha="center",
                      fontsize=7.2, color=MUTE, zorder=6)

    axis.set_xlabel("Platform Traverse Time (Seconds)")
    axis.set_ylabel("Final-10 Success Rate")

    # Every whole second from 3 to 12, so the gap between ticks is constant
    # 6.0s was not one of the traverse times run and is still given a tick
    axis.set_xticks(np.arange(3, 13, 1))
    axis.set_xlim(2.6, 12.4)
    axis.set_ylim(0, 1.16)

    figure.savefig(out_dir / "fig_duration_curve.pdf")
    plt.close(figure)
    print("  " + "  ".join(f"{x:.1f} s: {m:.3f} (n={c})"
                           for x, m, c in zip(xs, means, counts)))


# ===== FIGURE 3.3 (PER-SEED TRAJECTORIES) =====
# The 3 arms at the 10.0 s traverse, with the line style each is drawn in
ARMS_AT_10S = [
    ("bl_h_t100", "Observation Stacking at Stack-4", ACCENT, "-"),
    ("ls_h_t100", "LSTM", WARM, "-"),
    ("tf_h_t100", "Transformer", GREEN, "-"),
]

# How opaque each of the 5 per-seed traces is drawn in Figure 3.3
# Lower values separate overlapping traces and lighten the panel, higher values keep a single trace legible where it runs alone
# 0.25 was tried and rejected because it faded the Transformer's late collapses, which are the evidence panel (c) exists to carry
TRACE_ALPHA = 0.50


def figure_3_3(out_dir, cells):
    """Per-seed success trajectories, 1 panel per architecture.

    Fifteen traces on a single pair of axes cannot be read.
    This figure exists to show dispersion within an arm, which overplotting hides.
    The axes are shared, so the panels stay directly comparable.
    """
    figure, axes = plt.subplots(3, 1, figsize=(WIDTH, 5.6), sharex=True, sharey=True)

    # Each panel carries 1 architecture, so the colour alone tells them apart and no dash pattern is needed
    # Partial opacity is what makes 5 overlapping traces readable as 5
    for index, (axis, (key, label, colour, dashes)) in enumerate(zip(axes, ARMS_AT_10S)):
        group = seeds(cells, key)
        for run in group:
            keep = run["steps"] >= 0
            axis.plot(run["steps"][keep] / 1e6, run["series"][keep],
                      color=colour, linestyle=dashes, linewidth=0.9,
                      alpha=TRACE_ALPHA)
        axis.set_title(f"({chr(97 + index)}) {label} ({len(group)} Seeds)",
                       fontsize=8.2, color=INK)
        axis.set_xlim(0, 1.0)
        axis.set_ylim(0, 1.0)
        axis.set_xticks(np.arange(0, 1.01, 0.1))
        axis.set_yticks(np.arange(0, 1.01, 0.2))
        axis.grid(alpha=0.22, linewidth=0.45)
        axis.set_ylabel("Success Rate")
        axis.set_xlabel("")

    axes[-1].set_xlabel("Environment Decision Steps (Millions)")
    figure.tight_layout(pad=0.5)

    figure.savefig(out_dir / "fig_per_seed_trajectories.pdf")
    plt.close(figure)
    print("  " + "  ".join(f"{key}: {len(cells.get(key, []))} seeds"
                           for key, _, _, _ in ARMS_AT_10S))


# ===== FIGURE 3.4 (TRAINING CURVES) =====
# The 4 traverse times given a panel, with the cell-key suffix of each
PANELS = [("3.0 Second Traverse", "030"), ("7.0 Second Traverse", "070"),
          ("10.0 Second Traverse", "100"), ("12.0 Second Traverse", "120")]

# The 3 arms, with the cell-key prefix and line style of each
ARMS = [("bl", "Observation Stacking\n(Stack-4)", ACCENT, "-"),
        ("ls", "LSTM", WARM, (0, (5, 2))),
        ("tf", "Transformer", GREEN, (0, (1.6, 1.6)))]

# This grid keeps a full box around each panel and draws its own grid
# A panel this small reads better closed than open, so 2 shared settings flip
GRID_STYLE = {
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
    "axes.labelsize": 8.5, "axes.titlesize": 8.5, "legend.fontsize": 7.2,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.grid": False, "lines.linewidth": 1.5,
    "figure.dpi": 220, "savefig.bbox": "tight",
}


def figure_3_4(out_dir, cells):
    """Mean training curves at 4 traverse times, banded over the seed range."""
    with plt.rc_context(GRID_STYLE):
        figure, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.1),
                                    sharex=True, sharey=True)

        for index, (axis, (title, tag)) in enumerate(zip(axes.ravel(), PANELS)):
            for prefix, label, colour, dashes in ARMS:
                group = seeds(cells, f"{prefix}_h_t{tag}")
                if not group:
                    continue
                keep = group[0]["steps"] >= 0
                stacked = np.vstack([run["series"][keep] for run in group])
                x = group[0]["steps"][keep] / 1e6
                axis.fill_between(x, stacked.min(axis=0), stacked.max(axis=0),
                                  color=colour, alpha=0.16, linewidth=0)
                axis.plot(x, stacked.mean(axis=0), color=colour,
                          linestyle=dashes, linewidth=1.1, label=label)
            axis.set_title(f"({chr(97 + index)}) {title}")
            axis.set_xlim(0, 1.0)
            axis.set_ylim(0, 1.0)
            axis.set_xticks(np.arange(0, 1.01, 0.2))
            axis.set_yticks(np.arange(0, 1.01, 0.2))
            axis.grid(alpha=0.22, linewidth=0.45)
            axis.tick_params(labelbottom=True)

        for axis in axes[:, 0]:
            axis.set_ylabel("Success Rate")
        figure.supxlabel("Environment Decision Steps (Millions)", fontsize=8.5, y=0.015)

        # One key for the whole grid, placed above the panels
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
                      fontsize=8.4, bbox_to_anchor=(0.5, 1.015),
                      handlelength=2.2, columnspacing=2.4)

        figure.tight_layout(pad=0.5, rect=(0, 0.04, 1, 0.90))
        figure.savefig(out_dir / "fig_training_curves.pdf")
        plt.close(figure)

    for title, tag in PANELS:
        parts = []
        for prefix, _, _, _ in ARMS:
            group = cells.get(f"{prefix}_h_t{tag}", [])
            mean = np.mean([run["success"] for run in group]) if group else float("nan")
            parts.append(f"{prefix}={mean:.3f}(n={len(group)})")
        print(f"  {title:22s} " + "  ".join(parts))


# ===== FIGURE 3.6 (PER-FRAME SENSITIVITY) =====
# Gradient-norm sensitivity per stacked frame, 1 row per seed, oldest frame first
# These are probe outputs, not training outputs, so they are not in the results
# They are reproduced by running the sensitivity probe over each cell's exports
# They match Table 3.14 of the thesis row for row
SENSITIVITY = {
    2: [[0.7757, 0.9503], [0.8050, 0.9305], [0.8968, 0.8772],
        [0.9887, 1.0771], [0.6857, 0.7942]],
    4: [[0.5102, 0.5240, 0.5833, 0.7293],
        [0.6081, 0.6524, 0.5951, 0.8225],
        [0.5467, 0.5440, 0.5838, 0.7897]],
    8: [[0.2799, 0.2827, 0.2716, 0.3004, 0.2951, 0.3452, 0.3573, 0.4344],
        [0.2454, 0.3028, 0.2502, 0.2871, 0.3059, 0.3000, 0.3092, 0.4431],
        [0.2810, 0.2712, 0.2956, 0.3307, 0.3245, 0.3606, 0.3423, 0.4722]],
    16: [[0.0904, 0.0916, 0.0885, 0.0853, 0.0903, 0.0936, 0.0958, 0.0872,
          0.0937, 0.0992, 0.1230, 0.1211, 0.1112, 0.1330, 0.1444, 0.2184],
         [0.1073, 0.1040, 0.1153, 0.1032, 0.1257, 0.1065, 0.1174, 0.1329,
          0.1377, 0.1215, 0.1381, 0.1387, 0.1355, 0.1506, 0.1500, 0.2486],
         [0.1130, 0.1117, 0.1237, 0.1235, 0.1286, 0.1273, 0.1087, 0.1268,
          0.1341, 0.1587, 0.1662, 0.1515, 0.1677, 0.1749, 0.1917, 0.3395],
         [0.0807, 0.0870, 0.0979, 0.0847, 0.0981, 0.0931, 0.0899, 0.1145,
          0.1190, 0.1101, 0.1012, 0.1270, 0.1476, 0.1393, 0.1620, 0.2551]],
}

SENSITIVITY_COLOUR = {2: ACCENT, 4: WARM, 8: GREEN, 16: PURPLE}
SENSITIVITY_MARKER = {2: "o", 4: "s", 8: "^", 16: "D"}


def figure_3_9(out_dir):
    """Per-frame sensitivity profiles, 1 series per stack size."""
    figure, axis = plt.subplots(figsize=(WIDTH, 3.2))

    for size in (2, 4, 8, 16):
        rows = np.array(SENSITIVITY[size])
        x = np.arange(1, size + 1)
        # A band over the seed range, which is this chapter's convention
        # Drawn as 5 separate thin traces it reads as stray marks, not as spread
        axis.fill_between(x, rows.min(axis=0), rows.max(axis=0),
                          color=SENSITIVITY_COLOUR[size], alpha=0.16, linewidth=0)
        axis.plot(x, rows.mean(axis=0), color=SENSITIVITY_COLOUR[size],
                  marker=SENSITIVITY_MARKER[size], markersize=3.6,
                  markerfacecolor="white", markeredgewidth=1.0,
                  label=f"Stack-{size} ({rows.shape[0]} Seeds)")
        print(f"  stack-{size:<2d} mean {np.round(rows.mean(axis=0), 3).tolist()}"
              f"  total {rows.mean(axis=0).sum():.2f}")

    axis.set_xlabel("Frame Position in the Stack (Oldest to Newest)")
    axis.set_ylabel("Per-Frame Sensitivity")

    # Constant steps of 2, rather than a 1 a single unit from its neighbour
    # Every other gap on this axis is 2 units wide
    axis.set_xticks(np.arange(2, 17, 2))
    axis.set_xlim(0.4, 16.6)
    axis.set_ylim(0, 1.15)

    # The upper left is where the stack-2 curve sits, so the key goes right
    # No series reaches that part of the field
    axis.legend(loc="upper right", frameon=False, ncol=1)

    figure.savefig(out_dir / "fig_per_frame_sensitivity.pdf")
    plt.close(figure)


# ===== FIGURE 3.8 (TRAINING DIAGNOSTICS) =====
# The 3 logged series that rule out an exploration or critic explanation
# Group 3 is value_data [v, q, log_p, entropy, alpha]; group 2 is loss_data
# [actor, critic, alpha]; both are (series, runs, iterations)
# Title, group, element, y scale, and the x limit in millions of steps
# Entropy gets a shorter x range because all of its behaviour is in the first 100,000 steps and the rest of the axis would be flat
DIAGNOSTICS = [
    ("Policy Entropy (First 100,000 Steps)", 3, 3, None, 0.1),
    ("Critic Loss", 2, 1, "log", 1.0),
    ("Mean State-Action Value", 3, 1, None, 1.0),
]


def figure_3_8(out_dir, cells, results_dir):
    """The 3 logged diagnostics behind the late-training instability."""
    with plt.rc_context(GRID_STYLE):
        figure, axes = plt.subplots(3, 1, figsize=(WIDTH, 6.0))

        for index, (axis, (title, group, element, scale, limit)) in enumerate(
                zip(axes, DIAGNOSTICS)):
            for prefix, label, colour, dashes in ARMS:
                runs = cells.get(f"{prefix}_h_t100", [])
                if not runs:
                    continue
                stacked, steps = [], None
                for run in runs:
                    raw = np.load(results_dir / f"{run['stem']}__results.npy",
                                  allow_pickle=True)
                    series = np.asarray(raw[group], dtype=float)[element].ravel()
                    axis_steps = np.asarray(raw[STEP_GROUP][0][0],
                                            dtype=float).ravel()
                    keep = axis_steps >= 0
                    stacked.append(series[keep])
                    steps = axis_steps[keep]
                width = min(len(row) for row in stacked)
                block = np.vstack([row[:width] for row in stacked])
                x = steps[:width] / 1e6
                axis.fill_between(x, block.min(axis=0), block.max(axis=0),
                                  color=colour, alpha=0.14, linewidth=0)
                axis.plot(x, block.mean(axis=0), color=colour,
                          linestyle=dashes, linewidth=1.0, label=label)
            if scale == "log":
                axis.set_yscale("log")
            axis.set_title(f"({chr(97 + index)}) {title}", fontsize=8.6,
                           linespacing=1.3, pad=5.0)
            axis.set_xlim(0, limit)
            axis.set_xticks(np.linspace(0, limit, 5))
            axis.grid(alpha=0.22, linewidth=0.45)
            axis.set_xlabel("Decision Steps (Millions)", fontsize=8)
            axis.tick_params(labelsize=7.6)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
                      fontsize=8.6, bbox_to_anchor=(0.5, 0.995),
                      handlelength=2.2, columnspacing=2.4)
        figure.tight_layout(pad=0.7, h_pad=2.2, rect=(0, 0, 1, 0.925))
        figure.savefig(out_dir / "fig_training_diagnostics.pdf")
        plt.close(figure)
    print("  entropy, critic loss and mean Q drawn for the 3 arms at 10.0 s")


# ===== DRIVER =====
# Every figure this script can build, with what it needs and what builds it
FIGURES = [
    ("1.1", None, figure_1_1),
    ("2.1", "screenshots", figure_2_1),
    ("3.1", "results", figure_3_1),
    ("3.2", "results", figure_3_2),
    ("3.3", "results", figure_3_3),
    ("3.4", "results", figure_3_4),
    ("3.8", "results", figure_3_8),
    ("3.9", None, figure_3_9),
]

ALL_NUMBERS = [number for number, _, _ in FIGURES]

# Figure 2.1 is left out of a default run because the committed version carries arrows drawn by hand over what this script produces
# Rebuilding it would replace those with the measured arrows, so it has to be asked for by name with --only 2.1
HAND_FINISHED = {"2.1"}
DEFAULT_NUMBERS = [number for number in ALL_NUMBERS if number not in HAND_FINISHED]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=None,
                        help="folder of collected *__results.npy files")
    parser.add_argument("--screenshots", type=Path, default=None,
                        help="folder holding the 2 Unity environment captures")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--only", nargs="+", metavar="N", default=DEFAULT_NUMBERS,
                        help=f"build only these, from {' '.join(ALL_NUMBERS)};"
                             f" {' '.join(sorted(HAND_FINISHED))} is left out of a"
                             f" default run and has to be asked for by name")
    args = parser.parse_args()

    unknown = [number for number in args.only if number not in ALL_NUMBERS]
    if unknown:
        raise SystemExit(f"Unknown figure(s): {' '.join(unknown)}")

    if args.results is not None and not args.results.is_dir():
        raise SystemExit(f"Results directory not found: {args.results}")

    if args.screenshots is not None and not args.screenshots.is_dir():
        raise SystemExit(f"Screenshots directory not found: {args.screenshots}")

    print(f"results    : {args.results}")
    print(f"screenshots: {args.screenshots}")
    print(f"out        : {args.out}\n")

    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)

    inputs = {"results": args.results, "screenshots": args.screenshots}
    wanted = [entry for entry in FIGURES if entry[0] in args.only]

    # Results are loaded once, and only if a figure in this run needs them
    cells = None
    if any(needs == "results" for _, needs, _ in wanted) and args.results is not None:
        cells = load_cells(args.results)
        print(f"loaded {sum(len(v) for v in cells.values())} runs "
              f"across {len(cells)} cells\n")

    built, skipped = [], []

    for number, needs, builder in wanted:
        if needs is not None and inputs[needs] is None:
            skipped.append(f"Figure {number} needs --{needs}")
            continue
        print(f"Figure {number}")
        if needs == "results":
            # The diagnostics figure reads series the cell records do not hold, so it alone also needs the folder they came from
            if number == "3.8":
                builder(args.out, cells, args.results)
            else:
                builder(args.out, cells)
        elif needs == "screenshots":
            builder(args.out, args.screenshots)
        else:
            builder(args.out)
        built.append(number)
        print()

    print(f"wrote to {args.out}")
    print(f"built  : {', '.join(built) if built else 'nothing'}")
    for note in skipped:
        print(f"skipped: {note}")



if __name__ == "__main__":
    main()
