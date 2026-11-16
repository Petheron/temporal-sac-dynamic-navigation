"""Settle whether the per seed differences are about training or about activation structure.

Usage:
    python Algorithms\\probe_06_seed_table.py "C:\\Users\\Adam_\\Downloads\\HPC_Download_24th_October"
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

try:
    from probe_02_attention_map import (
        load_actor_state,
        extract_submodule_state,
        build_encoder_stack,
        infer_transformer_geometry,
        build_transformer,
        effective_context_batch,
        AttentionRecorder,
        TRANSFORMER_PREFIX,
        OBS_ENCODER_PREFIX,
        TRUNK_PREFIX,
        TRUNK_ACTIVATIONS,
    )
    from probe_03_gate_and_lstm import ablation_ratio, measured_window, LSTM_PREFIX
    from probe_05_summary import collect_checkpoints, identify_architecture, seed_from_name
except ImportError as error:
    raise SystemExit(
        "Could not import the earlier probe files.\n"
        "probe_02, probe_03, probe_05 and this file must all sit in the Algorithms folder.\n"
        f"Original error: {error}"
    )

# The magnitudes the Gaussian sweep is read at
# 1 to 100 is the range where probe_03 found the pathway conducting
SWEEP = (1.0, 10.0, 100.0)


def transformer_forward(module):
    """Return a callable mapping a window to the output at its last position."""
    def forward(window):
        return module(window)[:, -1, :]
    return forward


def recurrent_forward(module, layers, hidden_size):
    """Return the same callable for a recurrent module, starting from a zero state."""
    def forward(window):
        zeros = torch.zeros(layers, window.shape[0], hidden_size)
        out, _ = module(window, (zeros, zeros))
        return out[:, -1, :]
    return forward


def effective_context_of(module, window, seq_len):
    """Return the effective context at the last query position of the final layer."""
    with AttentionRecorder(module) as recorder:
        with torch.no_grad():
            module(window)
    if not recorder.weights:
        return float("nan")
    final = max(recorder.weights)
    weights = recorder.weights[final]
    if weights is None:
        return float("nan")
    return float(effective_context_batch(weights[:, seq_len - 1, :], seq_len).mean())


def examine(path, args):
    """Measure 1 checkpoint and return its row, or None when it is not a temporal arm."""
    state = load_actor_state(path)
    architecture = identify_architecture(state)
    if architecture == "frame stacking":
        return None

    row = {
        "label": path.parent.parent.name or path.name,
        "architecture": architecture,
        "seed": seed_from_name(path),
        "context": float("nan"),
    }

    if architecture == "transformer":
        sub = extract_submodule_state(state, TRANSFORMER_PREFIX)
        geometry = infer_transformer_geometry(sub)
        module, report = build_transformer(geometry, sub)
        if report[0] or report[1]:
            raise RuntimeError("weight load was not clean")
        control, _ = build_transformer(geometry)
        forward = transformer_forward(module)
        control_forward = transformer_forward(control)
        input_size = geometry["input_size"]
    else:
        sub = extract_submodule_state(state, LSTM_PREFIX)
        weight_ih = sub["weight_ih_l0"]
        weight_hh = sub["weight_hh_l0"]
        input_size = int(weight_ih.shape[1])
        hidden_size = int(weight_hh.shape[1])
        layers = len({int(k.split("_l")[-1]) for k in sub if k.startswith("weight_ih_l")})
        module = nn.LSTM(input_size, hidden_size, layers, batch_first=True)
        missing, unexpected = module.load_state_dict(sub, strict=False)
        if missing or unexpected:
            raise RuntimeError("weight load was not clean")
        module.eval()
        control = nn.LSTM(input_size, hidden_size, layers, batch_first=True)
        control.eval()
        forward = recurrent_forward(module, layers, hidden_size)
        control_forward = recurrent_forward(control, layers, hidden_size)

    # The Gaussian sweep, where the structure is fixed and only the magnitude moves
    row["sweep"] = []
    for scale in SWEEP:
        generator = torch.Generator().manual_seed(args.seed)
        window = torch.randn(
            args.batch, args.seq_len, input_size, generator=generator
        ) * scale
        row["sweep"].append(
            ablation_ratio(forward, window, args.seq_len,
                           torch.Generator().manual_seed(args.seed))["ratio"]
        )

    # The checkpoint's own activation structure, rebuilt from its own encoders
    window, measured_rms = measured_window(
        state, args.seq_len, args.batch, args.occupancy, args.depth_max,
        args.grid_size, args.depth_size, args.seed,
    )
    row["rms"] = measured_rms
    if window is None:
        row["measured"] = float("nan")
        row["current"] = float("nan")
        row["control"] = float("nan")
        return row

    result = ablation_ratio(forward, window, args.seq_len,
                            torch.Generator().manual_seed(args.seed))
    row["measured"] = result["ratio"]
    row["current"] = result["current_relative"]
    row["history"] = result["history_relative"]
    row["control"] = ablation_ratio(
        control_forward, window, args.seq_len, torch.Generator().manual_seed(args.seed)
    )["ratio"]
    if architecture == "transformer":
        row["context"] = effective_context_of(module, window, args.seq_len)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--occupancy", type=float, default=0.15)
    parser.add_argument("--depth-max", type=float, default=1.0)
    parser.add_argument("--grid-size", type=int, default=9)
    parser.add_argument("--depth-size", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20261106)
    args = parser.parse_args()

    print("=" * 108)
    print("PROBE 06  GAUSSIAN AGAINST OWN STRUCTURE, PER SEED")
    print("=" * 108)
    print(f"paths  {', '.join(str(p) for p in args.paths)}")
    print(f"torch  {torch.__version__}")
    print(f"seed   {args.seed}")
    print()

    checkpoints = collect_checkpoints(args.paths)
    if not checkpoints:
        raise SystemExit("no checkpoints found under the supplied paths")

    rows = []
    seen = set()
    for path in checkpoints:
        label = path.parent.parent.name or path.name
        if label in seen:
            continue
        try:
            row = examine(path, args)
        except Exception as error:
            print(f"  {label}  failed: {type(error).__name__}: {error}")
            continue
        if row is None:
            continue
        seen.add(label)
        rows.append(row)
        print(f"  measured {label}")

    print()
    print()
    print(f"  {'run':<24} {'arch':<12} {'RMS':>8} {'G=1':>7} {'G=10':>7} {'G=100':>7} "
          f"{'own':>7} {'current':>8} {'control':>8} {'context':>8}")
    for row in rows:
        sweep = row["sweep"]
        context = f"{row['context']:.2f}" if row["context"] == row["context"] else "n/a"
        rms = f"{row['rms']:.4g}" if row.get("rms") is not None else "n/a"
        print(f"  {row['label'][:24]:<24} {row['architecture'][:12]:<12} {rms:>8} "
              f"{sweep[0]:>7.3f} {sweep[1]:>7.3f} {sweep[2]:>7.3f} "
              f"{row['measured']:>7.3f} {row['current']:>8.4f} {row['control']:>8.4f} "
              f"{context:>8}")
    print()

    print("=" * 108)


if __name__ == "__main__":
    main()
