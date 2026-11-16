"""Find out why the Transformer's history does not reach its output, and test the recurrent arm the same way.

Usage:
    python Algorithms\\probe_03_gate_and_lstm.py "<path to a Transformer actor_dict_0>"
    python Algorithms\\probe_03_gate_and_lstm.py "<Transformer actor_dict_0>" --lstm "<LSTM actor_dict_0>"
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn

try:
    from probe_02_attention_map import (
        load_actor_state,
        extract_submodule_state,
        build_encoder_stack,
        sample_observations,
        encode_batch,
        infer_transformer_geometry,
        build_transformer,
        effective_context_batch,
        AttentionRecorder,
        TRANSFORMER_PREFIX,
        OBS_ENCODER_PREFIX,
        TRUNK_PREFIX,
        TRUNK_ACTIVATIONS,
        N_HEADS,
    )
except ImportError as error:
    raise SystemExit(
        "Could not import probe_02_attention_map.py.\n"
        "Both probe files must sit in the Algorithms folder and be run from the project root.\n"
        f"Original error: {error}"
    )

# The recurrent module sits under this prefix inside a Long Short-Term Memory actor
LSTM_PREFIX = "body.lstm.lstm"

# GRUGate initialises its bias at this value, per the constructor default in Encoders.py
GATE_BIAS_INIT = 2.0


def measured_window(state, seq_len, batch, occupancy, depth_max, grid_size, depth_size, seed):
    """Build a window of real trunk activations from a checkpoint's own encoders.

    Returns the window and the measured per element root mean square, or None when
    the encoder keys are absent from the checkpoint.
    """
    obs_state = extract_submodule_state(state, OBS_ENCODER_PREFIX)
    trunk_state = extract_submodule_state(state, TRUNK_PREFIX)
    if not obs_state or not trunk_state:
        return None, None

    activation = TRUNK_ACTIVATIONS[0][1]
    try:
        stack = build_encoder_stack(obs_state, trunk_state, activation, grid_size, depth_size)
    except RuntimeError as error:
        print(f"  encoder rebuild failed: {error}")
        return None, None

    driver = torch.Generator().manual_seed(seed)
    frames = []
    for _ in range(seq_len):
        local = torch.Generator().manual_seed(
            int(torch.randint(0, 2 ** 31 - 1, (1,), generator=driver))
        )
        tensors = sample_observations(stack, batch, occupancy, depth_max, local)
        with torch.no_grad():
            _concat, frame = encode_batch(stack, tensors)
        frames.append(frame)
    window = torch.stack(frames, dim=1)
    return window, float(window.pow(2).mean().sqrt())


# ===== PART A: READ THE GATE PARAMETERS =====

def report_gate_parameters(module):
    """Print the learned gate bias of every gate, with no input involved at all.

    GRUGate computes z = sigmoid(Wz(y) + Uz(x) - bg) and returns (1 - z) * x + z * h.
    Here x is the residual carrying the current frame and y is the sublayer output.
    A large bg pushes z toward 0, which returns the residual and discards the sublayer.
    The bias is initialised at 2.0, so a value above that means training pushed the gate further shut.
    """
    print("  layer  gate         bias mean   bias min   bias max   implied z at zero input")
    rows = []
    for index, layer in enumerate(module.layers):
        for name in ("gate_attn", "gate_ff"):
            gate = getattr(layer, name, None)
            if gate is None:
                continue
            bias = gate.bg.detach()
            mean = float(bias.mean())
            # With no contribution from Wz and Uz, z reduces to sigmoid of the negated bias
            implied = 1.0 / (1.0 + math.exp(mean))
            print(f"  {index:<6} {name:<12} {mean:>9.4f}  {float(bias.min()):>9.4f}  "
                  f"{float(bias.max()):>9.4f}  {implied:>22.4f}")
            rows.append((index, name, mean, implied))
    print()
    return rows


# ===== PART B: MEASURE THE GATE OPENING ON REAL ACTIVATIONS =====

class GateRecorder:
    """Capture the residual and the sublayer output arriving at every gate."""

    def __init__(self, module):
        self.module = module
        self.handles = []
        self.captured = {}

    def _make_hook(self, key):
        def hook(_module, inputs, output):
            # GRUGate.forward(x, y) takes the residual first and the sublayer output second
            self.captured[key] = (inputs[0].detach(), inputs[1].detach(), output.detach())
        return hook

    def __enter__(self):
        for index, layer in enumerate(self.module.layers):
            for name in ("gate_attn", "gate_ff"):
                gate = getattr(layer, name, None)
                if gate is None:
                    continue
                self.handles.append(gate.register_forward_hook(self._make_hook((index, name))))
        return self

    def __exit__(self, *_args):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def report_gate_behaviour(module, window):
    """Print what each gate actually does when real activations pass through it."""
    with GateRecorder(module) as recorder:
        with torch.no_grad():
            module(window)

    if not recorder.captured:
        print("  no gates found, so this network was built without gating")
        print()
        return []

    print()
    print("  layer  gate         mean z    median z   sublayer size   movement")
    rows = []
    for key in sorted(recorder.captured):
        index, name = key
        residual, sublayer, output = recorder.captured[key]
        gate = getattr(module.layers[index], name)
        with torch.no_grad():
            z = torch.sigmoid(gate.Wz(sublayer) + gate.Uz(residual) - gate.bg)
        mean_z = float(z.mean())
        median_z = float(z.median())
        residual_norm = float(residual.norm(dim=-1).mean())
        sublayer_size = float(sublayer.norm(dim=-1).mean()) / max(residual_norm, 1e-12)
        movement = float((output - residual).norm(dim=-1).mean()) / max(residual_norm, 1e-12)
        print(f"  {index:<6} {name:<12} {mean_z:>7.4f}  {median_z:>10.4f}  "
              f"{sublayer_size:>13.4f}  {movement:>9.4f}")
        rows.append({"layer": index, "gate": name, "mean_z": mean_z,
                     "sublayer_size": sublayer_size, "movement": movement})
    print()
    return rows


# ===== PART C: ABLATION ACROSS INPUT MAGNITUDES =====

def ablation_ratio(forward, window, seq_len, generator):
    """Return how much the last output depends on the history against the current frame.

    The forward argument is a callable mapping a window to an output at the last position.
    Both swaps keep the window length identical, so only the swapped content differs.
    """
    history = seq_len - 1
    permutation = torch.randperm(window.shape[0], generator=generator)

    swapped_history = window.clone()
    swapped_history[:, :history, :] = window[permutation][:, :history, :]

    swapped_current = window.clone()
    swapped_current[:, history, :] = window[permutation][:, history, :]

    with torch.no_grad():
        full = forward(window)
        no_history = forward(swapped_history)
        no_current = forward(swapped_current)

    baseline = float(full.norm(dim=-1).mean())
    history_effect = float((full - no_history).norm(dim=-1).mean())
    current_effect = float((full - no_current).norm(dim=-1).mean())
    return {
        "baseline": baseline,
        "history": history_effect,
        "current": current_effect,
        "history_relative": history_effect / max(baseline, 1e-12),
        "current_relative": current_effect / max(baseline, 1e-12),
        "ratio": history_effect / current_effect if current_effect > 0 else float("nan"),
    }


def report_scale_sweep(module, control, geometry, args, measured, measured_rms, generator):
    """Re-run the history ablation at several input magnitudes.

    probe_02 measured the ratio at 1 magnitude, which left it open to the charge
    that the result was an artefact of the magnitude chosen.
    """
    def forward(window):
        return module(window)[:, -1, :]

    print("  input RMS        history relative   current relative   ratio")
    scales = [0.01, 0.1, 1.0, 10.0, 100.0]
    for scale in scales:
        window = torch.randn(
            args.batch, args.seq_len, geometry["input_size"], generator=generator
        ) * scale
        result = ablation_ratio(forward, window, args.seq_len, generator)
        print(f"  {scale:<14g}  {result['history_relative']:>16.4f}   "
              f"{result['current_relative']:>16.4f}   {result['ratio']:>7.4f}")

    if measured is not None:
        result = ablation_ratio(forward, measured, args.seq_len, generator)
        print(f"  {measured_rms:<14.4g}  {result['history_relative']:>16.4f}   "
              f"{result['current_relative']:>16.4f}   {result['ratio']:>7.4f}"
              "   <-- the measured real activation")

    # An untrained network at the same geometry gives the reading to judge the rest against
    if control is not None:
        reference = measured if measured is not None else torch.randn(
            args.batch, args.seq_len, geometry["input_size"], generator=generator
        )
        result = ablation_ratio(lambda w: control(w)[:, -1, :], reference, args.seq_len, generator)
        print(f"  {'untrained':<14}  {result['history_relative']:>16.4f}   "
              f"{result['current_relative']:>16.4f}   {result['ratio']:>7.4f}"
              "   <-- untrained control at the same geometry")
    print()


# ===== PART D: THE RECURRENT ARM =====

def build_lstm(state):
    """Rebuild the recurrent module from its weight shapes and load the trained weights."""
    sub = extract_submodule_state(state, LSTM_PREFIX)
    if not sub:
        return None, None

    weight_ih = sub.get("weight_ih_l0")
    weight_hh = sub.get("weight_hh_l0")
    if weight_ih is None or weight_hh is None:
        return None, None

    input_size = int(weight_ih.shape[1])
    hidden_size = int(weight_hh.shape[1])
    layer_count = len({
        int(k.split("_l")[-1]) for k in sub if k.startswith("weight_ih_l")
    })

    module = nn.LSTM(input_size, hidden_size, layer_count, batch_first=True)
    missing, unexpected = module.load_state_dict(sub, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"recurrent weight load left {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
    module.eval()
    return module, {"input_size": input_size, "hidden_size": hidden_size, "layers": layer_count}


def report_lstm(module, geometry, args, measured, measured_rms, generator):
    """Run the same history ablation on the recurrent arm.

    The hidden state starts at zero, which matches the Transformer's fixed window.
    In deployment the recurrent state persists beyond the window, so a short test here
    understates how much history the recurrent arm could in principle reach.
    A short test that still finds no dependence is therefore the stronger result.
    """
    def forward(window):
        batch = window.shape[0]
        zeros = torch.zeros(geometry["layers"], batch, geometry["hidden_size"])
        out, _ = module(window, (zeros, zeros))
        return out[:, -1, :]

    print("  sequence  input RMS       history relative   current relative   ratio")
    for seq_len in (args.seq_len, args.seq_len * 2, args.seq_len * 5):
        for scale in (1.0, 100.0):
            window = torch.randn(
                args.batch, seq_len, geometry["input_size"], generator=generator
            ) * scale
            result = ablation_ratio(forward, window, seq_len, generator)
            print(f"  {seq_len:<8}  {scale:<14g}  {result['history_relative']:>16.4f}   "
                  f"{result['current_relative']:>16.4f}   {result['ratio']:>7.4f}")

    if measured is not None:
        result = ablation_ratio(forward, measured, args.seq_len, generator)
        print(f"  {args.seq_len:<8}  {measured_rms:<14.4g}  {result['history_relative']:>16.4f}   "
              f"{result['current_relative']:>16.4f}   {result['ratio']:>7.4f}"
              "   <-- the measured real activation")

    # An untrained recurrent module at the same geometry is the reference point
    control = nn.LSTM(geometry["input_size"], geometry["hidden_size"],
                      geometry["layers"], batch_first=True)
    control.eval()

    def control_forward(window):
        batch = window.shape[0]
        zeros = torch.zeros(geometry["layers"], batch, geometry["hidden_size"])
        out, _ = control(window, (zeros, zeros))
        return out[:, -1, :]

    reference = measured if measured is not None else torch.randn(
        args.batch, args.seq_len, geometry["input_size"], generator=generator
    )
    result = ablation_ratio(control_forward, reference, args.seq_len, generator)
    print(f"  {args.seq_len:<8}  {'untrained':<14}  {result['history_relative']:>16.4f}   "
          f"{result['current_relative']:>16.4f}   {result['ratio']:>7.4f}"
          "   <-- untrained control at the same geometry")
    print()


# ===== MAIN =====

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--lstm", type=Path, default=None)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--occupancy", type=float, default=0.15)
    parser.add_argument("--depth-max", type=float, default=1.0)
    parser.add_argument("--grid-size", type=int, default=9)
    parser.add_argument("--depth-size", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20261106)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"Not a file: {args.checkpoint}")
    if args.lstm is not None and not args.lstm.is_file():
        raise SystemExit(f"Not a file: {args.lstm}")

    print("=" * 76)
    print("PROBE 03  GATE READOUT AND RECURRENT COMPARISON")
    print("=" * 76)
    print(f"transformer  {args.checkpoint}")
    print(f"recurrent    {args.lstm if args.lstm is not None else 'not supplied'}")
    print(f"torch        {torch.__version__}")
    print(f"seed         {args.seed}")
    print()

    generator = torch.Generator().manual_seed(args.seed)

    state = load_actor_state(args.checkpoint)
    transformer_state = extract_submodule_state(state, TRANSFORMER_PREFIX)
    if not transformer_state:
        raise SystemExit(f"no keys found under the prefix {TRANSFORMER_PREFIX}")

    geometry = infer_transformer_geometry(transformer_state)
    trained, report = build_transformer(geometry, transformer_state)
    if report[0] or report[1]:
        raise SystemExit("transformer weight loading was not clean, so nothing below would be trustworthy")
    print(f"transformer loaded cleanly, {len(transformer_state)} tensors, "
          f"d_model {geometry['d_model']}, {geometry['n_layers']} layers")
    print()

    measured, measured_rms = measured_window(
        state, args.seq_len, args.batch, args.occupancy, args.depth_max,
        args.grid_size, args.depth_size, args.seed,
    )
    if measured is not None:
        print(f"measured real activation window {tuple(measured.shape)} at RMS {measured_rms:.1f}")
    else:
        print("the encoder keys are absent, so the measured activation is unavailable")
    print()

    print("=" * 76)
    print("PART A  GATE PARAMETERS, READ STRAIGHT FROM THE CHECKPOINT")
    print("=" * 76)
    report_gate_parameters(trained)

    print("=" * 76)
    print("PART B  GATE BEHAVIOUR ON REAL ACTIVATIONS")
    print("=" * 76)
    if measured is not None:
        report_gate_behaviour(trained, measured)
    else:
        print("  skipped, because the measured activation is unavailable")
        print()

    print("=" * 76)
    print("PART C  HISTORY ABLATION ACROSS INPUT MAGNITUDES")
    print("=" * 76)
    print()
    control, _ = build_transformer(geometry)
    report_scale_sweep(trained, control, geometry, args, measured, measured_rms or 0.0, generator)

    print("=" * 76)
    print("PART D  THE RECURRENT ARM, TESTED THE SAME WAY")
    print("=" * 76)
    if args.lstm is None:
        print("  no recurrent checkpoint supplied, so this comparison was not run")
        print("  rerun with --lstm \"<path to an LSTM actor_dict_0>\" to include it")
        print()
    else:
        lstm_state = load_actor_state(args.lstm)
        lstm_module, lstm_geometry = build_lstm(lstm_state)
        if lstm_module is None:
            print(f"  no keys found under the prefix {LSTM_PREFIX}, so this is not a recurrent actor")
            print()
        else:
            print(f"  recurrent module loaded cleanly, input {lstm_geometry['input_size']}, "
                  f"hidden {lstm_geometry['hidden_size']}, {lstm_geometry['layers']} layer(s)")
            print()
            lstm_measured, lstm_rms = measured_window(
                lstm_state, args.seq_len, args.batch, args.occupancy, args.depth_max,
                args.grid_size, args.depth_size, args.seed,
            )
            report_lstm(lstm_module, lstm_geometry, args, lstm_measured, lstm_rms or 0.0, generator)

    print("=" * 76)


if __name__ == "__main__":
    main()
