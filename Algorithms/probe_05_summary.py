"""Verify all 3 architectures across every checkpoint supplied, in 1 run.

Usage:
    python Algorithms\\probe_05_summary.py "Results\\SortMe"
    python Algorithms\\probe_05_summary.py "Results\\SortMe\\tf_d_t030_s3_197815\\SAC_DRLNavigation\\actor_dict_0"
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn

try:
    from Encoders import Conv2DEncoder, Conv3DEncoder, LinearEncoder
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
        N_HEADS,
    )
except ImportError as error:
    raise SystemExit(
        "Could not import Encoders.py or probe_02_attention_map.py.\n"
        "Run this from the project root with probe_02_attention_map.py in the Algorithms folder.\n"
        f"Original error: {error}"
    )

LSTM_PREFIX = "body.lstm.lstm"
GATE_BIAS_INIT = 2.0

# The run naming convention, used only to cross check the detected architecture
NAME_PREFIXES = {"bl": "frame stacking", "ls": "recurrent", "tf": "transformer"}


def rms(tensor):
    """Return the per element root mean square of a tensor."""
    return float(tensor.pow(2).mean().sqrt())


def relative_change(reference, altered):
    """Return the mean change between 2 batches of vectors, divided by the reference size."""
    baseline = float(reference.norm(dim=-1).mean())
    moved = float((reference - altered).norm(dim=-1).mean())
    return moved / max(baseline, 1e-12)


# ===== DISCOVERY AND IDENTIFICATION =====

def collect_checkpoints(paths):
    """Expand every supplied path into a sorted list of checkpoint files."""
    found = []
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            found.append(path)
        elif path.is_dir():
            found.extend(sorted(path.glob("**/actor_dict_0")))
            found.extend(sorted(path.glob("**/actor_dict_0.pt")))
        else:
            print(f"  skipped, not a file or directory: {path}")
    # Preserve order while removing duplicates
    unique = []
    for path in found:
        if path not in unique:
            unique.append(path)
    return unique


def identify_architecture(state):
    """Return the architecture implied by the stored keys, not by the folder name."""
    if any(k.startswith(TRANSFORMER_PREFIX + ".") for k in state):
        return "transformer"
    if any(k.startswith(LSTM_PREFIX + ".") for k in state):
        return "recurrent"
    return "frame stacking"


def expected_from_name(path):
    """Return the architecture the run name implies, or None when the name does not say."""
    for part in path.parts:
        prefix = part.split("_")[0].lower()
        if prefix in NAME_PREFIXES:
            return NAME_PREFIXES[prefix]
    return None


def seed_from_name(path):
    """Return the seed in the run name, or None when the name does not carry one."""
    for part in path.parts:
        for token in part.split("_"):
            if token.startswith("s") and token[1:].isdigit():
                return int(token[1:])
    return None


# ===== ENCODER STACK, TRAINED AND UNTRAINED =====

def build_untrained_twin(stack, activation_type):
    """Rebuild the encoder stack at identical geometry without loading any trained weights.

    This is the control. Every constructor argument is taken from the trained stack,
    so the only difference between the 2 is training.
    """
    modules = []
    for entry in stack["modules"]:
        if entry[0] == "vector":
            modules.append(("vector", entry[1]))
            continue
        group = entry[1]
        chosen = group["chosen"]
        side = chosen["input"]
        if group["kind"] == "conv3d":
            module = Conv3DEncoder(
                input_shape=(side, side, side, group["in_channels"]),
                output_size=group["output_size"],
                init_channel=group["init_channel"],
                init_kernel_size=group["kernel"],
                init_stride=chosen["stride"],
                init_padding=chosen["padding"],
                num_convs=group["n_convs"],
                use_embedding=False,
                use_visual_dense_layers=True,
            )
        else:
            module = Conv2DEncoder(
                input_shape=(side, side, group["in_channels"]),
                output_size=group["output_size"],
                init_channel=group["init_channel"],
                init_kernel_size=group["kernel"],
                init_stride=chosen["stride"],
                init_padding=chosen["padding"],
                num_convs=group["n_convs"],
                use_embedding=False,
                use_visual_dense_layers=True,
            )
        module.eval()
        modules.append(("conv", group, module))

    trunk = LinearEncoder(
        input_size=stack["trunk_input"],
        num_layers=stack["num_layers"],
        hidden_size=stack["hidden_size"],
        activation_type=activation_type,
    )
    trunk.eval()
    return {"modules": modules, "trunk": trunk, "trunk_input": stack["trunk_input"],
            "hidden_size": stack["hidden_size"], "num_layers": stack["num_layers"],
            "geometry": stack["geometry"]}


def raw_observations(stack, batch, occupancy, depth_max, generator):
    """Draw 1 batch of observations shaped for this stack, with no modules attached.

    The modules are kept separate so the same observations can be run through
    the trained stack and the untrained twin without resampling.
    """
    drawn = []
    for entry in stack["modules"]:
        if entry[0] == "vector":
            group = entry[1]
            raw = torch.randn(batch, group["size"], generator=generator)
            drawn.append(("vector", torch.clamp(raw, -5.0, 5.0)))
            continue
        group = entry[1]
        side = group["chosen"]["input"]
        channels = group["in_channels"]
        if group["kind"] == "conv3d":
            noise = torch.rand(batch, side, side, side, channels, generator=generator)
            drawn.append(("conv3d", (noise < occupancy).float()))
        else:
            noise = torch.rand(batch, side, side, channels, generator=generator)
            drawn.append(("conv2d", noise * depth_max))
    return drawn


def forward_stack(stack, drawn):
    """Run 1 batch of observations through a stack's encoders and trunk."""
    parts = []
    conv_index = 0
    for entry in stack["modules"]:
        if entry[0] == "vector":
            parts.append(drawn[conv_index][1])
        else:
            with torch.no_grad():
                parts.append(entry[2](drawn[conv_index][1]))
        conv_index += 1
    concatenated = torch.cat(parts, dim=-1)
    with torch.no_grad():
        return stack["trunk"](concatenated)


def stack_count(stack):
    """Return how many frames the observation carries, read from the convolution channels."""
    for entry in stack["modules"]:
        if entry[0] == "conv":
            return entry[1]["in_channels"]
    return 1


# ===== THE HISTORY MEASUREMENT (1 DEFINITION PER ARCHITECTURE) =====

def window_history(forward, window, seq_len, generator):
    """Return how much the last output depends on the earlier window positions.

    Used for the recurrent and transformer arms, where history lives in the window.
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

    history_relative = relative_change(full, no_history)
    current_relative = relative_change(full, no_current)
    return {
        "history": history_relative,
        "current": current_relative,
        "ratio": history_relative / current_relative if current_relative > 0 else float("nan"),
    }


def swap_frame(drawn, other, frame, vector_width):
    """Return a copy of 1 batch of observations with 1 stacked frame replaced.

    A stacked frame occupies 1 convolution channel in each sensor branch and
    1 contiguous block of the vector branch.
    """
    altered = []
    for (kind, tensor), (_other_kind, other_tensor) in zip(drawn, other):
        copy = tensor.clone()
        if kind == "vector":
            start = frame * vector_width
            copy[:, start:start + vector_width] = other_tensor[:, start:start + vector_width]
        elif kind == "conv3d":
            copy[:, :, :, :, frame] = other_tensor[:, :, :, :, frame]
        else:
            copy[:, :, :, frame] = other_tensor[:, :, :, frame]
        altered.append((kind, copy))
    return altered


def stacked_history(stack, drawn, other, frames, vector_width):
    """Measure the per frame sensitivity of a frame stacking arm, and the history ratio.

    The current frame is identified empirically as the most influential one,
    so no assumption is made about whether the newest frame is stored first or last.
    """
    reference = forward_stack(stack, drawn)
    per_frame = []
    for frame in range(frames):
        altered = swap_frame(drawn, other, frame, vector_width)
        per_frame.append(relative_change(reference, forward_stack(stack, altered)))

    current = max(range(frames), key=lambda f: per_frame[f])

    all_but_current = [(kind, tensor.clone()) for kind, tensor in drawn]
    for frame in range(frames):
        if frame == current:
            continue
        all_but_current = swap_frame(all_but_current, other, frame, vector_width)
    history_relative = relative_change(reference, forward_stack(stack, all_but_current))

    current_relative = per_frame[current]
    return {
        "per_frame": per_frame,
        "current_index": current,
        "history": history_relative,
        "current": current_relative,
        "ratio": history_relative / current_relative if current_relative > 0 else float("nan"),
    }


# ===== VERIFICATION CHECKS =====

def check_causal_mask(module, seq_len, input_size, generator):
    """Confirm no position can see the future, and that the present does reach it.

    A network that output a constant would pass the first half of this on its own,
    so the second half is checked as well.
    """
    base = torch.randn(2, seq_len, input_size, generator=generator)
    with torch.no_grad():
        reference = module(base)
    for position in range(1, seq_len):
        perturbed = base.clone()
        perturbed[:, position, :] += 10.0
        with torch.no_grad():
            output = module(perturbed)
        leaked = float((output[:, :position, :] - reference[:, :position, :]).abs().max())
        propagated = float((output[:, position:, :] - reference[:, position:, :]).abs().max())
        if leaked > 1e-5:
            return False, f"position {position} leaked backwards at {leaked:.2e}"
        if propagated < 1e-6:
            return False, f"position {position} did not propagate forwards"
    return True, "no future leakage, forward propagation intact"


def check_gates(module):
    """Confirm the gates are not pinned shut by their learned bias."""
    biases = []
    for layer in module.layers:
        gate = getattr(layer, "gate_attn", None)
        if gate is not None:
            with torch.no_grad():
                biases.append(float(gate.bg.mean()))
    if not biases:
        return True, "no gates in this configuration"
    worst = max(biases)
    # A bias far above the initialisation would drive the gate opening toward 0
    if worst > GATE_BIAS_INIT + 2.0:
        return False, f"attention gate bias reached {worst:.3f} against an initialisation of {GATE_BIAS_INIT:.1f}"
    return True, f"attention gate bias {min(biases):.3f} to {worst:.3f}, initialised at {GATE_BIAS_INIT:.1f}"


# ===== PER CHECKPOINT =====

def examine(path, args):
    """Measure 1 checkpoint and return a row of results plus a detail block."""
    row = {"path": path, "seed": seed_from_name(path), "detail": [], "checks": []}
    state = load_actor_state(path)
    architecture = identify_architecture(state)
    row["architecture"] = architecture

    expected = expected_from_name(path)
    if expected is not None and expected != architecture:
        row["checks"].append((False, f"the run name implies {expected} but the keys hold {architecture}"))
    elif expected is not None:
        row["checks"].append((True, f"the run name and the stored keys both say {architecture}"))

    generator = torch.Generator().manual_seed(args.seed)

    obs_state = extract_submodule_state(state, OBS_ENCODER_PREFIX)
    trunk_state = extract_submodule_state(state, TRUNK_PREFIX)
    stack = None
    twin = None
    if obs_state and trunk_state:
        try:
            stack = build_encoder_stack(
                obs_state, trunk_state, TRUNK_ACTIVATIONS[0][1], args.grid_size, args.depth_size
            )
            twin = build_untrained_twin(stack, TRUNK_ACTIVATIONS[0][1])
        except RuntimeError as error:
            row["detail"].append(f"encoder rebuild failed: {error}")

    frames = stack_count(stack) if stack is not None else None
    row["frames"] = frames

    # Build a window of real trunk activations, 1 frame at a time
    window = None
    twin_window = None
    if stack is not None:
        driver = torch.Generator().manual_seed(args.seed)
        trained_frames = []
        control_frames = []
        for _ in range(args.seq_len):
            inner = torch.Generator().manual_seed(
                int(torch.randint(0, 2 ** 31 - 1, (1,), generator=driver))
            )
            drawn = raw_observations(stack, args.batch, args.occupancy, args.depth_max, inner)
            trained_frames.append(forward_stack(stack, drawn))
            control_frames.append(forward_stack(twin, drawn))
        window = torch.stack(trained_frames, dim=1)
        twin_window = torch.stack(control_frames, dim=1)
        row["trunk_rms"] = rms(window)
    else:
        row["trunk_rms"] = None

    if architecture == "transformer":
        transformer_state = extract_submodule_state(state, TRANSFORMER_PREFIX)
        geometry = infer_transformer_geometry(transformer_state)
        trained, report = build_transformer(geometry, transformer_state)
        control, _ = build_transformer(geometry)
        row["checks"].append((
            report[0] == 0 and report[1] == 0,
            f"weight load, {len(transformer_state)} tensors, {report[0]} missing, {report[1]} unexpected",
        ))
        row["checks"].append(check_causal_mask(
            trained, args.seq_len, geometry["input_size"], torch.Generator().manual_seed(args.seed)
        ))
        row["checks"].append(check_gates(trained))
        row["detail"].append(
            f"geometry d_model {geometry['d_model']}, {geometry['n_layers']} layers, "
            f"max_seq_len {geometry['max_seq_len']}, input {geometry['input_size']}"
        )

        if window is not None:
            trained_result = window_history(
                lambda w: trained(w)[:, -1, :], window, args.seq_len,
                torch.Generator().manual_seed(args.seed),
            )
            control_result = window_history(
                lambda w: control(w)[:, -1, :], window, args.seq_len,
                torch.Generator().manual_seed(args.seed),
            )
            row["trained"] = trained_result
            row["control"] = control_result

            with AttentionRecorder(trained) as recorder:
                with torch.no_grad():
                    trained(window)
            final = max(recorder.weights)
            rows_at_last = recorder.weights[final][:, args.seq_len - 1, :]
            row["context"] = float(effective_context_batch(rows_at_last, args.seq_len).mean())

    elif architecture == "recurrent":
        sub = extract_submodule_state(state, LSTM_PREFIX)
        weight_ih = sub["weight_ih_l0"]
        weight_hh = sub["weight_hh_l0"]
        input_size = int(weight_ih.shape[1])
        hidden_size = int(weight_hh.shape[1])
        layers = len({int(k.split("_l")[-1]) for k in sub if k.startswith("weight_ih_l")})
        trained = nn.LSTM(input_size, hidden_size, layers, batch_first=True)
        missing, unexpected = trained.load_state_dict(sub, strict=False)
        trained.eval()
        control = nn.LSTM(input_size, hidden_size, layers, batch_first=True)
        control.eval()
        row["checks"].append((
            len(missing) == 0 and len(unexpected) == 0,
            f"weight load, {len(sub)} tensors, {len(missing)} missing, {len(unexpected)} unexpected",
        ))
        row["detail"].append(
            f"geometry input {input_size}, hidden {hidden_size}, {layers} layer(s)"
        )

        def make_forward(module):
            def forward(w):
                zeros = torch.zeros(layers, w.shape[0], hidden_size)
                out, _ = module(w, (zeros, zeros))
                return out[:, -1, :]
            return forward

        if window is not None:
            row["trained"] = window_history(
                make_forward(trained), window, args.seq_len,
                torch.Generator().manual_seed(args.seed),
            )
            row["control"] = window_history(
                make_forward(control), window, args.seq_len,
                torch.Generator().manual_seed(args.seed),
            )

    else:
        row["detail"].append("no temporal module, so history is carried in the stacked observation")
        if stack is not None and frames is not None and frames > 1:
            vector_width = None
            for entry in stack["modules"]:
                if entry[0] == "vector":
                    vector_width = entry[1]["size"] // frames
            if vector_width is None:
                vector_width = 0
            inner = torch.Generator().manual_seed(args.seed)
            drawn = raw_observations(stack, args.batch, args.occupancy, args.depth_max, inner)
            other = raw_observations(
                stack, args.batch, args.occupancy, args.depth_max,
                torch.Generator().manual_seed(args.seed + 101),
            )
            row["trained"] = stacked_history(stack, drawn, other, frames, vector_width)
            row["control"] = stacked_history(twin, drawn, other, frames, vector_width)
            row["detail"].append(
                f"{frames} stacked frames, vector block {vector_width} wide per frame"
            )
        elif stack is not None:
            row["detail"].append(
                f"the observation carries {frames} frame, so there is no stacked history to test"
            )

    return row


# ===== MAIN =====

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
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many checkpoints, 0 for no limit")
    args = parser.parse_args()

    print("=" * 100)
    print("PROBE 05  ALL ARCHITECTURES, ALL SUPPLIED CHECKPOINTS")
    print("=" * 100)
    print(f"paths  {', '.join(str(p) for p in args.paths)}")
    print(f"torch  {torch.__version__}")
    print(f"seed   {args.seed}")
    print()

    checkpoints = collect_checkpoints(args.paths)
    if args.limit:
        checkpoints = checkpoints[: args.limit]
    if not checkpoints:
        raise SystemExit("no checkpoints found under the supplied paths")
    print(f"found {len(checkpoints)} checkpoint(s)")
    print()

    rows = []
    for index, path in enumerate(checkpoints, start=1):
        label = path.parent.parent.name or path.name
        print("-" * 100)
        print(f"[{index} of {len(checkpoints)}]  {label}")
        print("-" * 100)
        try:
            row = examine(path, args)
        except Exception as error:
            print(f"  failed: {type(error).__name__}: {error}")
            print()
            continue

        row["label"] = label
        print(f"  architecture       {row['architecture']}")
        if row.get("seed") is not None:
            print(f"  seed               {row['seed']}")
        if row.get("trunk_rms") is not None:
            print(f"  trunk output RMS   {row['trunk_rms']:.4g}")
        for line in row["detail"]:
            print(f"  {line}")
        for passed, message in row["checks"]:
            print(f"  [{'PASS' if passed else 'FAIL'}]  {message}")

        trained = row.get("trained")
        control = row.get("control")
        if trained is not None and control is not None:
            print(f"  history dependence, trained    {trained['history']:.4f} relative, "
                  f"ratio {trained['ratio']:.4f}")
            print(f"  history dependence, untrained  {control['history']:.4f} relative, "
                  f"ratio {control['ratio']:.4f}")
            if control["ratio"] > 0:
                print(f"  trained over untrained         {trained['ratio'] / control['ratio']:.2f}")
            if row.get("context") is not None:
                print(f"  attention effective context    {row['context']:.2f} of {args.seq_len}")
            if "per_frame" in trained:
                per = " ".join(f"{v:.4f}" for v in trained["per_frame"])
                print(f"  per frame sensitivity          {per}")
                print(f"  most influential frame         index {trained['current_index']}")
        print()
        rows.append(row)

    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"  {'run':<28} {'architecture':<15} {'trunk RMS':>10} {'trained':>9} "
          f"{'untrained':>10} {'factor':>8} {'context':>8}")
    for row in rows:
        trained = row.get("trained")
        control = row.get("control")
        trunk = f"{row['trunk_rms']:.4g}" if row.get("trunk_rms") is not None else "n/a"
        trained_text = f"{trained['ratio']:.4f}" if trained else "n/a"
        control_text = f"{control['ratio']:.4f}" if control else "n/a"
        if trained and control and control["ratio"] > 0:
            factor = f"{trained['ratio'] / control['ratio']:.2f}"
        else:
            factor = "n/a"
        context = f"{row['context']:.2f}" if row.get("context") is not None else "n/a"
        print(f"  {row['label'][:28]:<28} {row['architecture']:<15} {trunk:>10} "
              f"{trained_text:>9} {control_text:>10} {factor:>8} {context:>8}")
    print()

    failures = [(r, m) for r in rows for passed, m in r["checks"] if not passed]
    print("=" * 100)
    print("IMPLEMENTATION VERDICT")
    print("=" * 100)
    if failures:
        print(f"  {len(failures)} check(s) failed:")
        for row, message in failures:
            print(f"    {row['label']}  {message}")
    else:
        print(f"  Every structural check passed on all {len(rows)} checkpoint(s).")
    print()
    print("=" * 100)


if __name__ == "__main__":
    main()
