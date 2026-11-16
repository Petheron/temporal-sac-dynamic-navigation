"""Separate magnitude from structure, and find where the large activation comes from.

Usage:
    python Algorithms\\probe_04_scaling.py "<path to a Transformer actor_dict_0>"
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
    from probe_03_gate_and_lstm import GateRecorder, ablation_ratio
except ImportError as error:
    raise SystemExit(
        "Could not import probe_02_attention_map.py or probe_03_gate_and_lstm.py.\n"
        "All 3 probe files must sit in the Algorithms folder and be run from the project root.\n"
        f"Original error: {error}"
    )


def rms(tensor):
    """Return the per element root mean square of a tensor."""
    return float(tensor.pow(2).mean().sqrt())


# ===== PART 1: WHERE THE MAGNITUDE IS PRODUCED =====

def report_stack_magnitudes(stack, tensors):
    """Walk the encoder stack and report the magnitude entering and leaving each stage.

    The gain column is the output magnitude divided by the input magnitude.
    A gain above 1 at several stages in a row is what produces a large final activation.
    """
    print("  stage                         input RMS     output RMS     gain   weight norm")
    parts = []
    for entry in tensors:
        if entry[0] == "vector":
            value = entry[1]
            print(f"  {'vector branch':<28}  {rms(value):>9.4f}  {rms(value):>13.4f}  "
                  f"{1.0:>7.3f}   {'none':>11}")
            parts.append(value)
            continue

        source, module = entry[1], entry[2]
        with torch.no_grad():
            convolved = module.conv_layers(source.permute(
                [0, 4, 1, 2, 3] if source.dim() == 5 else [0, 3, 1, 2]
            ))
            flattened = convolved.reshape(-1, module.final_flat)
            dense = module.dense_layers(flattened)

        kind = "conv3d branch" if source.dim() == 5 else "conv2d branch"
        with torch.no_grad():
            conv_weight = float(module.conv_layers[0].weight.norm())
            dense_weight = float(module.dense_layers[0].weight.norm())
        print(f"  {kind + ', convolutions':<28}  {rms(source):>9.4f}  {rms(flattened):>13.4f}  "
              f"{rms(flattened) / max(rms(source), 1e-12):>7.3f}   {conv_weight:>11.2f}")
        print(f"  {kind + ', dense':<28}  {rms(flattened):>9.4f}  {rms(dense):>13.4f}  "
              f"{rms(dense) / max(rms(flattened), 1e-12):>7.3f}   {dense_weight:>11.2f}")
        parts.append(dense)

    concatenated = torch.cat(parts, dim=-1)
    print(f"  {'concatenated':<28}  {'':>9}  {rms(concatenated):>13.4f}  "
          f"{'':>7}   {'none':>11}")

    current = concatenated
    trunk = stack["trunk"]
    for index, layer in enumerate(trunk.sequence):
        with torch.no_grad():
            output = layer(current)
        weight = getattr(layer, "weight", None)
        with torch.no_grad():
            weight_text = f"{float(weight.norm()):.2f}" if weight is not None else "none"
        name = type(layer).__name__
        print(f"  {'trunk ' + str(index) + ', ' + name:<28}  {rms(current):>9.4f}  "
              f"{rms(output):>13.4f}  {rms(output) / max(rms(current), 1e-12):>7.3f}   "
              f"{weight_text:>11}")
        current = output

    print()
    print(f"  trunk output, which is the transformer input, sits at RMS {rms(current):.4f}")
    print()
    return current


# ===== PART 2: THE SCALING PREDICTION =====

def attention_share(module, window):
    """Return the mean attention branch output divided by the mean residual, per attention gate.

    This is the quantity the architecture predicts should fall as 1 over the input magnitude.
    """
    with GateRecorder(module) as recorder:
        with torch.no_grad():
            module(window)

    shares = []
    for key in sorted(recorder.captured):
        index, name = key
        if name != "gate_attn":
            continue
        residual, sublayer, _output = recorder.captured[key]
        residual_norm = float(residual.norm(dim=-1).mean())
        shares.append(float(sublayer.norm(dim=-1).mean()) / max(residual_norm, 1e-12))
    return shares


def context_at(module, window, seq_len):
    """Return the effective context at the last query position in the final layer."""
    with AttentionRecorder(module) as recorder:
        with torch.no_grad():
            module(window)
    if not recorder.weights:
        return float("nan")
    final = max(recorder.weights)
    weights = recorder.weights[final]
    if weights is None:
        return float("nan")
    rows = weights[:, seq_len - 1, :]
    return float(effective_context_batch(rows, seq_len).mean())


def report_scaling(trained, control, geometry, args, measured, measured_rms, generator):
    """Sweep the input magnitude and report the attention share, context and history ratio."""
    print()
    header = ("  input RMS      share L0   share L1    product L0   context   "
              "history ratio   untrained ratio")
    print(header)

    scales = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    for scale in scales:
        window = torch.randn(
            args.batch, args.seq_len, geometry["input_size"], generator=generator
        ) * scale
        shares = attention_share(trained, window)
        context = context_at(trained, window, args.seq_len)
        trained_ratio = ablation_ratio(
            lambda w: trained(w)[:, -1, :], window, args.seq_len, generator
        )["ratio"]
        control_ratio = ablation_ratio(
            lambda w: control(w)[:, -1, :], window, args.seq_len, generator
        )["ratio"]
        share_0 = shares[0] if shares else float("nan")
        share_1 = shares[1] if len(shares) > 1 else float("nan")
        print(f"  {scale:<12g}  {share_0:>8.4f}  {share_1:>9.4f}   {share_0 * scale:>10.3f}   "
              f"{context:>7.2f}   {trained_ratio:>13.4f}   {control_ratio:>15.4f}")

    if measured is not None:
        shares = attention_share(trained, measured)
        context = context_at(trained, measured, args.seq_len)
        trained_ratio = ablation_ratio(
            lambda w: trained(w)[:, -1, :], measured, args.seq_len, generator
        )["ratio"]
        control_ratio = ablation_ratio(
            lambda w: control(w)[:, -1, :], measured, args.seq_len, generator
        )["ratio"]
        share_0 = shares[0] if shares else float("nan")
        share_1 = shares[1] if len(shares) > 1 else float("nan")
        print(f"  {measured_rms:<12.4g}  {share_0:>8.4f}  {share_1:>9.4f}   "
              f"{share_0 * measured_rms:>10.3f}   {context:>7.2f}   {trained_ratio:>13.4f}   "
              f"{control_ratio:>15.4f}   <-- real encoder output")
    print()


# ===== PART 3: MAGNITUDE AGAINST STRUCTURE =====

def report_two_by_two(trained, geometry, args, measured, measured_rms, generator):
    """Vary magnitude and structure 1 at a time, which is what stage 3 could not do.

    Real encoder output and Gaussian noise are each run at the measured magnitude and at 1.
    A ratio that tracks the magnitude column points at magnitude.
    A ratio that tracks the structure column points at structure.
    """
    if measured is None:
        print("  skipped, because the real encoder window is unavailable")
        print()
        return

    gaussian_unit = torch.randn(
        args.batch, args.seq_len, geometry["input_size"], generator=generator
    )
    # Rescaling preserves the structure and changes only the magnitude
    scaled_real = measured * (1.0 / max(measured_rms, 1e-12))
    scaled_gaussian = gaussian_unit * measured_rms

    cases = (
        ("real encoder output", "measured", measured),
        ("real encoder output", "1", scaled_real),
        ("Gaussian noise", "measured", scaled_gaussian),
        ("Gaussian noise", "1", gaussian_unit),
    )

    print("  structure              magnitude     actual RMS   share L0   context   history ratio")
    for structure, magnitude, window in cases:
        shares = attention_share(trained, window)
        context = context_at(trained, window, args.seq_len)
        ratio = ablation_ratio(
            lambda w: trained(w)[:, -1, :], window, args.seq_len, generator
        )["ratio"]
        share_0 = shares[0] if shares else float("nan")
        print(f"  {structure:<21}  {magnitude:<11}  {rms(window):>10.4g}   {share_0:>8.4f}   "
              f"{context:>7.2f}   {ratio:>13.4f}")
    print()


# ===== MAIN =====

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
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

    print("=" * 84)
    print("PROBE 04  MAGNITUDE AGAINST STRUCTURE")
    print("=" * 84)
    print(f"file   {args.checkpoint}")
    print(f"torch  {torch.__version__}")
    print(f"seed   {args.seed}")
    print()

    generator = torch.Generator().manual_seed(args.seed)
    state = load_actor_state(args.checkpoint)

    transformer_state = extract_submodule_state(state, TRANSFORMER_PREFIX)
    if not transformer_state:
        raise SystemExit(f"no keys found under the prefix {TRANSFORMER_PREFIX}")
    geometry = infer_transformer_geometry(transformer_state)
    trained, report = build_transformer(geometry, transformer_state)
    if report[0] or report[1]:
        raise SystemExit("transformer weight loading was not clean")
    control, _ = build_transformer(geometry)
    print(f"transformer loaded cleanly, d_model {geometry['d_model']}, "
          f"{geometry['n_layers']} layers")
    print()

    print("=" * 84)
    print("PART 1  WHERE THE LARGE ACTIVATION IS PRODUCED")
    print("=" * 84)

    obs_state = extract_submodule_state(state, OBS_ENCODER_PREFIX)
    trunk_state = extract_submodule_state(state, TRUNK_PREFIX)
    measured = None
    measured_rms = None

    if not obs_state or not trunk_state:
        print("  the encoder or trunk keys are absent, so this part cannot run")
        print()
    else:
        try:
            stack = build_encoder_stack(
                obs_state, trunk_state, TRUNK_ACTIVATIONS[0][1], args.grid_size, args.depth_size
            )
        except RuntimeError as error:
            print(f"  encoder rebuild failed: {error}")
            print()
            stack = None

        if stack is not None:
            local = torch.Generator().manual_seed(args.seed)
            tensors = sample_observations(
                stack, args.batch, args.occupancy, args.depth_max, local
            )
            report_stack_magnitudes(stack, tensors)

            driver = torch.Generator().manual_seed(args.seed)
            frames = []
            for _ in range(args.seq_len):
                inner = torch.Generator().manual_seed(
                    int(torch.randint(0, 2 ** 31 - 1, (1,), generator=driver))
                )
                frame_tensors = sample_observations(
                    stack, args.batch, args.occupancy, args.depth_max, inner
                )
                parts = []
                for entry in frame_tensors:
                    if entry[0] == "vector":
                        parts.append(entry[1])
                    else:
                        with torch.no_grad():
                            parts.append(entry[2](entry[1]))
                with torch.no_grad():
                    frames.append(stack["trunk"](torch.cat(parts, dim=-1)))
            measured = torch.stack(frames, dim=1)
            measured_rms = rms(measured)

    print("=" * 84)
    print("PART 2  THE SCALING PREDICTION, TESTED")
    print("=" * 84)
    report_scaling(trained, control, geometry, args, measured, measured_rms or 0.0, generator)

    print("=" * 84)
    print("PART 3  MAGNITUDE AGAINST STRUCTURE, VARIED 1 AT A TIME")
    print("=" * 84)
    report_two_by_two(trained, geometry, args, measured, measured_rms or 0.0, generator)

    print("=" * 84)


if __name__ == "__main__":
    main()
