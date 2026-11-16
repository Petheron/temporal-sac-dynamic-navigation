"""Read the trained Transformer actor's attention distribution directly.

Usage:
    python Algorithms\\probe_02_attention_map.py "<path to actor_dict_0>"
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from Encoders import GTrXL, Conv2DEncoder, Conv3DEncoder, LinearEncoder
    from EncoderUtils import ActivationType
except ImportError as error:
    raise SystemExit(
        "Could not import from Encoders.py or EncoderUtils.py.\n"
        "Run this from the project root so those modules are importable.\n"
        f"Original error: {error}"
    )


# The transformer lives under this prefix inside the actor state dictionary
TRANSFORMER_PREFIX = "body.transformer"

# The observation encoders and the trunk live under these prefixes
OBS_ENCODER_PREFIX = "body.observation_encoder"
TRUNK_PREFIX = "body.body_encoder"

# These were confirmed from the configuration and from probe_01_attention.py
N_HEADS = 4
D_FF_MULTIPLIER = 4
USE_GRU_GATING = True

# The trunk activation type is not recorded in a state dictionary
# All 3 plausible choices are measured so the spread is visible
TRUNK_ACTIVATIONS = (
    ("swish", ActivationType.Swish),
    ("leakyrelu", ActivationType.LeakyReLU),
    ("relu", ActivationType.ReLU),
)


# ===== LOADING =====

def load_actor_state(path):
    """Load a checkpoint and return the plain state dictionary inside it."""
    loaded = torch.load(str(path), map_location="cpu")
    if not isinstance(loaded, dict):
        raise SystemExit(f"Unexpected checkpoint type: {type(loaded)}")
    for wrapper_key in ("state_dict", "model_state_dict", "actor"):
        inner = loaded.get(wrapper_key)
        if isinstance(inner, dict):
            return inner
    return loaded


def extract_submodule_state(state, prefix):
    """Return every tensor under a prefix, with the prefix stripped from each key."""
    cut = len(prefix) + 1
    return {k[cut:]: v for k, v in state.items() if k.startswith(prefix + ".")}


# ===== PART A: REBUILD THE ENCODER STACK AND MEASURE THE REAL ACTIVATION =====

def conv_out_size(size, kernel, stride, padding, dilation=1):
    """Return the output size of 1 convolution dimension."""
    return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def solve_conv_geometry(kernel, n_convs, n_dims, target_flat, out_channels,
                        input_size, max_padding=None):
    """Find every stride and padding reproducing the observed flattened size.

    The kernel, the convolution count and the channel counts come from the weight shapes.
    The input size is supplied by the caller from the documented sensor resolution.
    Searching over the input size as well leaves the problem badly under-determined.
    Padding is capped at half the kernel, which is the only convention a sane configuration uses.
    Every candidate is checked against target_flat, so a returned solution is verified.
    """
    if max_padding is None:
        max_padding = kernel // 2
    solutions = []
    for stride in range(1, 9):
        for padding in range(0, max_padding + 1):
            size = input_size
            valid = True
            for index in range(n_convs):
                size = conv_out_size(size, kernel, stride, padding)
                if size < 1:
                    valid = False
                    break
                # Every convolution except the last must leave room for the next kernel
                if index < n_convs - 1 and size < kernel:
                    valid = False
                    break
            if not valid:
                continue
            if (size ** n_dims) * out_channels == target_flat:
                solutions.append(
                    {"input": input_size, "stride": stride, "padding": padding, "out": size}
                )
    return solutions


def search_input_sizes(kernel, n_convs, n_dims, target_flat, out_channels, max_input=32):
    """List the input sizes that admit any geometry reproducing the flattened size.

    This is only used to explain a failure, never to pick a geometry.
    """
    found = []
    for input_size in range(kernel, max_input + 1):
        if solve_conv_geometry(kernel, n_convs, n_dims, target_flat, out_channels, input_size):
            found.append(input_size)
    return found


def describe_encoder_group(sub_state, index):
    """Classify 1 observation encoder from its tensor shapes alone."""
    prefix = f"encoders.{index}."
    keys = [k for k in sub_state if k.startswith(prefix)]
    if not keys:
        return None

    conv0 = sub_state.get(prefix + "conv_layers.0.weight")
    if conv0 is not None:
        # A 5-dimensional convolution weight is Conv3d and a 4-dimensional one is Conv2d
        n_dims = conv0.dim() - 2
        kernel = int(conv0.shape[2])
        init_channel = int(conv0.shape[0])
        in_channels = int(conv0.shape[1])
        # The key is encoders.<i>.conv_layers.<j>.weight, so index j sits after the prefix
        conv_indices = sorted(
            int(k[len(prefix):].split(".")[1]) for k in keys
            if k.startswith(prefix + "conv_layers.") and k.endswith(".weight")
        )
        n_convs = len(conv_indices)
        last_conv = sub_state[prefix + f"conv_layers.{conv_indices[-1]}.weight"]
        out_channels = int(last_conv.shape[0])
        dense = sub_state.get(prefix + "dense_layers.0.weight")
        if dense is None:
            return {"kind": "conv_no_dense", "index": index}
        output_size = int(dense.shape[0])
        final_flat = int(dense.shape[1])
        return {
            "kind": "conv3d" if n_dims == 3 else "conv2d",
            "index": index,
            "n_dims": n_dims,
            "kernel": kernel,
            "init_channel": init_channel,
            "in_channels": in_channels,
            "n_convs": n_convs,
            "out_channels": out_channels,
            "output_size": output_size,
            "final_flat": final_flat,
        }

    mean = sub_state.get(prefix + "normalizer.running_mean")
    if mean is not None:
        return {
            "kind": "vector",
            "index": index,
            "size": int(mean.numel()),
            "steps": int(sub_state[prefix + "normalizer.normalization_steps"].item()),
            "mean": mean,
            "variance": sub_state[prefix + "normalizer.running_variance"],
        }

    return {"kind": "unknown", "index": index}


def build_encoder_stack(obs_state, trunk_state, activation_type, grid_size, depth_size):
    """Rebuild the observation encoders and the trunk, loading the trained weights.

    Returns a dictionary holding the built modules and the geometry that was solved.
    Raises RuntimeError with a plain reason when something cannot be rebuilt.
    """
    indices = sorted({
        int(k.split(".")[1]) for k in obs_state
        if k.startswith("encoders.") and k.split(".")[1].isdigit()
    })
    if not indices:
        raise RuntimeError("no encoders.<n> keys found under the observation encoder prefix")

    groups = [describe_encoder_group(obs_state, i) for i in indices]
    modules = []
    geometry = []

    for group in groups:
        if group is None or group["kind"] == "unknown":
            raise RuntimeError(f"encoder {group['index'] if group else '?'} could not be classified")

        if group["kind"] == "vector":
            modules.append(("vector", group))
            geometry.append(group)
            continue

        # The sensor resolution is a documented fact, so it anchors the search
        anchor = grid_size if group["n_dims"] == 3 else depth_size
        solutions = solve_conv_geometry(
            kernel=group["kernel"],
            n_convs=group["n_convs"],
            n_dims=group["n_dims"],
            target_flat=group["final_flat"],
            out_channels=group["out_channels"],
            input_size=anchor,
        )
        if not solutions:
            candidates = search_input_sizes(
                group["kernel"], group["n_convs"], group["n_dims"],
                group["final_flat"], group["out_channels"],
            )
            raise RuntimeError(
                f"encoder {group['index']} has no geometry at input size {anchor} reproducing "
                f"final_flat {group['final_flat']}; input sizes that do work are {candidates}"
            )
        group["solutions"] = solutions
        chosen = solutions[0]
        group["chosen"] = chosen

        if group["kind"] == "conv3d":
            side = chosen["input"]
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
            side = chosen["input"]
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

        if module.final_flat != group["final_flat"]:
            raise RuntimeError(
                f"encoder {group['index']} rebuilt with final_flat {module.final_flat} "
                f"but the checkpoint holds {group['final_flat']}"
            )

        own_state = {
            k[len(f"encoders.{group['index']}."):]: v
            for k, v in obs_state.items()
            if k.startswith(f"encoders.{group['index']}.")
        }
        missing, unexpected = module.load_state_dict(own_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"encoder {group['index']} weight load left "
                f"{len(missing)} missing and {len(unexpected)} unexpected keys"
            )
        module.eval()
        modules.append(("conv", group, module))
        geometry.append(group)

    first = trunk_state.get("sequence.0.weight")
    if first is None:
        raise RuntimeError("no sequence.0.weight found under the trunk prefix")
    trunk_input = int(first.shape[1])
    hidden_size = int(first.shape[0])
    layer_indices = sorted({
        int(k.split(".")[1]) for k in trunk_state
        if k.startswith("sequence.") and k.endswith(".weight")
    })
    num_layers = len(layer_indices)

    trunk = LinearEncoder(
        input_size=trunk_input,
        num_layers=num_layers,
        hidden_size=hidden_size,
        activation_type=activation_type,
    )
    missing, unexpected = trunk.load_state_dict(trunk_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"trunk weight load left {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
    trunk.eval()

    return {
        "modules": modules,
        "geometry": geometry,
        "trunk": trunk,
        "trunk_input": trunk_input,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
    }


def sample_observations(stack, batch, occupancy, depth_max, generator):
    """Draw 1 batch of synthetic observations matching each encoder's input shape.

    The occupancy grid is Bernoulli, because a boxcast occupancy sensor emits 0 or 1.
    The depth map is uniform on the stated range, because a depth sensor emits a bounded distance.
    The vector branch is fed its normalised form directly, drawn standard normal and clamped.
    That last choice is exact rather than approximate.
    A trained normaliser maps in-distribution input to roughly standard normal by construction.
    """
    tensors = []
    for entry in stack["modules"]:
        if entry[0] == "vector":
            group = entry[1]
            raw = torch.randn(batch, group["size"], generator=generator)
            tensors.append(("vector", torch.clamp(raw, -5.0, 5.0)))
            continue

        group, module = entry[1], entry[2]
        side = group["chosen"]["input"]
        channels = group["in_channels"]
        if group["kind"] == "conv3d":
            noise = torch.rand(batch, side, side, side, channels, generator=generator)
            tensors.append(("conv", (noise < occupancy).float(), module))
        else:
            noise = torch.rand(batch, side, side, channels, generator=generator)
            tensors.append(("conv", noise * depth_max, module))
    return tensors


def encode_batch(stack, tensors):
    """Run 1 batch of observations through the encoders and the trunk."""
    parts = []
    for entry in tensors:
        if entry[0] == "vector":
            parts.append(entry[1])
        else:
            parts.append(entry[2](entry[1]))
    concatenated = torch.cat(parts, dim=-1)
    if concatenated.shape[-1] != stack["trunk_input"]:
        raise RuntimeError(
            f"concatenated encoder output is {concatenated.shape[-1]} wide "
            f"but the trunk expects {stack['trunk_input']}"
        )
    return concatenated, stack["trunk"](concatenated)


# ===== PARTS B AND C: ATTENTION WEIGHTS =====

def infer_transformer_geometry(sub_state):
    """Read the transformer geometry straight out of its tensor shapes."""
    pos = sub_state.get("positional_embedding.weight")
    if pos is None:
        raise SystemExit("no positional_embedding.weight found under the transformer prefix")
    max_seq_len, d_model = int(pos.shape[0]), int(pos.shape[1])
    proj = sub_state.get("input_projection.weight")
    if proj is None:
        raise SystemExit("no input_projection.weight found under the transformer prefix")
    input_size = int(proj.shape[1])
    n_layers = len({
        int(k.split(".")[1]) for k in sub_state
        if k.startswith("layers.") and k.split(".")[1].isdigit()
    })
    return {
        "input_size": input_size,
        "d_model": d_model,
        "n_layers": n_layers,
        "max_seq_len": max_seq_len,
    }


def build_transformer(geometry, state=None):
    """Build a GTrXL at the given geometry and optionally load trained weights."""
    module = GTrXL(
        input_size=geometry["input_size"],
        d_model=geometry["d_model"],
        n_heads=N_HEADS,
        n_layers=geometry["n_layers"],
        d_ff_multiplier=D_FF_MULTIPLIER,
        max_seq_len=geometry["max_seq_len"],
        use_gru_gating=USE_GRU_GATING,
        dropout=0.0,
    )
    report = None
    if state is not None:
        missing, unexpected = module.load_state_dict(state, strict=False)
        report = (len(missing), len(unexpected), list(missing)[:5], list(unexpected)[:5])
    module.eval()
    return module, report


class AttentionRecorder:
    """Capture the attention weights and the attention inputs of every layer."""

    def __init__(self, module):
        self.module = module
        self.handles = []
        self.weights = {}
        self.inputs = {}

    def _make_hook(self, index):
        def hook(_module, inputs, output):
            # inputs[0] is the query, shaped (seq, batch, d_model) in this codebase
            self.inputs[index] = inputs[0].detach()
            # nn.MultiheadAttention returns a pair and GTrXLLayer discards the second element
            # need_weights is not passed by GTrXLLayer, so it defaults to True and the pair is populated
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                self.weights[index] = output[1].detach()
            else:
                self.weights[index] = None
        return hook

    def __enter__(self):
        for index, layer in enumerate(self.module.layers):
            self.handles.append(layer.attention.register_forward_hook(self._make_hook(index)))
        return self

    def __exit__(self, *_args):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def per_head_attention(attention_module, query, mask, n_heads):
    """Recompute the attention distribution per head from in_proj_weight.

    This reproduces what nn.MultiheadAttention does internally.
    It exists so the hooked head-average can be checked against an independent calculation.
    Input query is shaped (seq, batch, d_model) to match the module's own convention.
    Output is shaped (batch, n_heads, seq, seq).
    """
    seq_len, batch, d_model = query.shape
    d_head = d_model // n_heads

    packed = F.linear(query, attention_module.in_proj_weight, attention_module.in_proj_bias)
    q, k, _v = packed.chunk(3, dim=-1)

    q = q.contiguous().view(seq_len, batch * n_heads, d_head).transpose(0, 1)
    k = k.contiguous().view(seq_len, batch * n_heads, d_head).transpose(0, 1)

    q = q / math.sqrt(d_head)
    logits = torch.bmm(q, k.transpose(1, 2))

    if mask is not None:
        logits = logits.masked_fill(mask.unsqueeze(0), float("-inf"))

    weights = F.softmax(logits, dim=-1)
    return weights.view(batch, n_heads, seq_len, seq_len)


def row_entropy_nats(rows, allowed):
    """Return the entropy in nats of every attention row in a batch.

    Input rows is shaped (batch, seq_len) and holds 1 query position per sample.
    Output is shaped (batch,).

    This is deliberately computed per sample and never on a batch average.
    Averaging the rows first destroys the measurement being taken here.
    Samples each peaked on a different position average to a flat row.
    That flat row would report a wide context for a network that has in fact collapsed.
    """
    probabilities = rows[:, :allowed].clamp_min(1e-12)
    return -(probabilities * probabilities.log()).sum(dim=-1)


def effective_context_batch(rows, allowed):
    """Return the perplexity of every attention row in a batch, as a count of positions.

    A value near 1.0 means the row attends to 1 position in practice.
    A value near the allowed count means the row spreads evenly across the window.
    """
    return torch.exp(row_entropy_nats(rows, allowed))


def normalised_entropy_batch(rows, allowed):
    """Return the entropy of every row divided by the entropy of a uniform row.

    A value of 1.000 means the row is uniform over every position it is allowed to see.
    A value of 0.000 means the row puts all of its mass on 1 position.
    """
    if allowed <= 1:
        return torch.full((rows.shape[0],), float("nan"))
    return row_entropy_nats(rows, allowed) / math.log(allowed)


def mean_and_spread(values):
    """Return the mean and the standard deviation of a 1 dimensional tensor."""
    if values.numel() < 2:
        return float(values.mean()), 0.0
    return float(values.mean()), float(values.std(unbiased=False))


# ===== REPORTING =====

def report_attention(label, module, sequence, n_heads, seq_len, verbose_matrix=True):
    """Run 1 forward pass, capture the attention, and print what it shows."""
    with AttentionRecorder(module) as recorder:
        with torch.no_grad():
            module(sequence)

    mask = module._generate_causal_mask(seq_len, sequence.device) if seq_len > 1 else None

    summary = {}
    print(f"  --- {label} ---")
    for index in sorted(recorder.weights):
        hooked = recorder.weights[index]
        query = recorder.inputs[index]

        if hooked is None:
            print(f"  layer {index}  attention weights were not returned by the module")
            print("    GTrXLLayer passed need_weights=False, so this run cannot read them")
            continue

        with torch.no_grad():
            heads = per_head_attention(module.layers[index].attention, query, mask, n_heads)
            rebuilt_average = heads.mean(dim=1)
            agreement = float((rebuilt_average - hooked).abs().max())

        last = seq_len - 1
        rows = hooked[:, last, :]
        entropy, entropy_spread = mean_and_spread(normalised_entropy_batch(rows, seq_len))
        effective, effective_spread = mean_and_spread(effective_context_batch(rows, seq_len))
        mean_row = rows.mean(dim=0)
        self_mass = float(rows[:, last].mean())
        history_mass = float(rows[:, :last].sum(dim=-1).mean())

        print(f"  layer {index}")
        print(f"    hook against recomputation   max absolute difference {agreement:.3e}")
        print(f"    mean attention row at q{last}   " + " ".join(f"{v:.3f}" for v in mean_row))
        print(f"    normalised entropy           {entropy:.3f}  "
              f"standard deviation {entropy_spread:.3f}")
        print(f"    effective context            {effective:.2f} of {seq_len} positions,  "
              f"standard deviation {effective_spread:.2f}")
        print(f"    mass on the current frame    {self_mass:.3f}")
        print(f"    mass on the earlier frames   {history_mass:.3f}")

        per_head_effective = []
        for head in range(n_heads):
            head_rows = heads[:, head, last, :]
            per_head_effective.append(float(effective_context_batch(head_rows, seq_len).mean()))
        print("    effective context per head   " + " ".join(f"{v:.2f}" for v in per_head_effective))

        summary[index] = {
            "entropy": entropy,
            "effective": effective,
            "effective_spread": effective_spread,
            "self_mass": self_mass,
            "history_mass": history_mass,
            "per_head": per_head_effective,
            "agreement": agreement,
        }

        if verbose_matrix and index == max(recorder.weights):
            print("    full attention matrix, 1 row per query position")
            average = hooked.mean(dim=0)
            for row_index in range(seq_len):
                allowed = row_index + 1
                per_sample = effective_context_batch(hooked[:, row_index, :], allowed)
                values = " ".join(f"{v:.3f}" for v in average[row_index])
                print(f"      q{row_index}  {values}   effective {float(per_sample.mean()):.2f}")
    print()
    return summary


def context_ablation(module, sequence, seq_len, generator):
    """Measure how much the output at the last position depends on the earlier positions.

    Three comparisons are run and each isolates a different thing.
    The first swaps the history and holds the current frame, so it measures history dependence.
    The second swaps the current frame and holds the history, so it gives the denominator.
    Both keep the window length, the positional embedding and the causal mask identical.
    The third drops the window to 1 position, which is what the exported policy does.
    The ratio of the first to the second is scale free and does not depend on input magnitude.
    """
    history = seq_len - 1
    permutation = torch.randperm(sequence.shape[0], generator=generator)

    swapped_history = sequence.clone()
    swapped_history[:, :history, :] = sequence[permutation][:, :history, :]

    swapped_current = sequence.clone()
    swapped_current[:, history, :] = sequence[permutation][:, history, :]

    with torch.no_grad():
        full = module(sequence)[:, -1, :]
        no_history = module(swapped_history)[:, -1, :]
        no_current = module(swapped_current)[:, -1, :]
        single = module(sequence[:, -1:, :])[:, -1, :]

    baseline = float(full.norm(dim=-1).mean())
    history_effect = float((full - no_history).norm(dim=-1).mean())
    current_effect = float((full - no_current).norm(dim=-1).mean())
    single_effect = float((full - single).norm(dim=-1).mean())

    print("  --- context ablation at the last position ---")
    print(f"    output vector norm                          {baseline:.4f}")
    print(f"    change when the {history} earlier frames are swapped  "
          f"{history_effect:.4f}   relative {history_effect / max(baseline, 1e-12):.4f}")
    print(f"    change when the current frame is swapped     "
          f"{current_effect:.4f}   relative {current_effect / max(baseline, 1e-12):.4f}")
    print(f"    change when the window drops to 1 position   "
          f"{single_effect:.4f}   relative {single_effect / max(baseline, 1e-12):.4f}")
    ratio = history_effect / current_effect if current_effect > 0 else float("nan")
    print(f"    history against current frame                {ratio:.4f}")
    print()
    return {"ratio": ratio, "history": history_effect, "current": current_effect,
            "single": single_effect, "baseline": baseline}


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

    print("=" * 72)
    print("PROBE 02  ATTENTION MAP")
    print("=" * 72)
    print(f"file     {args.checkpoint}")
    print(f"torch    {torch.__version__}")
    print(f"seed     {args.seed}")
    print()

    state = load_actor_state(args.checkpoint)
    transformer_state = extract_submodule_state(state, TRANSFORMER_PREFIX)
    if not transformer_state:
        raise SystemExit(f"no keys found under the prefix {TRANSFORMER_PREFIX}")

    geometry = infer_transformer_geometry(transformer_state)
    print("TRANSFORMER GEOMETRY")
    for key, value in geometry.items():
        print(f"  {key:<14} {value}")
    print()

    trained, report = build_transformer(geometry, transformer_state)
    print("WEIGHT LOADING")
    print(f"  tensors offered   {len(transformer_state)}")
    print(f"  missing keys      {report[0]}  {report[2]}")
    print(f"  unexpected keys   {report[1]}  {report[3]}")
    if report[0] or report[1]:
        raise SystemExit("weight loading was not clean, so nothing below would be trustworthy")
    print()

    generator = torch.Generator().manual_seed(args.seed)
    control, _ = build_transformer(geometry)

    # ===== PART A =====
    print("=" * 72)
    print("PART A  MEASURED ACTIVATION ENTERING THE TRANSFORMER")
    print("=" * 72)

    obs_state = extract_submodule_state(state, OBS_ENCODER_PREFIX)
    trunk_state = extract_submodule_state(state, TRUNK_PREFIX)
    measured_rms = None
    measured_sequence = None

    if not obs_state or not trunk_state:
        print("  the encoder or trunk keys are absent, so the real scale cannot be measured")
        print()
    else:
        results = {}
        for name, activation in TRUNK_ACTIVATIONS:
            try:
                stack = build_encoder_stack(
                    obs_state, trunk_state, activation, args.grid_size, args.depth_size
                )
            except RuntimeError as error:
                print(f"  rebuild failed for activation {name}: {error}")
                continue

            if name == TRUNK_ACTIVATIONS[0][0]:
                print("  SOLVED ENCODER GEOMETRY")
                for group in stack["geometry"]:
                    if group["kind"] == "vector":
                        print(f"    encoder {group['index']}  vector, {group['size']} features, "
                              f"{group['steps']:,} normaliser steps")
                    else:
                        chosen = group["chosen"]
                        count = len(group["solutions"])
                        print(f"    encoder {group['index']}  {group['kind']}, kernel {group['kernel']}, "
                              f"stride {chosen['stride']}, padding {chosen['padding']}, "
                              f"input {chosen['input']}, output {group['output_size']}")
                        print(f"      final_flat {group['final_flat']} reproduced exactly, "
                              f"{count} geometry solution(s) consistent with it")
                        if count > 1:
                            for alternative in group["solutions"]:
                                print(f"      alternative  input {alternative['input']}, "
                                      f"stride {alternative['stride']}, padding {alternative['padding']}")
                print(f"    trunk      {stack['trunk_input']} to {stack['hidden_size']}, "
                      f"{stack['num_layers']} layers")
                print()

            local = torch.Generator().manual_seed(args.seed)
            tensors = sample_observations(stack, args.batch, args.occupancy, args.depth_max, local)
            with torch.no_grad():
                concatenated, activation_out = encode_batch(stack, tensors)

            rms = float(activation_out.pow(2).mean().sqrt())
            norm = float(activation_out.norm(dim=-1).mean())
            results[name] = {"rms": rms, "norm": norm, "stack": stack,
                             "concat": concatenated, "activation": activation_out}

        if results:
            print("  MEASURED TRUNK OUTPUT, WHICH IS THE TRANSFORMER INPUT")
            print("  activation      per element RMS     mean vector norm")
            for name in results:
                print(f"  {name:<14}  {results[name]['rms']:>14.4f}  {results[name]['norm']:>18.4f}")
            print()

            chosen_name = TRUNK_ACTIVATIONS[0][0] if TRUNK_ACTIVATIONS[0][0] in results else next(iter(results))
            measured_rms = results[chosen_name]["rms"]

            projection_weight = transformer_state["input_projection.weight"]
            positional = transformer_state["positional_embedding.weight"][: args.seq_len]
            projected_scale = measured_rms * float(projection_weight.norm()) / math.sqrt(
                projection_weight.shape[1]
            )
            positional_scale = float(positional.norm(dim=-1).mean())

            print("  BALANCE INSIDE THE TRANSFORMER INPUT")
            print(f"    projected content vector norm, estimated   {projected_scale:.4f}")
            print(f"    positional embedding vector norm, measured {positional_scale:.4f}")
            if positional_scale > 0:
                ratio = projected_scale / positional_scale
                print(f"    content against position ratio             {ratio:.2f}")
            print()

            # Build a 10-position window from independently sampled observations
            # Independent frames are the easiest case for attention to distinguish
            # A collapsed row under independent frames is therefore a strong result
            stack = results[chosen_name]["stack"]
            frames = []
            for _ in range(args.seq_len):
                local = torch.Generator().manual_seed(int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator)))
                tensors = sample_observations(stack, args.batch, args.occupancy, args.depth_max, local)
                with torch.no_grad():
                    _concat, frame = encode_batch(stack, tensors)
                frames.append(frame)
            measured_sequence = torch.stack(frames, dim=1)
            print(f"  built a real activation window of shape {tuple(measured_sequence.shape)}")
            print()

            # The occupancy fraction and the depth units are the 2 remaining assumptions
            # Neither is recorded in the checkpoint, so both are swept rather than asserted
            # A result that holds across the sweep does not depend on either assumption
            print("  SENSITIVITY TO THE 2 OBSERVATION ASSUMPTIONS")
            print("  occupancy  depth max   activation RMS   effective context of "
                  f"{args.seq_len}")
            for occupancy in (0.05, 0.15, 0.30):
                for depth_max in (0.2, 1.0, 5.0, 20.0):
                    frames = []
                    probe_generator = torch.Generator().manual_seed(args.seed + 1)
                    for _ in range(args.seq_len):
                        local = torch.Generator().manual_seed(
                            int(torch.randint(0, 2 ** 31 - 1, (1,), generator=probe_generator))
                        )
                        tensors = sample_observations(stack, args.batch, occupancy, depth_max, local)
                        with torch.no_grad():
                            _concat, frame = encode_batch(stack, tensors)
                        frames.append(frame)
                    window = torch.stack(frames, dim=1)
                    rms = float(window.pow(2).mean().sqrt())
                    with AttentionRecorder(trained) as recorder:
                        with torch.no_grad():
                            trained(window)
                    final_index = max(recorder.weights)
                    rows = recorder.weights[final_index][:, args.seq_len - 1, :]
                    effective = float(effective_context_batch(rows, args.seq_len).mean())
                    print(f"  {occupancy:<9.2f}  {depth_max:<9.1f}   {rms:>14.4f}   "
                          f"{effective:>10.2f}")
            print()

    # ===== PARTS B AND C =====
    print("=" * 72)
    print("PARTS B AND C  ATTENTION READ DIRECTLY FROM THE TRAINED NETWORK")
    print("=" * 72)
    print()

    measured_summary = None
    control_summary = None
    ablation = None

    if measured_sequence is not None:
        print("ON THE MEASURED REAL ACTIVATION")
        measured_summary = report_attention(
            "trained actor", trained, measured_sequence, N_HEADS, args.seq_len
        )
        control_summary = report_attention(
            "random control", control, measured_sequence, N_HEADS, args.seq_len,
            verbose_matrix=False,
        )
        ablation = context_ablation(trained, measured_sequence, args.seq_len, generator)
    else:
        print("ON THE MEASURED REAL ACTIVATION")
        print("  skipped, because Part A could not measure it")
        print()

    print("ON A SYNTHETIC SWEEP, SO THE MEASURED POINT CAN BE PLACED ON THE CURVE")
    sweep_scales = [0.01, 0.1, 1.0, 10.0, 100.0]
    if measured_rms is not None:
        sweep_scales.append(round(measured_rms, 4))
        sweep_scales = sorted(set(sweep_scales))
    sweep_rows = []
    for scale in sweep_scales:
        sequence = torch.randn(
            args.batch, args.seq_len, geometry["input_size"], generator=generator
        ) * scale
        summary = report_attention(
            f"trained actor at input RMS {scale:g}", trained, sequence,
            N_HEADS, args.seq_len, verbose_matrix=False,
        )
        final = max(summary) if summary else None
        sweep_rows.append((scale, summary[final]["effective"] if final is not None else float("nan")))

    print("  SWEEP SUMMARY, FINAL LAYER, LAST QUERY POSITION")
    print("  input RMS        effective context of 10")
    for scale, effective in sweep_rows:
        marker = ""
        if measured_rms is not None and abs(scale - round(measured_rms, 4)) < 1e-9:
            marker = "   <-- the measured real activation"
        print(f"  {scale:<14g}  {effective:>10.2f}{marker}")
    print()

    # ===== VERDICT =====
    print("=" * 72)
    print("MEASURED ON THE REAL ACTIVATION")
    print("=" * 72)

    if measured_summary is None:
        print("=" * 72)
        return

    final_layer = max(measured_summary)
    measured = measured_summary[final_layer]
    print(f"    effective context      {measured['effective']:.2f} of {args.seq_len} positions")
    print(f"    mass on earlier frames {measured['history_mass']:.3f}")
    if ablation is not None:
        print(f"    history against current frame at the output {ablation['ratio']:.4f}")
    if control_summary is not None:
        control_final = control_summary[max(control_summary)]
        print(f"    untrained control for comparison {control_final['effective']:.2f}")
    print()
    print()
    print("=" * 72)


if __name__ == "__main__":
    main()
