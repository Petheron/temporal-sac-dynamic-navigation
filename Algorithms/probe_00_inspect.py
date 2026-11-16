"""Inspect a trained actor checkpoint and report its structure.

Usage:
    python Algorithms\\probe_00_inspect.py "<path to actor_dict_0>"
"""

import argparse
from pathlib import Path

import torch


# Keys holding any of these substrings are reported in the focused section
# The list is wide so nothing relevant is filtered out
KEYS_OF_INTEREST = (
    "transformer",
    "positional",
    "pos_emb",
    "attn",
    "attention",
    "in_proj",
    "out_proj",
    "gate",
    "gru",
    "lstm",
    "input_projection",
    "output_norm",
    "norm",
)


def load_state_dict(path):
    """Load a checkpoint and return the state dictionary inside it."""
    # map_location keeps this working on a machine with no graphics card
    loaded = torch.load(str(path), map_location="cpu")

    # A checkpoint is sometimes a bare state dictionary
    # It is sometimes a wrapper holding one under a key
    if isinstance(loaded, dict):
        for wrapper_key in ("state_dict", "model_state_dict", "actor"):
            inner = loaded.get(wrapper_key)
            if isinstance(inner, dict):
                print(f"state dictionary found under the key {wrapper_key!r}")
                return inner
        return loaded

    raise SystemExit(f"Unexpected checkpoint type: {type(loaded)}")


def is_tensor(value):
    """Return True when a value is a tensor with a shape."""
    return hasattr(value, "shape")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"Not a file: {args.checkpoint}")

    print(f"file      {args.checkpoint}")
    print(f"size      {args.checkpoint.stat().st_size:,} bytes")
    print(f"torch     {torch.__version__}")
    print()

    state = load_state_dict(args.checkpoint)

    tensor_items = [(k, v) for k, v in state.items() if is_tensor(v)]
    other_items = [(k, v) for k, v in state.items() if not is_tensor(v)]

    total = sum(int(v.numel()) for _, v in tensor_items)
    print(f"entries   {len(state)} total, {len(tensor_items)} tensors")
    print(f"elements  {total:,}")
    print()

    # A genuine Transformer actor is 5,355,592 elements
    # A Long Short-Term Memory actor is 2,842,440
    # A frame-stacking actor at 4 stacks is 2,085,576
    known = {
        5355592: "Transformer actor",
        2842440: "Long Short-Term Memory actor",
        2085576: "frame-stacking actor, 4 stacks",
        2065864: "frame-stacking actor, 2 stacks",
        2125000: "frame-stacking actor, 8 stacks",
        2144712: "frame-stacking actor, 10 stacks",
        2203848: "frame-stacking actor, 16 stacks",
    }
    verdict = known.get(total)
    if verdict:
        print(f"IDENTIFIED  {verdict}")
    else:
        print("NOT IDENTIFIED against any known actor parameter count")
    print()

    print("===== EVERY KEY, IN ORDER =====")
    for key, value in tensor_items:
        shape = tuple(int(d) for d in value.shape)
        print(f"  {key:<70} {str(shape):<24} {int(value.numel()):>12,}")
    print()

    if other_items:
        print("===== NON-TENSOR ENTRIES =====")
        for key, value in other_items:
            print(f"  {key:<70} {type(value).__name__}")
        print()

    print("===== FOCUSED: TEMPORAL MODULE =====")
    matched = [
        (k, v) for k, v in tensor_items
        if any(token in k.lower() for token in KEYS_OF_INTEREST)
    ]
    if not matched:
        print("  no key matched any token of interest")
    for key, value in matched:
        shape = tuple(int(d) for d in value.shape)
        print(f"  {key:<70} {str(shape)}")
    print()

    print("===== INFERRED GEOMETRY =====")

    # The positional embedding table pins 2 numbers at once
    # Its first dimension is max_seq_len and its second is d_model
    pos_keys = [k for k, _ in tensor_items if "positional" in k.lower() or "pos_emb" in k.lower()]
    if pos_keys:
        for key in pos_keys:
            shape = tuple(int(d) for d in state[key].shape)
            print(f"  positional table   {key}  shape {shape}")
            if len(shape) == 2:
                print(f"    max_seq_len      {shape[0]}")
                print(f"    d_model          {shape[1]}")
                # max_seq_len is max(sequence_length * 2, 64) in this codebase
                # 64 means the configured sequence length was 32 or below
                # 256 would mean the constructor default was used instead
                if shape[0] == 64:
                    print("    reading          consistent with max(sequence_length * 2, 64)")
                elif shape[0] == 256:
                    print("    reading          the 256 constructor default was used")
                else:
                    print("    reading          neither 64 nor 256")
    else:
        print("  no positional embedding table found")

    # Layer indices appear in the key names of a module list
    # Collecting them gives the block count without constructing anything
    layer_indices = set()
    for key, _ in tensor_items:
        parts = key.split(".")
        for index, part in enumerate(parts):
            if part == "layers" and index + 1 < len(parts) and parts[index + 1].isdigit():
                layer_indices.add(int(parts[index + 1]))
    if layer_indices:
        print(f"  layer indices      {sorted(layer_indices)}")
        print(f"    block count      {len(layer_indices)}")
    else:
        print("  no layers.<n> keys found")

    # The attention packed projection is 3 times d_model by d_model
    # That cross-checks d_model independently of the positional table
    for key, value in tensor_items:
        if "in_proj_weight" in key:
            shape = tuple(int(d) for d in value.shape)
            print(f"  attention in_proj  {key}  shape {shape}")
            if len(shape) == 2 and shape[0] == 3 * shape[1]:
                print(f"    d_model          {shape[1]}  (from a 3x packed projection)")
            break

    # The prefix in front of the temporal module is what the probe must load
    if pos_keys:
        prefix = pos_keys[0].rsplit(".", 1)[0]
        print()
        print(f"  MODULE PREFIX      {prefix}")


if __name__ == "__main__":
    main()
