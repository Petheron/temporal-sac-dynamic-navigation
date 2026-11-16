"""Probe a trained windowed causal transformer for mask integrity and context use.

Usage:
    python Algorithms\\probe_01_attention.py "<path to actor_dict_0>"
"""

import argparse
import sys
from pathlib import Path

import torch


# The attribute path to the transformer inside the actor state dictionary
# Measured from a real checkpoint on 6 November 2026
MODULE_PREFIX = "body.transformer"

# Geometry measured from the same checkpoint, used as a cross-check
EXPECTED = {
    "input_size": 512,
    "d_model": 256,
    "n_layers": 2,
    "max_seq_len": 64,
}

# Heads cannot be read from the weights, because a packed projection of 3 times d_model by d_model is the same shape at any head count
# 4 is the value every production config sets
N_HEADS = 4

# The feedforward multiplier is readable and is checked below
D_FF_MULTIPLIER = 4


def load_actor_state(path):
    """Load a checkpoint and return the state dictionary inside it."""
    loaded = torch.load(str(path), map_location="cpu")
    if not isinstance(loaded, dict):
        raise SystemExit(f"Unexpected checkpoint type: {type(loaded)}")
    for wrapper in ("state_dict", "model_state_dict", "actor"):
        inner = loaded.get(wrapper)
        if isinstance(inner, dict):
            return inner
    return loaded


def extract_submodule_state(state, prefix):
    """Return the entries under a prefix, with the prefix stripped."""
    head = prefix + "."
    out = {}
    for key, value in state.items():
        if key.startswith(head):
            out[key[len(head):]] = value
    if not out:
        raise SystemExit(
            f"No keys found under {prefix!r}.\n"
            f"Run probe_00_inspect.py and check the key names."
        )
    return out


def infer_geometry(sub_state):
    """Read the module geometry out of the weights themselves."""
    geometry = {}

    pos = sub_state.get("positional_embedding.weight")
    if pos is None:
        raise SystemExit("positional_embedding.weight is absent")
    geometry["max_seq_len"] = int(pos.shape[0])
    geometry["d_model"] = int(pos.shape[1])

    proj = sub_state.get("input_projection.weight")
    if proj is None:
        raise SystemExit("input_projection.weight is absent")
    geometry["input_size"] = int(proj.shape[1])

    # Layer indices give the block count without constructing anything
    indices = set()
    for key in sub_state:
        parts = key.split(".")
        if parts[0] == "layers" and len(parts) > 1 and parts[1].isdigit():
            indices.add(int(parts[1]))
    geometry["n_layers"] = len(indices)

    # The first feedforward layer pins the multiplier
    ff = sub_state.get("layers.0.feedforward.0.weight")
    if ff is not None:
        geometry["d_ff"] = int(ff.shape[0])
        geometry["d_ff_multiplier"] = int(ff.shape[0]) // geometry["d_model"]

    # Gate keys present means gating is in use
    geometry["use_gru_gating"] = any("gate_attn" in k for k in sub_state)

    return geometry


def build_module(geometry):
    """Construct the production GTrXL at the inferred geometry."""
    # Encoders.py sits beside this script in Algorithms
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    try:
        from Encoders import GTrXL
    except ImportError as error:
        raise SystemExit(
            f"Could not import GTrXL from Encoders.py in {here}.\n"
            f"Original error: {error}"
        )

    try:
        module = GTrXL(
            input_size=geometry["input_size"],
            d_model=geometry["d_model"],
            n_heads=N_HEADS,
            n_layers=geometry["n_layers"],
            d_ff_multiplier=geometry.get("d_ff_multiplier", D_FF_MULTIPLIER),
            max_seq_len=geometry["max_seq_len"],
            use_gru_gating=geometry["use_gru_gating"],
            dropout=0.0,
        )
    except TypeError as error:
        raise SystemExit(
            f"GTrXL would not construct with the inferred arguments.\n"
            f"Its signature may differ from what this script expects.\n"
            f"Original error: {error}"
        )

    return module, GTrXL


def forward_once(module, x):
    """Run the module in evaluation mode with no gradient."""
    with torch.no_grad():
        return module(x)


def section_1_mask_integrity(module, seq_len, d_in, scale, generator):
    """Perturb each position and check that no earlier position moves.

    A causal mask means position q attends only to positions at or below q.
    So perturbing position p must leave every output at q < p untouched.
    Attention through a boolean mask gives exactly 0 weight, not a small one,
    so the expected difference below p is an exact 0 rather than a tiny value.
    """
    print()
    print("=" * 74)
    print(f"SECTION 1: CAUSAL MASK INTEGRITY, L4   (input scale {scale})")
    print("=" * 74)

    x = torch.randn(1, seq_len, d_in, generator=generator) * scale
    base = forward_once(module, x)

    worst_leak = 0.0
    worst_at = None
    rows = []

    for p in range(seq_len):
        perturbed = x.clone()
        # A large perturbation makes any leak easy to see
        perturbed[0, p, :] = perturbed[0, p, :] + 10.0 * scale
        out = forward_once(module, perturbed)

        delta = (out - base)[0].abs().max(dim=1).values

        below = [float(delta[q]) for q in range(p)]
        at_or_above = [float(delta[q]) for q in range(p, seq_len)]

        leak = max(below) if below else 0.0
        if leak > worst_leak:
            worst_leak = leak
            worst_at = p

        rows.append((p, leak, max(at_or_above)))

    print()
    print("  perturbed   max change BELOW it   max change AT or ABOVE it")
    print("  " + "-" * 62)
    for p, leak, forward_change in rows:
        flag = "" if leak == 0.0 else "   <-- LEAK"
        print(f"  {p:>9}   {leak:>19.3e}   {forward_change:>24.3e}{flag}")

    print()
    if worst_leak == 0.0:
        print("  VERDICT: PASS")
        print("  No perturbation moved any earlier position, by an exact 0.")
    else:
        print("  VERDICT: FAIL")
        print(f"  Worst leak {worst_leak:.3e}, from perturbing position {worst_at}.")

    # Position 0 should move nothing above it only if nothing attends to it
    # In a causal model every later position attends to 0, so it should move
    forward_from_zero = rows[0][2]
    print()
    if forward_from_zero == 0.0:
        print("  SECOND VERDICT: FAIL")
        print("  Perturbing position 0 changed nothing anywhere.")
    else:
        print("  SECOND VERDICT: PASS")
        print(f"  Perturbing position 0 propagated forward, max {forward_from_zero:.3e}.")

    return worst_leak == 0.0 and forward_from_zero > 0.0


def section_2_sensitivity(module, reference, seq_len, d_in, scales, generator):
    """Measure how much each context position affects the position read.

    At inference the output is taken at the last filled position.
    So the question is how much each earlier position changes that output.
    A trained module is compared against a randomly initialised one,
    because a random module is position sensitive by construction and is
    therefore the right control for whether training preserved that.
    """
    print()
    print("=" * 74)
    print("SECTION 2: CONTEXT SENSITIVITY PROFILE, L5")
    print("=" * 74)
    print()
    print()

    results = {}

    for label, mod in (("trained", module), ("random", reference)):
        for scale in scales:
            x = torch.randn(1, seq_len, d_in, generator=generator) * scale
            base = forward_once(mod, x)
            read_at = seq_len - 1

            raw = []
            for p in range(seq_len):
                perturbed = x.clone()
                perturbed[0, p, :] = perturbed[0, p, :] + 1.0 * scale
                out = forward_once(mod, perturbed)
                change = float((out[0, read_at] - base[0, read_at]).abs().max())
                raw.append(change)

            anchor = raw[-1]
            norm = [v / anchor if anchor > 0 else 0.0 for v in raw]
            results[(label, scale)] = (raw, norm)

    header = "  " + "position".ljust(22) + "".join(f"{p:>7}" for p in range(seq_len))
    for label in ("trained", "random"):
        print(f"  --- {label} ---")
        print(header)
        for scale in scales:
            raw, norm = results[(label, scale)]
            line = f"  scale {scale:<16g}" + "".join(f"{v:>7.3f}" for v in norm)
            print(line)
        print(f"  {'raw at last position':<22}" + "".join(
            f"{results[(label, s)][0][-1]:>7.2e}" for s in scales[:1]))
        print()

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    parser.add_argument("--seed", type=int, default=20261106)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"Not a file: {args.checkpoint}")

    print(f"checkpoint  {args.checkpoint}")
    print(f"torch       {torch.__version__}")
    print(f"seq_len     {args.seq_len}")
    print(f"seed        {args.seed}")

    state = load_actor_state(args.checkpoint)
    sub_state = extract_submodule_state(state, MODULE_PREFIX)
    geometry = infer_geometry(sub_state)

    print()
    print("===== GEOMETRY READ FROM THE CHECKPOINT =====")
    for key in sorted(geometry):
        expected = EXPECTED.get(key)
        mark = ""
        if expected is not None:
            mark = "  OK" if geometry[key] == expected else f"  DIFFERS from {expected}"
        print(f"  {key:<20} {geometry[key]}{mark}")

    if args.seq_len > geometry["max_seq_len"]:
        raise SystemExit(
            f"seq_len {args.seq_len} exceeds max_seq_len {geometry['max_seq_len']}"
        )

    module, gtrxl_class = build_module(geometry)

    missing, unexpected = module.load_state_dict(sub_state, strict=False)
    print()
    print("===== WEIGHT LOADING =====")
    print(f"  loaded      {len(sub_state)} tensors")
    print(f"  missing     {len(missing)}  {missing if missing else ''}")
    print(f"  unexpected  {len(unexpected)}  {unexpected if unexpected else ''}")
    if missing or unexpected:
        print()
    module.eval()

    # The control is the same class at the same geometry, untrained
    reference = gtrxl_class(
        input_size=geometry["input_size"],
        d_model=geometry["d_model"],
        n_heads=N_HEADS,
        n_layers=geometry["n_layers"],
        d_ff_multiplier=geometry.get("d_ff_multiplier", D_FF_MULTIPLIER),
        max_seq_len=geometry["max_seq_len"],
        use_gru_gating=geometry["use_gru_gating"],
        dropout=0.0,
    )
    reference.eval()

    generator = torch.Generator().manual_seed(args.seed)
    d_in = geometry["input_size"]

    passed = True
    for scale in args.scales:
        ok = section_1_mask_integrity(module, args.seq_len, d_in, scale, generator)
        passed = passed and ok

    section_2_sensitivity(module, reference, args.seq_len, d_in, args.scales, generator)

    print()
    print("=" * 74)
    print(f"SECTION 1 OVERALL: {'PASS at every scale' if passed else 'FAIL, see above'}")
    print("=" * 74)


if __name__ == "__main__":
    main()
