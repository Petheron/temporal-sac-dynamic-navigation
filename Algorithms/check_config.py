"""Config check for the temporal_architecture dispatch bug.

Usage:
    python check_config.py <path-to-config.yaml>     -> full detail for one file
    python check_config.py <path-to-a-directory>     -> recurses, prints a summary table
    python check_config.py <directory> --full        -> recurses, full detail for every file
"""
import sys, os, re, yaml


class UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.YAMLError(
                    f"duplicate key {key!r} at line {key_node.start_mark.line + 1}")
            seen.add(key)
        return super().construct_mapping(node, deep)


def inspect(path):
    """Return a dict of findings for one config file."""
    out = {"path": path, "error": None, "ep_blocks": [], "dup": None,
           "ep_present": False, "ep_type": "-", "ta": "<ABSENT>", "ta_type": "-",
           "match": False, "nested": [], "extras": {}, "memory": {}}
    try:
        raw = open(path, encoding="utf-8-sig").read()
    except OSError as e:
        out["error"] = f"cannot read: {e}"
        return out

    out["ep_blocks"] = [i + 1 for i, l in enumerate(raw.splitlines())
                        if re.match(r"^environment_parameters\s*:", l)]
    try:
        yaml.load(raw, Loader=UniqueKeyLoader)
    except yaml.YAMLError as e:
        out["dup"] = str(e).strip().splitlines()[0]

    try:
        cfg = yaml.safe_load(raw) or {}
    except yaml.YAMLError as e:
        out["error"] = f"YAML parse error: {e}"
        return out
    if not isinstance(cfg, dict):
        out["error"] = f"top level is {type(cfg).__name__}, not a mapping"
        return out

    ep = cfg.get("environment_parameters")
    out["ep_present"] = ep is not None
    out["ep_type"] = type(ep).__name__
    ep = ep if isinstance(ep, dict) else {}

    ta = ep.get("temporal_architecture", "<ABSENT>")
    out["ta"], out["ta_type"] = ta, type(ta).__name__
    out["match"] = (ta == "transformer")

    for k in ("transformer_n_heads", "transformer_n_layers",
              "transformer_d_ff_multiplier", "transformer_use_gru_gating",
              "burn_in_length", "overlap_length", "num_test_eps", "seeds"):
        out["extras"][k] = ep.get(k, "<ABSENT>")

    for name, bs in (cfg.get("behaviors") or {}).items():
        bs = bs or {}
        out["memory"][name] = ((bs.get("network_settings") or {}).get("memory"))
        if "environment_parameters" in bs:
            out["nested"].append(name)
    return out


def detail(r):
    print(f"path     : {os.path.abspath(r['path'])}")
    if r["error"]:
        print(f"RESULT   : {r['error']}")
        return
    print(f"top-level 'environment_parameters:' lines : {r['ep_blocks'] or 'NONE'}")
    if len(r["ep_blocks"]) > 1:
        print("RESULT   : DUPLICATE BLOCKS -- PyYAML keeps the LAST and silently drops the first")
    print(f"duplicate-key scan (whole file)          : {r['dup'] or 'clean'}")
    print(f"environment_parameters present           : {r['ep_present']} (type {r['ep_type']})")
    print(f"temporal_architecture                    : {r['ta']!r} (type {r['ta_type']})")
    print(f"matches == 'transformer'                 : {r['match']}")
    for k, v in r["extras"].items():
        print(f"  {k:<28}: {v!r}")
    for name, mem in r["memory"].items():
        print(f"behaviour {name!r} network_settings.memory : {mem}")
    for name in r["nested"]:
        print(f"  WARNING: 'environment_parameters' nested under behaviour {name!r}; "
              f"RunPolicy reads only the top-level block")


def label_for(path, root):
    rel = os.path.relpath(path, root)
    first = rel.split(os.sep)[0]
    return first if first not in (".", "..") else os.path.basename(path)


def main(argv):
    if not argv:
        print(__doc__)
        return
    target, full = argv[0], ("--full" in argv)

    if os.path.isfile(target):
        detail(inspect(target))
        return
    if not os.path.isdir(target):
        print(f"RESULT   : not found -- {os.path.abspath(target)}")
        return

    paths = []
    for dirpath, _, names in os.walk(target):
        for n in sorted(names):
            if n.lower().endswith((".yaml", ".yml")):
                paths.append(os.path.join(dirpath, n))
    paths.sort()
    if not paths:
        print(f"RESULT   : no .yaml files under {os.path.abspath(target)}")
        return

    rows = []
    for p in paths:
        r = inspect(p)
        rows.append((label_for(p, target), r))
        if full:
            print("\n==== " + p + " ====")
            detail(r)

    w = max(len(l) for l, _ in rows)
    print(f"\n{'run':<{w}}  blocks  present  temporal_architecture            match  flags")
    print("-" * (w + 62))
    for l, r in rows:
        flags = []
        if r["error"]:
            flags.append(r["error"])
        if len(r["ep_blocks"]) > 1:
            flags.append(f"DUPLICATE BLOCKS at lines {r['ep_blocks']}")
        if r["dup"]:
            flags.append(r["dup"])
        if r["nested"]:
            flags.append(f"nested under behaviour {r['nested']}")
        print(f"{l:<{w}}  {len(r['ep_blocks']):^6}  {str(r['ep_present']):^7}  "
              f"{repr(r['ta']):<32} {str(r['match']):<5}  {'; '.join(flags)}")

    print(f"\n{len(paths)} config(s) scanned. Re-run with --full for per-file detail.")


if __name__ == "__main__":
    main(sys.argv[1:])