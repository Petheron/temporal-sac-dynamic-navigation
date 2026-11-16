"""Name the series that are identical across runs, and print headline metrics.

Usage: python3 -I headline.py LABEL=path.npy [...]
"""
import sys
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from compare_results import LAYOUT, TIMING, flatten  # noqa: E402

runs = {}
for arg in sys.argv[1:]:
    label, path = arg.split("=", 1)
    runs[label] = flatten(path)

keys = [k for k in runs[next(iter(runs))] if k not in TIMING]
labels = list(runs)

print("Series identical across ALL runs, and whether they carry any signal")
print("-" * 78)
for key in keys:
    arrays = [runs[l][key] for l in labels]
    if all(np.array_equal(arrays[0], a) for a in arrays[1:]):
        a = arrays[0].astype(float)
        uniq = np.unique(a[np.isfinite(a)])
        if key == ("step", "steps_so_far"):
            note = ("the checkpoint schedule, i.e. the x-axis -> identical "
                    "by construction, not a metric")
        elif uniq.size == 1:
            note = f"constant {uniq[0]:g}  -> degenerate, carries no signal"
        elif uniq.size <= 4:
            note = f"only {uniq.size} distinct values {uniq}  -> near-degenerate"
        else:
            note = (f"{uniq.size} distinct values, min {a.min():g} "
                    f"max {a.max():g}  -> CARRIES SIGNAL, investigate")
        print(f"  {key[0]}.{key[1]:<22} {note}")

print()
print("Headline metrics at the final checkpoint")
print("-" * 78)
hdr = f"  {'metric':<24}" + "".join(f"{l:>14}" for l in labels)
print(hdr)
for key in [("side_stats", "success_rate"), ("side_stats", "fail_rate"),
            ("side_stats", "timeout_fail_rate"), ("return", "avg_ep_return"),
            ("length", "avg_ep_length")]:
    if key not in runs[labels[0]]:
        continue
    row = f"  {key[1]:<24}"
    for l in labels:
        row += f"{float(runs[l][key].ravel()[-1]):>14.4f}"
    print(row)

print()
print("Mean over the final 10 checkpoints (less noisy)")
print("-" * 78)
print(hdr)
for key in [("side_stats", "success_rate"), ("return", "avg_ep_return"),
            ("length", "avg_ep_length")]:
    row = f"  {key[1]:<24}"
    for l in labels:
        row += f"{float(np.nanmean(runs[l][key].ravel()[-10:])):>14.4f}"
    print(row)
