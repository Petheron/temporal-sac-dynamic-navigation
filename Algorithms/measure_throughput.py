"""Turn the smoke logs into a throughput table, and project cluster run times.

Usage:
    python results_check\\measure_throughput.py smoke_logs
"""
import os
import re
import sys

STEP = re.compile(r"Decision Step (\d+) \(\d+\): Time Elapsed: ([0-9.]+) s")

# Only runs on the same Unity level can be compared, because the simulation cost differs per level
# Level 5 is the depthwise moving platform, which is what every ablation uses
LEVEL_5 = {
    "10_ref": "reference, as production",
    "11_seq05": "sequence_length 5",
    "04_seq20": "sequence_length 20",
    "12_seq15": "sequence_length 15",
    "03_mem128": "memory_size 128",
    "13_mem256": "memory_size 256",
    "14_burnin05": "burn_in_length 5",
    "02_burnin10": "burn_in_length 10",
}
ORDER = ["10_ref", "11_seq05", "12_seq15", "04_seq20",
         "13_mem256", "03_mem128", "14_burnin05", "02_burnin10"]

# Measured on the cluster for the production LSTM configuration
CLUSTER_HOURS = 41.5
CLUSTER_CAP = 48.0


def steady_rate(path):
    """Steps per second, ignoring the first interval's startup cost."""
    text = open(path, encoding="utf-8", errors="replace").read()
    points = [(int(s), float(t)) for s, t in STEP.findall(text)]
    if len(points) < 3:
        return None
    steps = points[-1][0] - points[1][0]
    seconds = points[-1][1] - points[1][1]
    return steps / seconds if seconds > 0 else None


def main(folder):
    folder = os.path.abspath(folder)
    rates = {}
    for name in os.listdir(folder):
        if not name.endswith(".log"):
            continue
        stem = name[:-4]
        if stem in LEVEL_5:
            rate = steady_rate(os.path.join(folder, name))
            if rate is not None:
                rates[stem] = rate

    if "10_ref" not in rates:
        sys.exit("10_ref.log is missing, and it is the denominator for every "
                 "ratio. Run round two before using this script.")

    ref = rates["10_ref"]
    print(f"smoke logs folder : {folder}")
    print(f"reference rate    : {ref:.1f} steps/s, local, level 5\n")
    print("Each ablation against the unmodified reference, all on level 5.")
    print("Projected hours scale the measured 41.5 h cluster run by the ratio.\n")
    print(f"{'config':<28}{'steps/s':>9}{'vs ref':>9}{'projected h':>13}{'verdict':>24}")
    print("-" * 83)

    risky = []
    for stem in ORDER:
        if stem not in rates:
            print(f"{LEVEL_5[stem]:<28}{'not run':>9}")
            continue
        rate = rates[stem]
        ratio = ref / rate
        hours = CLUSTER_HOURS * ratio
        if hours <= CLUSTER_CAP * 0.9:
            verdict = "fits"
        elif hours <= CLUSTER_CAP:
            verdict = "tight, under 10% spare"
            risky.append(stem)
        else:
            verdict = "EXCEEDS the 48 h cap"
            risky.append(stem)
        print(f"{LEVEL_5[stem]:<28}{rate:>9.1f}{ratio:>8.2f}x{hours:>13.1f}{verdict:>24}")

    print("\n" + "-" * 81)
    if risky:
        print(f"\nAt risk: {', '.join(LEVEL_5[s] for s in risky)}")
    else:
        print("\nNo configuration projects past the cap.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "smoke_logs")
