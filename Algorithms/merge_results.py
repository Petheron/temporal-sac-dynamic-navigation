import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description="Merge results.npy files from individual seed runs.")
parser.add_argument("-i", "--inputs", nargs="+", required=True, help="Paths to seed result folders (each containing results.npy)")
parser.add_argument("-o", "--output", required=True, help="Output folder for merged results.npy")
parser.add_argument("--force", action="store_true", help="Allow an existing results.npy at the output path to be overwritten")
parser.add_argument("--expect-steps", type=int, default=1000000, help="Reject any input that did not reach this step count. Pass 0 to disable, which is needed for smoke tests")
args = parser.parse_args()

# Refuse to overwrite an existing merged file unless that is asked for explicitly
# Without this guard, pointing -o at a run directory destroys that run's results
output_path = os.path.join(args.output, "results.npy")
if os.path.exists(output_path) and not args.force:
  raise SystemExit(f"REFUSED: {output_path} already exists. Pass --force to overwrite it.")

datasets = []
for path in args.inputs:
  filepath = os.path.join(path, "results.npy") if not path.endswith(".npy") else path
  dataset = np.load(filepath, allow_pickle=True)

  # Group 0 element 0 is steps_so_far, so the last value of a row is where that run stopped
  # An unfinished run merged in alongside finished ones would silently drag the mean down
  # A length mismatch is already caught by concatenate, but 3 equally short runs are not
  if args.expect_steps:
    for seed_index, row in enumerate(np.asarray(dataset[0][0])):
      last_step = int(row[-1])
      if last_step != args.expect_steps:
        raise SystemExit(
          f"REFUSED: {filepath} seed {seed_index} stopped at {last_step} steps, "
          f"not {args.expect_steps}. Merging it would corrupt the mean."
        )

  datasets.append(dataset)
  print(f"Loaded {filepath} ({datasets[-1][0][0].shape[0]} seed(s))")

merged = []
for i in range(len(datasets[0])):
  merged_item = []
  for j in range(len(datasets[0][i])):
    combined = np.concatenate([dataset[i][j] for dataset in datasets], axis=0)
    merged_item.append(combined)
  merged.append(merged_item)

os.makedirs(args.output, exist_ok=True)
np.save(os.path.join(args.output, "results.npy"), np.array(merged, dtype=object))

total_seeds = sum(d[0][0].shape[0] for d in datasets)
print(f"Merged {len(datasets)} files ({total_seeds} seeds) into {args.output}/results.npy")