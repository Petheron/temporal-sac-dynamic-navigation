import numpy as np
import os

data1 = np.load("../Results/SAC_DRLNavigation_A/results.npy", allow_pickle=True)
data2 = np.load("../Results/SAC_DRLNavigation_B/results.npy", allow_pickle=True)

merged = []
for i in range(len(data1)):
  merged_item = []
  for j in range(len(data1[i])):
    combined = np.concatenate([data1[i][j], data2[i][j]], axis=0)
    merged_item.append(combined)
  merged.append(merged_item)

output_dir = "../Results/DepthwiseMovingPlatform/LSTM"
os.makedirs(output_dir, exist_ok=True)
np.save(f"{output_dir}/results.npy", np.array(merged, dtype=object))

print(f"Successfully Merged {data1[0][0].shape[0]} + {data2[0][0].shape[0]} Seeds into {output_dir}/results.npy")