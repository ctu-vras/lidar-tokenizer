import numpy as np
import matplotlib.pyplot as plt
import os
import random
import seaborn as sns

dir = '/mnt/data/Public_datasets/nuScenes/samples/LIDAR_TOP'

all_dist = []
files = os.listdir(dir)
random.shuffle(files)

for file in files[:1000]:
    arr = np.fromfile(os.path.join(dir,file), dtype=np.float32).reshape(-1,5)
    xyz = arr[:,:3]
    dist = np.linalg.norm(xyz,axis=1)
    all_dist += dist.tolist()

plt.figure(figsize=(10, 6))
sns.histplot(all_dist, bins=10, kde=True, color="skyblue", edgecolor="black")
plt.title("Histogram of Values")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.show()