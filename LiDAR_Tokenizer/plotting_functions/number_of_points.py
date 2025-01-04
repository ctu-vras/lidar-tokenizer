# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import seaborn as sns

plt.rcParams.update({
    'axes.titlesize': 20,  # Title font size
    'axes.labelsize': 20,  # Label font size
    'xtick.labelsize': 20,  # X-tick font size
    'ytick.labelsize': 20,  # Y-tick font size
    'legend.fontsize': 20  # Legend font size, if applicable
})


dir = '/mnt/data/Public_datasets/nuScenes/samples/LIDAR_TOP'

all_dist = []
files = os.listdir(dir)
random.shuffle(files)

all_lens = []
for file in files[:]:
    arr = np.fromfile(os.path.join(dir,file), dtype=np.float32).reshape(-1,5)
    all_lens.append(len(arr))


plt.figure(figsize=(8, 6))
sns.histplot(all_lens, bins=40, kde=True, color="skyblue", edgecolor="black", stat="probability")
plt.title("Number of points: nuScenes")
plt.xlabel("Number of points")
plt.ylabel("Probability")
plt.savefig("LiDAR_Tokenizer/plotting_functions/nuscenes_num_points.png")

print(f"nuscenes mean {np.mean(all_lens)}")
print(f"nuscenes std {np.std(all_lens)}")


dirs = ['/data/npz_2024-09-06-07-42-41/',
        '/data/npz_2024-09-06-10-59-33/',
        '/data/npz_2024-09-06-11-19-38/',
        '/data/npz_2024-09-06-11-42-34/',
        '/data/npz_2024-09-09-12-33-35/',
        '/data/npz_2024-09-13-06-43-51/',
        '/data/npz_2024-09-13-07-06-04/',
        '/data/npz_2024-09-13-10-31-13/',
        '/data/npz_2024-09-13-12-19-26/',
        '/data/npz_2024-09-17-06-52-07/'
        ]

all_dist = []
all_files = []

for dir in dirs:
    files = os.listdir(dir)
    files = [os.path.join(dir, f) for f in files]
    all_files += files

random.shuffle(all_files)

all_lens = []

for file in all_files[:]:
    rawdata = np.load(file)['point_cloud']

    channels = [2,3,4,5,6,7]

    all_points = []
    for i,channel in enumerate(channels):
        data_channel = rawdata[np.where(rawdata[:,4] == channel)]

        pcd = data_channel[:,:3]
        all_points.append(pcd)

    xyz = np.vstack(all_points)

    all_lens.append(len(xyz))


plt.figure(figsize=(8, 6))
sns.histplot(all_lens, bins=40, kde=True, color="skyblue", edgecolor="black", stat="probability")
plt.title("Number of points: FrontCam")
plt.xlabel("Number of points")
plt.ylabel("Probability")
plt.savefig("LiDAR_Tokenizer/plotting_functions/frontcam_num_points.png")

print(f"FrontCam mean {np.mean(all_lens)}")
print(f"FrontCam std {np.std(all_lens)}")

