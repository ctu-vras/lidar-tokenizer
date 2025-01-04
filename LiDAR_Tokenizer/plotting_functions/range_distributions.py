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

for file in files[:5000]:
    arr = np.fromfile(os.path.join(dir,file), dtype=np.float32).reshape(-1,5)
    xyz = arr[:,:3]
    dist = np.linalg.norm(xyz,axis=1)
    all_dist += dist.tolist()

data = np.array(all_dist)
data = data[np.where(data>2)]
data = data[np.where(data<70)]
data = np.array(random.sample(list(data), 1000000))

plt.figure(figsize=(8, 6))
sns.histplot(data, bins=40, kde=True, color="skyblue", edgecolor="black", stat="probability")
plt.title("Distribution of point distances: nuScenes")
plt.xlabel("Distance (m)")
plt.ylabel("Probability")
#plt.show()
plt.subplots_adjust(left=0.15, bottom=0.15)
plt.savefig("LiDAR_Tokenizer/plotting_functions/nuscenes_distribution.png")

del all_dist
del data


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

for file in all_files[:5000]:
    rawdata = np.load(file)['point_cloud']

    channels = [2,3,4,5,6,7]

    all_points = []
    for i,channel in enumerate(channels):
        data_channel = rawdata[np.where(rawdata[:,4] == channel)]

        pcd = data_channel[:,:3]
        all_points.append(pcd)

    xyz = np.vstack(all_points)

    dist = np.linalg.norm(xyz,axis=1)
    all_dist += dist.tolist()

data2 = np.array(all_dist)
data2 = data2[np.where(data2>3.7)]
data2 = data2[np.where(data2<200)]
data2 = np.array(random.sample(list(data2), 1000000))

#bins = np.linspace(0, 200, 50)  # 50 bins with uniform width
plt.figure(figsize=(8, 6))
#sns.histplot(data, bins=bins, kde=True, color="orange", edgecolor="black", stat="probability", label="nuScenes", alpha=0.6)
sns.histplot(data2, bins=40, kde=True, color="skyblue", edgecolor="black", stat="probability")
plt.title("Distribution of point distances: FrontCam")
plt.xlabel("Distance (m)")
plt.ylabel("Probability")
#plt.show()
plt.subplots_adjust(left=0.15, bottom=0.15)
plt.savefig("LiDAR_Tokenizer/plotting_functions/frontcam_distribution.png")

