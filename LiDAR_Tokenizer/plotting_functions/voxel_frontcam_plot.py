# %%
import importlib
#importlib.reload(my_module)

from data_managment import datasets, nuscenes
from data_managment.data_modules import collate_ultralidar
from model_managment.helpers import load_model
from model_managment.models import UltraLidarModel

import matplotlib.image as mpimg
from PIL import Image

import matplotlib
import numpy as np
import plotly.graph_objects as go
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import yaml
import os

root_dir = os.path.dirname(__file__)

config_path = os.path.join(root_dir, "model_managment/config_FINAL_FRONTCAM_ul_log_final.yaml")
model = load_model(config_path)
checkpoint = torch.load(os.path.join(root_dir, "slurm_logs/ul_FINAL_FRONTCAM_log_241123_1905/models/ultralidar-best-epoch=21-step=89847.ckpt"), map_location=torch.device('cpu'))
model.load_state_dict(checkpoint["state_dict"])

# disable randomness, dropout, etc...
model.eval()

# %%
""" test_dataset_nuscenes = nuscenes.Nuscenes('/mnt/data/Public_datasets/nuScenes/', info_file_path='data_managment/nuscenes_info_single.json' ,version='v1.0-trainval', split='test')

test_dataset = datasets.Voxels(test_dataset_nuscenes, is_train=False, return_original=True)

data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_ultralidar)

trainer = pl.Trainer() """
from data_managment.data_modules import LidarTokenizerModule

#config_path = "model_managment/config_ultralidar_livox.yaml"

data = LidarTokenizerModule(dataroot = '/data',#'/mnt/data/Public_datasets/nuScenes/',
                      batch_size = 1,
                      num_workers = 2,
                      info_file_path = "data_managment/frontcam_single.json",    # specifies which files are train/val/test
                      mode="ultralidar",
                      dataset="frontcam",
                      config_path=config_path,
                      version="v1.0-trainval",
                      )    

trainer = pl.Trainer(gpus=0)
model.sigma = 0.3
predictions = trainer.predict(model, datamodule=data)



# %%
import plotly.io as pio
pio.renderers.default = 'browser'

i = 0

metrics = predictions[i][0]
pcd_in = predictions[i][1][0][:,:3]
pcd_out = predictions[i][2][0]
voxels_pcd_in = predictions[i][3][0]

print(f"IOU             : {round(float(metrics['test/lidar_rec_iou'].item()),3)}")
print(f"Code utilization: {round(float(metrics['test/code_util'].item()),3)}")
print(f"Code uniformity : {round(float(metrics['test/code_uniformity'].item()),3)}")

fig = go.Figure(data=[go.Scatter3d(
    x=pcd_in[:,0],
    y=pcd_in[:,1],
    z=pcd_in[:,2],
    mode='markers',
    marker=dict(size=2, color='blue')
)])

fig.add_trace(go.Scatter3d(
    x=pcd_out[:,0],
    y=pcd_out[:,1],
    z=pcd_out[:,2],
    mode='markers',
    marker=dict(size=2, color='darkmagenta')
))

fig.add_trace(go.Scatter3d(
    x=voxels_pcd_in[:,0],
    y=voxels_pcd_in[:,1],
    z=voxels_pcd_in[:,2],
    mode='markers',
    marker=dict(size=2, color='red')
))

cube_x = [0, 100, 100, 0, 0,        0, 100, 100, 0, 0, 100,100    ,100,100,0,0]
cube_y = [-50, -50, 50, 50, -50,    -50, -50, 50, 50, -50, -50,-50, 50,50,50,50]
cube_z = [-2, -2, -2, -2, -2,       18, 18, 18, 18, 18, 18,    -2, -2, 18, 18,-2]

# Adding the cube outline
fig.add_trace(go.Scatter3d(
    x=cube_x,
    y=cube_y,
    z=cube_z,
    mode='lines',
    line=dict(color='red', width=2),
    name='Occupancy Grid Outline'
))

fig.update_layout(
    scene=dict(
        xaxis=dict(nticks=10, range=[-100, 300]),
        yaxis=dict(nticks=10, range=[-200, 200]),
        zaxis=dict(nticks=10, range=[-100, 300]),
        aspectmode='cube'
    )
)

#fig.show()
fig.write_html("LiDAR_Tokenizer/plotting_functions/voxel_frontcam.html")

