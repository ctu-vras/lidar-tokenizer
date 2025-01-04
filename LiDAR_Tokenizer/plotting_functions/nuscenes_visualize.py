import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# Set default renderer
pio.renderers.default = 'browser'

# Load data
fn = "/mnt/data/Public_datasets/nuScenes/samples/LIDAR_TOP/n008-2018-08-30-10-33-52-0400__LIDAR_TOP__1535639756450885.pcd.bin"
p = np.fromfile(fn, dtype=np.float32)
p = p.reshape((-1, 5))

# Create the 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=p[:, 0],
    y=p[:, 1],
    z=p[:, 2],
    mode='markers',
    marker=dict(size=2, color='blue')
)])

# Update layout to enforce symmetric axes
fig.update_layout(
    scene=dict(
        xaxis=dict(nticks=10, range=[-80,80], title="X Axis", backgroundcolor="white"),
        yaxis=dict(nticks=10, range=[-80,80], title="Y Axis", backgroundcolor="white"),
        zaxis=dict(nticks=10, range=[-80,80], title="Z Axis", backgroundcolor="white"),
        aspectmode='cube'  # Ensures all axes are scaled equally
    ),
    paper_bgcolor="white",  # Background color outside the 3D scene
    plot_bgcolor="white"    # Background color inside the 3D scene
)

xmin = -50
xmax = 50
zmin = -5
zmax = 5
cube_x = [xmin, xmax, xmax, xmin, xmin,        xmin, xmax, xmax, xmin, xmin, xmax, xmax    ,xmax,xmax,xmin,xmin]
cube_y = [-50, -50, 50, 50, -50,    -50, -50, 50, 50, -50, -50,-50, 50,50,50,50]
cube_z = [zmin, zmin, zmin, zmin, zmin,       zmax, zmax, zmax, zmax, zmax, zmax,    zmin, zmin, zmax, zmax,zmin]

# Adding the cube outline
fig.add_trace(go.Scatter3d(
    x=cube_x,
    y=cube_y,
    z=cube_z,
    mode='lines',
    line=dict(color='red', width=2),
    name='Occupancy Grid Outline'
))

# Show plot
fig.show()
