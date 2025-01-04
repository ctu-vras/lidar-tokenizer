import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'

i = 1000

ground_truth = np.load('fid_evaluation/frontcam/ground_truth.npz')
ground_truth = ground_truth[f'arr_{i}']

rec_linear = np.load('fid_evaluation/frontcam/reconstructions_ultralidar_linear.npz')
rec_linear = rec_linear[f'arr_{i}']

rec_log = np.load('fid_evaluation/frontcam/reconstructions_ultralidar_log.npz')
rec_log = rec_log[f'arr_{i}']

rec_polar = np.load('fid_evaluation/frontcam/reconstructions_ultralidar_polar.npz')
rec_polar = rec_polar[f'arr_{i}']

rec_ld = np.load('fid_evaluation/frontcam/reconstructions_ld_sensor_z.npz')
rec_ld = rec_ld[f'arr_{i}']

ld_truth = np.load('fid_evaluation/frontcam/ld_truth.npz')
ld_truth = ld_truth[f'arr_{i}']

fig = go.Figure(data=[go.Scatter3d(
    x=ground_truth[:,0],
    y=ground_truth[:,1],
    z=ground_truth[:,2],
    mode='markers',
    marker=dict(size=2, color='blue')
)])

fig.add_trace(go.Scatter3d(
    x=rec_linear[:,0],
    y=rec_linear[:,1],
    z=rec_linear[:,2],
    mode='markers',
    marker=dict(size=2, color='green')
))

fig.add_trace(go.Scatter3d(
    x=rec_log[:,0],
    y=rec_log[:,1],
    z=rec_log[:,2],
    mode='markers',
    marker=dict(size=2, color='black')
))

fig.add_trace(go.Scatter3d(
    x=rec_polar[:,0],
    y=rec_polar[:,1],
    z=rec_polar[:,2],
    mode='markers',
    marker=dict(size=2, color='red')
))

fig.add_trace(go.Scatter3d(
    x=rec_ld[:,0],
    y=rec_ld[:,1],
    z=rec_ld[:,2],
    mode='markers',
    marker=dict(size=2, color='orange')
))

fig.add_trace(go.Scatter3d(
    x=ld_truth[:,0],
    y=ld_truth[:,1],
    z=ld_truth[:,2],
    mode='markers',
    marker=dict(size=2, color='violet')
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

fig.show()