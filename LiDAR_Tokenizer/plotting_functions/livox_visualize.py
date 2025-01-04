import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# Set default renderer
pio.renderers.default = 'browser'

# Load data
data = np.load('/data/npz/time_1725621594.7428.npz')
p = data['point_cloud'][:,:3]

#mask = data['point_cloud'][:,4] == 6
p_masked = p#[mask,:]

# Create the 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=p_masked[:, 0],
    y=p_masked[:, 1],
    z=p_masked[:, 2],
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

# Show plot
fig.show()
