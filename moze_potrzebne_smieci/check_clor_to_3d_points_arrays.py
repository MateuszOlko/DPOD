from DPOD.models_handler import *
from plotly.graph_objects import Scatter3d, Figure
from plotly.subplots import make_subplots

model_handler = ModelsHandler('data/kaggle')
model_id = 5
vertices, _ = model_handler.model_id_to_vertices_and_triangles(model_id)
color_array = model_handler.get_color_to_3dpoints_arrays(model_id)
color_array_vertices = color_array.reshape(-1, 3)[::10]
fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])
fig.add_trace(
    Scatter3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        mode='markers',
        marker_color=model_handler.color_points(vertices, model_id)
    ),
    row=1, col=1
)
fig.add_trace(
    Scatter3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        mode='markers',
        marker_color=model_handler.color_points(color_array_vertices, model_id)
    ),
    row=1, col=2
)
fig.show()
