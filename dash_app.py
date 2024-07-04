import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import threading
import cv2
import base64
import numpy as np
from mediapipe_utils import write_pose_video
import dash_bootstrap_components as dbc

CSV_FILE = 'pose.csv'
VIDEO_FILE = 'DJI_0886.MP4'
OUTPUT_VISUALIZATION_FILE = 'visualization_output.mp4'

# Define CSS for aspect ratio (16:10 for Samsung Galaxy Tab S9)
external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    {
        'href': 'https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css',
        'rel': 'stylesheet',
        'integrity': 'sha384-J6qa4849blE2+poT4WnyOdI7mDE5P3p6L4L4/jWz2mgm3ApD4+rh4bcy4vZvVskR',
        'crossorigin': 'anonymous'
    },
    {
        'href': 'https://fonts.googleapis.com/css2?family=Roboto&display=swap',
        'rel': 'stylesheet'
    },
    {
        'href': 'https://use.fontawesome.com/releases/v5.8.1/css/all.css',
        'rel': 'stylesheet'
    }
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Define video dimensions
VIDEO_HEIGHT = 360  
VIDEO_WIDTH = 640   

# Define the layout using Dash Bootstrap Components and CSS for aspect ratio
app.layout = html.Div(
    className='aspect-ratio-16-10',  # CSS class for 16:10 aspect ratio
    children=[
        dbc.Row([
            dbc.Col(dcc.Graph(id='live-wrist-graph'), width={'size': 4}),
            dbc.Col(dcc.Graph(id='live-elbow-graph'), width={'size': 4}),
            dbc.Col(dcc.Graph(id='live-hip-graph'), width={'size': 4}),
        ]),
        dbc.Row([
            dbc.Col(html.Div(html.Img(id='live-video', style={'height': VIDEO_HEIGHT, 'width': VIDEO_WIDTH, 'display': 'block', 'margin': 'auto'}))),
        ]),
        dcc.Interval(
            id='interval-component',
            interval=100,  # 1000 milliseconds = 1 second
            n_intervals=0
        )
    ]
)

current_frame = [None]
frame_lock = threading.Lock()

t1 = threading.Thread(target=write_pose_video, args=(VIDEO_FILE, CSV_FILE, OUTPUT_VISUALIZATION_FILE, frame_lock, current_frame))
t1.start()

@app.callback(
    [Output('live-wrist-graph', 'figure'),
     Output('live-elbow-graph', 'figure'),
     Output('live-hip-graph', 'figure'),
     Output('live-video', 'src')],
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n_intervals):
    try:
        data = pd.read_csv(CSV_FILE)
    except Exception as e:
        return {}, {}, {}, ''

    x = np.arange(len(data))
    left_wrist_y = data['15']
    right_wrist_y = data['16']
    left_elbow_y = data['13']
    right_elbow_y = data['14']
    left_hip_y = data['23']
    right_hip_y = data['24']

    line_width = 1  # Adjust line width as needed

    wrist_fig = go.Figure()
    wrist_fig.add_trace(go.Scatter(x=x, y=left_wrist_y, mode='lines', name='Left wrist', line=dict(width=line_width)))
    wrist_fig.add_trace(go.Scatter(x=x, y=right_wrist_y, mode='lines', name='Right wrist', line=dict(width=line_width)))
    wrist_fig.update_layout(title='Wrist Movements', xaxis_title='Frame Index', yaxis_title='Pixel Position')

    elbow_fig = go.Figure()
    elbow_fig.add_trace(go.Scatter(x=x, y=left_elbow_y, mode='lines', name='Left elbow', line=dict(width=line_width)))
    elbow_fig.add_trace(go.Scatter(x=x, y=right_elbow_y, mode='lines', name='Right elbow', line=dict(width=line_width)))
    elbow_fig.update_layout(title='Elbow Movements', xaxis_title='Frame Index', yaxis_title='Pixel Position')

    hip_fig = go.Figure()
    hip_fig.add_trace(go.Scatter(x=x, y=left_hip_y, mode='lines', name='Left hip', line=dict(width=line_width)))
    hip_fig.add_trace(go.Scatter(x=x, y=right_hip_y, mode='lines', name='Right hip', line=dict(width=line_width)))
    hip_fig.update_layout(title='Hip Movements', xaxis_title='Frame Index', yaxis_title='Pixel Position')

    with frame_lock:
        if current_frame[0] is not None:
            ret, buffer = cv2.imencode('.jpg', current_frame[0])
            frame_src = 'data:image/jpg;base64,' + base64.b64encode(buffer).decode('utf-8')
        else:
            frame_src = ''

    return wrist_fig, elbow_fig, hip_fig, frame_src

if __name__ == '__main__':
    app.run_server(debug=True)
