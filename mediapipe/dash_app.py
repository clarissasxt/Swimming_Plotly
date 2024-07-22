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
VIDEO_FILE = 'DJI_0087.MP4'
OUTPUT_VISUALIZATION_FILE = 'visualization_output.mp4'

external_stylesheets = [dbc.themes.BOOTSTRAP, '/assets/styles.css']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

VIDEO_HEIGHT = 360  
VIDEO_WIDTH = 640

# Define the layout using Dash Bootstrap Components
app.layout = html.Div([
    dbc.Row([
        dbc.Col(dcc.Dropdown(
            id='graph-dropdown',
            options=[
                {'label': 'Speed', 'value': 'Speed'},
                {'label': 'Wrist', 'value': 'Wrist'},
                {'label': 'Elbow', 'value': 'Elbow'},
                {'label': 'Hip', 'value': 'Hip'},
                {'label': 'Knee', 'value': 'Knee'}
            ],
            value=['Speed','Wrist', 'Hip'], # default plots shown on screen
            multi=True,
            className='dropdown-bar'
        )),
    ], ),
    dbc.Row(id='graph-row'),
    dbc.Row([
        dbc.Col(html.Div(html.Img(id='live-video', style={'height': VIDEO_HEIGHT, 'width': VIDEO_WIDTH, 'display': 'block', 'margin': 'auto'}))),
    ]),
    dcc.Interval(
        id='interval-component',
        interval=100,  
        n_intervals=0
    )
])

current_frame = [None]
frame_lock = threading.Lock()

t1 = threading.Thread(target=write_pose_video, args=(VIDEO_FILE, CSV_FILE, OUTPUT_VISUALIZATION_FILE, frame_lock, current_frame))
t1.start()

@app.callback(
    Output('graph-row', 'children'),
    [Input('graph-dropdown', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_graph(selected_graphs, n_intervals):
    try:
        data = pd.read_csv(CSV_FILE)
    except Exception as e:
        return []

    x = np.arange(len(data))

    figures = {
        'Speed': {
            'left': go.Scatter(x=x, y=data['0'], mode='lines', name='Speed', line=dict(width=1)),
            'title': 'Speed'
        },
        'Wrist': {
            'left': go.Scatter(x=x, y=data['15'], mode='lines', name='Left wrist', line=dict(width=1)),
            'right': go.Scatter(x=x, y=data['16'], mode='lines', name='Right wrist', line=dict(width=1)),
            'title': 'Wrist Movements'
        },
        'Elbow': {
            'left': go.Scatter(x=x, y=data['13'], mode='lines', name='Left elbow', line=dict(width=1)),
            'right': go.Scatter(x=x, y=data['14'], mode='lines', name='Right elbow', line=dict(width=1)),
            'title': 'Elbow Movements'
        },
        'Hip': {
            'left': go.Scatter(x=x, y=data['23'], mode='lines', name='Left hip', line=dict(width=1)),
            'right': go.Scatter(x=x, y=data['24'], mode='lines', name='Right hip', line=dict(width=1)),
            'title': 'Hip Movements'
        },
        'Knee': {
            'left': go.Scatter(x=x, y=data['25'], mode='lines', name='Left knee', line=dict(width=1)),
            'right': go.Scatter(x=x, y=data['26'], mode='lines', name='Right knee', line=dict(width=1)),
            'title': 'Knee Movements'
        }
    }

    graph_components = []
    for graph_type in selected_graphs:
        if graph_type in figures:
            fig = go.Figure()
            if 'left' in figures[graph_type]:
                fig.add_trace(figures[graph_type]['left'])
            if 'right' in figures[graph_type]:
                fig.add_trace(figures[graph_type]['right'])
            fig.update_layout(
                title=figures[graph_type]['title'], 
                xaxis_title='Frame Index', 
                yaxis_title='Pixel Position',
                margin=dict(t=50, b=30, l=30, r=30), 
                height=350 # height of graph
                )

            graph_components.append(dbc.Col(dcc.Graph(figure=fig), width=4))

    return graph_components

@app.callback(
    Output('live-video', 'src'),
    [Input('interval-component', 'n_intervals')]
)
def update_video(n_intervals):
    with frame_lock:
        if current_frame[0] is not None:
            ret, buffer = cv2.imencode('.jpg', current_frame[0])
            frame_src = 'data:image/jpg;base64,' + base64.b64encode(buffer).decode('utf-8')
        else:
            frame_src = ''
    return frame_src

if __name__ == '__main__':
    app.run_server(debug=True)