import cv2
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from ultralytics import YOLO
import pandas as pd
import os

VIDEO_FILE = 'DJI_0886.MP4'
OUTPUT_FILE = 'keypoints.txt'

# Load your trained model
model = YOLO('yolov8n-pose.pt')

# Function to extract keypoints and save to file
def extract_keypoints():
    # Open the video file
    cap = cv2.VideoCapture(VIDEO_FILE)
    
    # Open the file to write keypoints
    with open(OUTPUT_FILE, 'w') as f:
        # Process video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run prediction on the frame
            results = model(frame)
            
            # Iterate over each result in the results list
            for result in results:
                # Check if the result has keypoints attribute
                if hasattr(result, 'keypoints') and result.keypoints.data.numel() > 0:
                    keypoints = result.keypoints.data
                    for kp in keypoints[0]:
                        x, y, conf = kp.tolist()
                        f.write(f'{x}, {y}, {conf}\n')
                else:
                    # Write zeros if no keypoints are detected
                    num_keypoints = 17  # assuming 17 keypoints
                    for _ in range(num_keypoints):
                        f.write('0, 0, 0\n')
    
    # Release the video capture object
    cap.release()

# Start keypoint extraction in a separate thread
import threading
threading.Thread(target=extract_keypoints).start()

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='live-update-graph'),
    dcc.Interval(
        id='interval-component',
        interval=1*1000,  # in milliseconds
        n_intervals=0
    )
])

# Function to read keypoints from file
def read_keypoints(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, names=['x', 'y', 'conf'])
        return df
    else:
        return pd.DataFrame(columns=['x', 'y', 'conf'])

@app.callback(
    Output('live-update-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_graph_live(n):
    df = read_keypoints(OUTPUT_FILE)

    # Create line plot
    trace = go.Scatter(
        x=df['x'],
        y=df['y'],
        mode='lines+markers',
        marker=dict(size=8, opacity=0.5)
    )

    layout = go.Layout(
        title='Real-Time Keypoint Plot',
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        showlegend=False
    )

    return {'data': [trace], 'layout': layout}

if __name__ == '__main__':
    app.run_server(debug=True)
