import cv2
import pandas as pd
from ultralytics import YOLO

# Define the keypoint indices for nose, wrist, elbow, hip, knee
keypoint_indices = [0, 9, 10, 7, 8, 5, 6, 11, 12, 13, 14]

def write_pose_video(video_path, csv_path, output_path, frame_lock, current_frame):
    model = YOLO("yolov8s-pose.pt")  # Load a pretrained YOLOv8 pose model
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for result in results:
            row = {}
            keypoints = result.keypoints
            print(f"Keypoints: {keypoints}")  # Debug: Print keypoints

            if keypoints is not None and keypoints.xy.shape[0] > 0:
                kp_val = keypoints.xy.cpu().numpy()
                for idx in keypoint_indices:
                    if kp_val.shape[0] > idx:
                        row[f'kp_{idx}_x'] = kp_val[idx][0] * width
                        row[f'kp_{idx}_y'] = kp_val[idx][1] * height
                    else:
                        row[f'kp_{idx}_x'] = None
                        row[f'kp_{idx}_y'] = None
            else:
                for idx in keypoint_indices:
                    row[f'kp_{idx}_x'] = None
                    row[f'kp_{idx}_y'] = None

            data.append(row)

            with frame_lock:
                current_frame[0] = frame.copy()

            writer.write(frame)

    cap.release()
    writer.release()

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
