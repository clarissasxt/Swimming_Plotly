import csv
import mediapipe as mp
import cv2
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import logging
from threading import Thread, Lock
import time

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def detect_pose_landmarks(frame):
    # Convert BGR frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with Mediapipe
    results = mp_pose.process(rgb_frame)

    return results

def draw_landmarks_on_image(rgb_image, detection_result):
    annotated_image = np.copy(rgb_image)

    if detection_result.pose_landmarks:
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            detection_result.pose_landmarks,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    
    return annotated_image

def write_pose_video(inp_video, csv_file, out_video, frame_lock, current_frame):
    fieldnames = ['15', '16', '13', '14', '23', '24']
    with open(csv_file, 'w') as file:
        csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
        csv_writer.writeheader()

    cap = cv2.VideoCapture(inp_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1.0 / fps

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video, fourcc, fps, (w, h))

    with mp_pose as pose:
        frame_i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Detect pose landmarks
                pose_landmarks_result = detect_pose_landmarks(frame)
                
                # Draw landmarks on frame
                annotated_image = draw_landmarks_on_image(frame, pose_landmarks_result)
                out.write(annotated_image)

                with frame_lock:
                    current_frame[0] = annotated_image.copy()
                
                with open(csv_file, 'a') as file:
                    csv_writer  = csv.DictWriter(file, fieldnames=fieldnames)
                    if pose_landmarks_result.pose_landmarks:
                        info = {
                            '15': int(pose_landmarks_result.pose_landmarks.landmark[15].x * w),
                            '16': int(pose_landmarks_result.pose_landmarks.landmark[16].x * w),
                            '13': int(pose_landmarks_result.pose_landmarks.landmark[13].x * w),
                            '14': int(pose_landmarks_result.pose_landmarks.landmark[14].x * w),
                            '23': int(pose_landmarks_result.pose_landmarks.landmark[23].x * w),
                            '24': int(pose_landmarks_result.pose_landmarks.landmark[24].x * w)
                        }
                    else:
                        info = {
                            '15': None, '16': None, '13': None, '14': None, '23': None, '24': None
                        }
                    csv_writer.writerow(info)
                    logging.info(info)

                frame_i += 1
            else:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()
