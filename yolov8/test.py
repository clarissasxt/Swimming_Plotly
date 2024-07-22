import cv2
from ultralytics import YOLO
import torch

VIDEO_FILE = 'DJI_0886.MP4'
OUTPUT_FILE = 'keypoints.txt'

# Load your trained model
model = YOLO('yolov8n-pose.pt') 

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

        # Optionally display the frame (for visualization purposes)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
