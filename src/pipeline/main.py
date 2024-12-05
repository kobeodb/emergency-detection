import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

# Load the YOLO model
model = YOLO("yolov8n.pt")  # Choose a smaller YOLO model for faster inference

# Initialize the SORT tracker
tracker = Sort()

# Video capture (0 for webcam or provide a video path)
cap = cv2.VideoCapture("./sudden cardiac arrest tatami.webm")  # Replace '0' with your video file path for testing

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for faster processing (adjust as needed)

    # Run YOLO inference
    results = model(frame, verbose=False)

    # Extract detections for people (class 0)
    detections = []
    for result in results[0].boxes:
        cls_id = int(result.cls)
        conf = float(result.conf)

        # Filter for class 0 (person) and confidence > 0.5
        if cls_id == 0 and conf > 0.5:
            x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates
            detections.append([x1, y1, x2, y2, conf])  # Add detection to list

    # Convert detections to NumPy array (required by SORT)
    detections = np.array(detections)

    # Update SORT tracker with detections
    tracked_objects = tracker.update(detections)

    # Draw bounding boxes with unique IDs
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = map(int, obj)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Bounding box
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame with detections
    cv2.imshow("Person Detection with SORT", frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
