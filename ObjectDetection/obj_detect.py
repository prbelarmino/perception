import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
detection_threshold = 0.8
# Capture video from the built-in camera
cap = cv2.VideoCapture(2)
color = (255,0,255)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    # Read the frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Mirror the frame horizontally
    results = model(frame)
    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > detection_threshold:
                #detections.append([x1, y1, x2, y2, score])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                label = f'{model.model.names[int(class_id)]}: {score:.2f}'
                cv2.putText(frame, label, (x1, y1 + -15), font, 1, color, 2)
    # Show the frame
    cv2.imshow("Object Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
