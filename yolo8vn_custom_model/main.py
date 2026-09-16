from ultralytics import YOLO
import cv2
import os

model = YOLO(os.path.join(os.path.dirname(__file__), "best1.pt"))

cap = cv2.VideoCapture(0)

cv2.namedWindow('Yolo Detection', cv2.WINDOW_NORMAL)  # <-- add this line
cv2.resizeWindow('Yolo Detection', 1280, 720)          # <-- optional: set a starting size

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    results = model(frame, conf=0.5)
    annoted_frame = results[0].plot()
    cv2.imshow('Yolo Detection', annoted_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()