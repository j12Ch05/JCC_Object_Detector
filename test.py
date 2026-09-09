from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")
model.predict(source="images/bus.jpg")