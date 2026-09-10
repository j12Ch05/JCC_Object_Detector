from ultralytics import YOLO

# Load a pretrained YOLOv8 model as a starting point (transfer learning)
model = YOLO('yolov8n.pt')  # 'n' = nano, smallest/fastest. Options: n, s, m, l, x

# Train
results = model.train(
    data='/content/dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='yolo8_custom_mod'
)