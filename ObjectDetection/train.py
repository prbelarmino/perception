from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
#model = YOLO("yolov8n.yaml")
# Train the model
results = model.train(data="data.yaml", epochs=3)