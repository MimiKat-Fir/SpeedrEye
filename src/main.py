from ultralytics import YOLO

# Load a pretrained YOLO26 model
model = YOLO("yolo26n.pt")

# Train on kitti dataset
results = model.train(data="kitti.yaml", epochs=100, imgsz=640)