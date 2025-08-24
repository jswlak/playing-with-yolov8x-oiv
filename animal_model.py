from ultralytics import YOLO

# Load YOLOv8 pretrained on Open Images V7
model = YOLO("yolov8x-oiv7.pt")   # 'x' = extra large, best accuracy
# You can also try: yolov8n-oiv7.pt, yolov8s-oiv7.pt, yolov8m-oiv7.pt, yolov8l-oiv7.pt

# Run detection on an image
results = model("test_image.jpg", save=True)

# Show class names
print(model.names)
