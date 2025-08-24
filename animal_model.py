from ultralytics import YOLO
import glob

# Load YOLOv8 pretrained on Open Images V7
model = YOLO("yolov8x-oiv7.pt")

## Get all jpg/png images inside test_images folder
image_paths = glob.glob("test_images/*.jpg") + glob.glob("test_images/*.png")

# Run detection on all images
results = model(image_paths, save=True)

# Print detected classes for each image
for img_path, r in zip(image_paths, results):
    print(f"\n📷 Results for {img_path}:")
    detected_classes = [model.names[int(c)] for c in r.boxes.cls]
    if detected_classes:
        print("Detected:", ", ".join(set(detected_classes)))
    else:
        print("No objects detected.")
