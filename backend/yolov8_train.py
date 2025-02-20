from ultralytics import YOLO
import os
from tqdm import tqdm

# Debugging: Print the current working directory
print(f"Current working directory: {os.getcwd()}")



# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Pretrained model

# Train on the blueprint dataset
data_path = os.path.join(os.getcwd(), "architectural-blueprint", "data.yaml")
print(f"Data path: {data_path}")

# Initialize progress bar
epochs = 10
with tqdm(total=epochs, desc="Training Progress") as pbar:
    for epoch in range(epochs):
        results = model.train(data=data_path, epochs=5, imgsz=320, device=0)
        pbar.update(1)

# Print the results
print(f"Training results: {results}")
print(f"Model accuracy: {results.metrics['accuracy']}")
print(f"Model F1 score: {results.metrics['f1']}")