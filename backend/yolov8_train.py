from ultralytics import YOLO
import os
from tqdm import tqdm
import torch

def train_model():
    # Debugging: Print the current working directory
    print(f"Current working directory: {os.getcwd()}")

    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please ensure you have a compatible GPU and CUDA installed.")

    device = 'cuda'
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Load YOLOv8 model
    model = YOLO("yolov8n.pt")  # Pretrained model

    # Train on the blueprint dataset
    data_path = os.path.join(os.getcwd(), "architectural-blueprint", "data.yaml")
    print(f"Data path: {data_path}")

    # Train the model for 20 epochs
    results = model.train(data=data_path, epochs=20, imgsz=320, device=device)

    # Save the trained model
    model.save("best.pt")

    
if __name__ == '__main__':
    train_model()