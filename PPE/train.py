import os
import itertools
from ultralytics import YOLO

def train_yolo(lr, batch_size, img_size, epochs=10, device=0):
    model = YOLO("yolov8n.pt")  # Load YOLOv8n model
    results = model.train(
        data="D:/L and T/data.yaml",
        epochs=epochs,
        imgsz=img_size,
        device=device,  # Use GPU
        lr0=lr,
        batch=batch_size,
        augment=True
    )
    return results

if __name__ == '__main__':
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Define hyperparameter grid
    learning_rates = [0.001, 0.005, 0.01]
    batch_sizes = [4, 8, 16]  # Reduce batch sizes
    img_sizes = [256, 320, 416]  # Reduce image sizes
    
    best_mAP = 0
    best_params = None
    best_model_path = ""
    
    # Iterate through hyperparameter combinations
    for lr, batch, img_size in itertools.product(learning_rates, batch_sizes, img_sizes):
        print(f"Training with LR={lr}, Batch Size={batch}, Image Size={img_size}")
        results = train_yolo(lr, batch, img_size)
        
        # Extract performance metrics
        mAP_05 = results.metrics['mAP_0.5']
        mAP_05_95 = results.metrics['mAP_0.5:0.95']
        
        # Save model and log results
        model_path = f"models/yolo_lr{lr}_batch{batch}_img{img_size}.pt"
        results.model.save(model_path)
        
        with open(f"logs/training_lr{lr}_batch{batch}_img{img_size}.log", "w") as f:
            f.write(f"Training Complete! Model saved at {model_path}\n")
            f.write(f"LR: {lr}, Batch Size: {batch}, Image Size: {img_size}\n")
            f.write(f"Final Epoch: {results.epoch}\n")
            f.write(f"Training Loss: {results.loss}\n")
            f.write(f"Validation Loss: {results.val_loss}\n")
            f.write(f"mAP@0.5: {mAP_05}\n")
            f.write(f"mAP@0.5:0.95: {mAP_05_95}\n")
            f.write(f"Precision: {results.metrics['precision']}\n")
            f.write(f"Recall: {results.metrics['recall']}\n")
        
        print(f"Training completed with mAP@0.5: {mAP_05}, mAP@0.5:0.95: {mAP_05_95}")
        
        # Update best model
        if mAP_05 > best_mAP:
            best_mAP = mAP_05
            best_params = (lr, batch, img_size)
            best_model_path = model_path
    
    print(f"Best model: {best_model_path} with LR={best_params[0]}, Batch={best_params[1]}, Img Size={best_params[2]}, mAP@0.5={best_mAP}")
