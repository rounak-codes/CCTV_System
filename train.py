import torch
from ultralytics import YOLO
import os
import time

def train_model(model_path, data_path, epochs, img_size, batch_size):
    """Train a YOLO model with the specified parameters"""
    
    # Force PyTorch to use CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only use the first GPU
    
    # Print debug info
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    
    # Use a simple integer for device
    device = 0 if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*50}")
    print(f"Starting training for model: {model_path}")
    print(f"Dataset: {data_path}")
    print(f"{'='*50}\n")
    
    # Initialize the model
    model = YOLO(model_path)
    
    # Train the model
    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
    )
    
    print(f"\n{'='*50}")
    print(f"Completed training for model: {model_path}")
    print(f"{'='*50}\n")
    
    # Optional: Add a small delay between training jobs
    time.sleep(5)

def main():
    # Define the training jobs as a list of dictionaries
    training_jobs = [
        {
            "model_path": 'yolo11s.pt',
            "data_path": 'datasets/violenceprediction/data.yaml',
            "epochs": 100,
            "img_size": 416,
            "batch_size": 4
        },
        {
            "model_path": 'yolo11s.pt',  # Using the same base model
            "data_path": 'datasets/weaponsdata/data.yaml',
            "epochs": 100,
            "img_size": 416,
            "batch_size": 4
        },
        {
            "model_path": 'yolo11s.pt',  # Using the same base model
            "data_path": 'datasets/weaponsdetection/data.yaml',
            "epochs": 100,
            "img_size": 416,
            "batch_size": 4
        }
    ]
    
    # Execute each training job sequentially
    for job in training_jobs:
        train_model(**job)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()