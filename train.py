import torch
from ultralytics import YOLO

def main():
    model = YOLO('yolo11x.pt')  # Load the YOLO model
    model.train(
        data='datasets/weaponsdetection/data.yaml',  # Path to the YAML file
        epochs=100,  # Number of training epochs
        imgsz=64,  # Image size
    )

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)  # Ensure proper process spawning
    main()
