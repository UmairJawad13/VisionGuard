"""
Quick training script for fine-tuning YOLOv8 on hazard detection
"""
from ultralytics import YOLO
import torch
import os

def train_hazard_detector(
    data_yaml='datasets/hazards/data.yaml',
    model_size='n',  # n, s, m, l, x
    epochs=50,
    batch_size=16,
    img_size=640
):
    """
    Fine-tune YOLOv8 on hazard detection dataset
    
    Args:
        data_yaml: Path to dataset configuration file
        model_size: YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=extra-large)
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size
    """
    
    # Check if dataset exists
    if not os.path.exists(data_yaml):
        print(f"[ERROR] Dataset configuration not found: {data_yaml}")
        print("[INFO] Please follow DATASET_SETUP.md to prepare your dataset")
        return None
    
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Model: YOLOv8{model_size}")
    print(f"Dataset: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Image Size: {img_size}")
    print(f"{'='*60}\n")
    
    if device == 'cpu':
        print("[WARNING] Training on CPU. This will be slow!")
        print("[WARNING] Consider using Google Colab with GPU for faster training")
        response = input("Continue with CPU training? (y/n): ")
        if response.lower() != 'y':
            return None
    
    # Load pretrained model
    model_name = f'yolov8{model_size}.pt'
    print(f"[INFO] Loading pretrained model: {model_name}")
    model = YOLO(model_name)
    
    # Training configuration
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': img_size,
        'batch': batch_size,
        'device': device,
        'project': 'models',
        'name': 'hazard_detection',
        'patience': 10,
        'save': True,
        'plots': True,
        'pretrained': True,
        'freeze': 0,
        
        # Data augmentation
        'augment': True,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'fliplr': 0.5,
        'mosaic': 1.0,
        
        # Optimizer
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
    }
    
    print(f"\n[INFO] Starting training...")
    print(f"[INFO] This may take 30-120 minutes depending on your hardware")
    print(f"[INFO] Training progress will be saved to: models/hazard_detection/")
    
    try:
        # Train
        results = model.train(**train_args)
        
        # Validate
        print(f"\n[INFO] Running validation...")
        metrics = model.val()
        
        # Print results
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        print(f"Best model: models/hazard_detection/weights/best.pt")
        print(f"Last model: models/hazard_detection/weights/last.pt")
        print(f"\nPerformance Metrics:")
        print(f"  mAP@50: {metrics.box.map50:.4f}")
        print(f"  mAP@50-95: {metrics.box.map:.4f}")
        print(f"  Precision: {metrics.box.mp:.4f}")
        print(f"  Recall: {metrics.box.mr:.4f}")
        print(f"{'='*60}\n")
        
        # Copy best model
        import shutil
        best_model_path = 'models/hazard_detection/weights/best.pt'
        target_path = 'models/yolov8_finetuned.pt'
        
        if os.path.exists(best_model_path):
            shutil.copy(best_model_path, target_path)
            print(f"[SUCCESS] Fine-tuned model copied to: {target_path}")
            print(f"[INFO] VisionGuard will now use this model automatically!")
        
        return model
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLOv8 for hazard detection')
    parser.add_argument('--data', type=str, default='datasets/hazards/data.yaml',
                       help='Path to dataset.yaml')
    parser.add_argument('--model', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                       help='Model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size')
    
    args = parser.parse_args()
    
    train_hazard_detector(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz
    )
