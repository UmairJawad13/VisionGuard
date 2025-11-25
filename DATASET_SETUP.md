# Dataset Setup & Model Fine-tuning Guide

This guide explains how to download hazard detection datasets and fine-tune YOLOv8 for VisionGuard.

## 🎯 Goal

Create a "Super Model" that detects:
- **COCO classes** (Person, Car, Chair, etc.) - Already trained
- **Hazard classes** (Stairs, Potholes, Doors, Stop Signs) - Need fine-tuning

## 📥 Step 1: Download Hazard Datasets

### Option A: Roboflow Universe (Recommended)

1. **Visit Roboflow Universe**: https://universe.roboflow.com/

2. **Search for datasets:**
   - "Stairs Detection"
   - "Pothole Detection"
   - "Sidewalk Obstacles"
   - "Accessibility Hazards"

3. **Download format:** Select **YOLOv8** format

4. **Good starter datasets:**
   - **Stairs Dataset**: Search "stairs detection object"
   - **Pothole Dataset**: Search "pothole detection"
   - **Accessibility**: Search "wheelchair navigation" or "sidewalk"

### Option B: Kaggle

1. **Visit Kaggle**: https://www.kaggle.com/datasets

2. **Search for:**
   - "Stairs detection dataset"
   - "Pothole detection"
   - "VisionGuard dataset"

3. **Download and convert** to YOLO format if needed

### Option C: Create Custom Dataset

Use tools like:
- **LabelImg**: https://github.com/heartexlabs/labelImg
- **Roboflow**: Create a free account and upload your images
- **CVAT**: https://www.cvat.ai/

## 📂 Step 2: Organize Dataset

After downloading, organize your dataset like this:

```
IPCV A2/
└── datasets/
    └── hazards/           # Your downloaded dataset
        ├── data.yaml      # Dataset configuration
        ├── train/
        │   ├── images/
        │   │   ├── img1.jpg
        │   │   ├── img2.jpg
        │   │   └── ...
        │   └── labels/
        │       ├── img1.txt
        │       ├── img2.txt
        │       └── ...
        ├── valid/
        │   ├── images/
        │   └── labels/
        └── test/
            ├── images/
            └── labels/
```

### Create `data.yaml`

If not provided, create `datasets/hazards/data.yaml`:

```yaml
# Dataset configuration for hazard detection

# Paths (use absolute paths or relative to this file)
path: ./  # Root directory
train: train/images
val: valid/images
test: test/images

# Classes
names:
  0: stairs
  1: pothole
  2: door
  3: stop_sign
  4: curb

# Number of classes
nc: 5
```

**Important:** Adjust class names and count based on your dataset!

## 🔧 Step 3: Verify Dataset

Run this Python script to verify your dataset:

```python
# verify_dataset.py
import os
from pathlib import Path
import yaml

# Load dataset config
with open('datasets/hazards/data.yaml', 'r') as f:
    data = yaml.safe_load(f)

print("Dataset Configuration:")
print(f"  Path: {data['path']}")
print(f"  Classes: {data['names']}")
print(f"  Number of classes: {data['nc']}")

# Check images and labels
for split in ['train', 'valid', 'test']:
    img_dir = Path(data[split]).parent / 'images'
    lbl_dir = Path(data[split]).parent / 'labels'
    
    img_count = len(list(img_dir.glob('*.jpg'))) + len(list(img_dir.glob('*.png')))
    lbl_count = len(list(lbl_dir.glob('*.txt')))
    
    print(f"\n{split.upper()}:")
    print(f"  Images: {img_count}")
    print(f"  Labels: {lbl_count}")
    print(f"  Match: {'✓' if img_count == lbl_count else '✗ MISMATCH!'}")
```

## 🚀 Step 4: Fine-tune YOLOv8

### Create Training Script

Save as `train_model.py`:

```python
"""
Fine-tune YOLOv8 on hazard detection dataset
"""
from ultralytics import YOLO
import torch

# Check GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training on: {device}")

# Load pretrained YOLOv8 model
model = YOLO('yolov8n.pt')  # Start with YOLOv8 nano (fastest)
# Use yolov8s.pt for small, yolov8m.pt for medium (more accurate, slower)

# Train the model
results = model.train(
    data='datasets/hazards/data.yaml',  # Path to your data.yaml
    epochs=50,                          # Number of epochs (50-100 recommended)
    imgsz=640,                          # Image size
    batch=16,                           # Batch size (adjust based on GPU memory)
    device=device,                      # GPU or CPU
    project='models',                   # Save location
    name='hazard_detection',            # Experiment name
    patience=10,                        # Early stopping patience
    save=True,                          # Save checkpoints
    plots=True,                         # Generate training plots
    
    # Transfer learning settings
    pretrained=True,                    # Use pretrained weights
    freeze=0,                           # Layers to freeze (0 = train all)
    
    # Data augmentation
    augment=True,
    hsv_h=0.015,                       # HSV-Hue augmentation
    hsv_s=0.7,                         # HSV-Saturation
    hsv_v=0.4,                         # HSV-Value
    degrees=0.0,                       # Rotation
    translate=0.1,                     # Translation
    scale=0.5,                         # Scale
    fliplr=0.5,                        # Horizontal flip probability
    mosaic=1.0,                        # Mosaic augmentation
)

# Evaluate
metrics = model.val()

print("\n" + "="*60)
print("Training Complete!")
print("="*60)
print(f"Best model saved to: models/hazard_detection/weights/best.pt")
print(f"mAP@50: {metrics.box.map50:.4f}")
print(f"mAP@50-95: {metrics.box.map:.4f}")
```

### Run Training

```bash
python train_model.py
```

**Training tips:**
- Start with 50 epochs, increase if needed
- Monitor training plots in `models/hazard_detection/`
- If overfitting: Reduce epochs, increase augmentation
- If underfitting: Increase epochs, use larger model (yolov8s/m)
- Training takes 30-120 minutes depending on dataset size and GPU

## 📊 Step 5: Copy Fine-tuned Model

After training completes:

```bash
# Copy best model to models directory
cp models/hazard_detection/weights/best.pt models/yolov8_finetuned.pt
```

Or on Windows PowerShell:
```powershell
Copy-Item models\hazard_detection\weights\best.pt models\yolov8_finetuned.pt
```

## 🔍 Step 6: Test Fine-tuned Model

### Quick Test Script

```python
# test_finetuned.py
from ultralytics import YOLO
import cv2

# Load fine-tuned model
model = YOLO('models/yolov8_finetuned.pt')

# Test on an image
img_path = 'test_images/test1.jpg'
results = model(img_path)

# Display results
results[0].show()

# Print detections
for box in results[0].boxes:
    class_id = int(box.cls[0])
    class_name = model.names[class_id]
    confidence = float(box.conf[0])
    print(f"Detected: {class_name} ({confidence:.2f})")
```

## 🔄 Step 7: Update VisionGuard to Use Fine-tuned Model

The application automatically uses the fine-tuned model if it exists at `models/yolov8_finetuned.pt`.

**Or manually specify in code:**

```python
# In main.py or when initializing Navigator
navigator = Navigator(use_finetuned=True)
```

**Update `utils/config.py`** to include your new classes:

```python
SAFETY_CLASSES = {
    # COCO classes
    0: "person",
    2: "car",
    56: "chair",
    
    # Custom hazard classes (adjust IDs based on your model)
    80: "stairs",
    81: "pothole",
    82: "door",
    83: "stop_sign",
    84: "curb"
}
```

## 📈 Step 8: Evaluate Fine-tuned Model

```bash
python evaluate.py --model models/yolov8_finetuned.pt --images test_images
```

## 🎯 Recommended Dataset Sizes

| Purpose | Minimum Images | Recommended |
|---------|---------------|-------------|
| Quick test | 100-200 | 500+ |
| Assignment | 300-500 | 1000+ |
| Production | 1000+ | 5000+ |

**Tip:** Quality > Quantity. Well-labeled, diverse images are better than many poor-quality ones.

## 🌍 Dataset Diversity Checklist

For robust performance, ensure your dataset includes:

- ✅ Different lighting conditions (day, night, indoor)
- ✅ Various weather (sunny, cloudy, rainy)
- ✅ Multiple angles and distances
- ✅ Different environments (urban, suburban, indoor)
- ✅ Occlusions and partial views
- ✅ Various object sizes and positions

## 🐛 Common Issues

### Issue: "Dataset not found"
**Solution:** Use absolute paths in `data.yaml`:
```yaml
path: C:/Users/user/Documents/UOW/.../IPCV A2/datasets/hazards
```

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size:
```python
batch=8  # or even batch=4
```

### Issue: Poor validation results
**Solution:**
1. Check label quality (visualize with `results[0].show()`)
2. Increase epochs (try 100)
3. Add more training data
4. Use data augmentation

### Issue: Model not detecting COCO classes after fine-tuning
**Solution:** This shouldn't happen with transfer learning, but if it does:
1. Don't freeze any layers (`freeze=0`)
2. Use lower learning rate
3. Consider training in two stages:
   - Stage 1: Freeze backbone, train head only
   - Stage 2: Unfreeze all, fine-tune end-to-end

## 📚 Additional Resources

- **YOLOv8 Documentation**: https://docs.ultralytics.com/
- **Roboflow Blog**: https://blog.roboflow.com/
- **Dataset Labeling Guide**: https://roboflow.com/annotate
- **Transfer Learning Guide**: https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings

## 🎓 For Your Report

Document these sections:
1. **Dataset Description**: Size, classes, sources
2. **Training Process**: Epochs, batch size, augmentation used
3. **Results**: mAP scores before/after fine-tuning
4. **Challenges**: Label quality issues, class imbalance, etc.
5. **Performance**: FPS comparison, accuracy tradeoffs

Save training plots from `models/hazard_detection/` for your report!

---

**Questions?** Check the main README.md or review training logs in `models/hazard_detection/`
