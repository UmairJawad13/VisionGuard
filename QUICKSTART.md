# 🚀 Quick Start Guide - VisionGuard

## Installation (5 minutes)

### 1. Install Python Packages
```powershell
pip install -r requirements.txt
```

### 2. Install Ollama + LLaVA
1. Download from: https://ollama.ai
2. Install and run:
   ```powershell
   ollama serve
   ```
3. In a new terminal:
   ```powershell
   ollama pull llava
   ```

### 3. Verify Installation
```powershell
python test_system.py
```

## Running the Application

### Start VisionGuard
```powershell
python main.py
```

### Controls
- **Click "Start Camera"** or press button
- **SPACE** - Describe scene with AI
- **R** - Read text from image
- **D** - Toggle debug mode
- **Q** - Quit

## For Your Assignment

### 1. Collect Debug Logs
- Enable logging (default: ON)
- Run in various conditions (dark, bright, motion)
- Check `logs/images/` and `logs/detections/` folders

### 2. Run Evaluation
```powershell
# Basic evaluation
python evaluate.py --images test_images

# With ground truth labels
python evaluate.py --images test_images --labels test_labels

# Full validation (if you have dataset.yaml)
python evaluate.py --data-yaml datasets/hazards/data.yaml
```

### 3. Run Bias Testing
Organize test images in folders:
```
test_images/bias_test/
├── western_cities/
├── asian_cities/
├── dark_lighting/
└── bright_lighting/
```

Then run:
```powershell
python bias_test.py
```

### 4. Fine-tune Model (Optional)

#### Download Dataset
1. Visit: https://universe.roboflow.com
2. Search: "stairs detection" or "pothole detection"
3. Download in **YOLOv8** format
4. Extract to `datasets/hazards/`

#### Verify Dataset
```powershell
python verify_dataset.py --data datasets/hazards/data.yaml
```

#### Train Model
```powershell
# Quick training (50 epochs, ~30-60 min)
python train_model.py --data datasets/hazards/data.yaml --epochs 50

# Better quality (100 epochs, ~60-120 min)
python train_model.py --data datasets/hazards/data.yaml --epochs 100 --model s
```

## Troubleshooting

### Camera not working?
- Check camera permissions
- Change `CAMERA_INDEX = 1` in `utils/config.py`

### Ollama connection error?
```powershell
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Check if running
ollama list
```

### Low FPS?
- Use CPU-friendly settings in `utils/config.py`
- Close other applications
- Reduce `CAMERA_WIDTH` and `CAMERA_HEIGHT`

### Import errors?
```powershell
pip install -r requirements.txt --upgrade
```

## Files for Your Report

After running the application and evaluations, collect:

### From `logs/` folder:
- `performance_report.json` - Performance metrics
- `evaluation_report.json` - mAP, IoU, F1 scores
- `iou_distribution.png` - IoU histogram
- `performance_metrics.png` - Precision/Recall/F1 chart
- `confusion_matrix.png` - Confusion matrix
- `bias_test_results.csv` - Confidence by category
- `bias_confidence_comparison.png` - Bias analysis

### From `logs/images/` and `logs/detections/`:
- Sample frames showing successful detections
- Sample frames showing failures (dark lighting, motion blur)
- JSON files with confidence scores

### From `models/hazard_detection/` (if you trained):
- Training plots
- Validation results
- mAP curves

## Next Steps

1. ✅ Run `test_system.py` to verify everything works
2. ✅ Run `main.py` and test all features
3. ✅ Collect debug logs in different scenarios
4. ✅ Run `evaluate.py` for metrics
5. ✅ Run `bias_test.py` for bias analysis
6. ✅ (Optional) Download dataset and train model
7. ✅ Write your report using the generated data!

## Need Help?

- Check `README.md` for detailed documentation
- Review `DATASET_SETUP.md` for training guide
- Check configuration in `utils/config.py`

---

**Good luck with your assignment! 🎓**
