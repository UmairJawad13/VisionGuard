# VisionGuard - Navigation Assistant for Visually Impaired Users

A hybrid computer vision application that combines real-time object detection (YOLOv8) with multimodal AI (Vision + Audio + Language) to assist visually impaired users with navigation and scene understanding.

## 🎯 Project Overview

VisionGuard implements a **Hybrid Architecture**:
- **Local Processing (YOLOv8)**: Fast, real-time obstacle detection with audio warnings
- **Cloud Processing (LLaVA via Ollama)**: Deep scene understanding and natural language descriptions

## ✨ Key Features

### Module A: Real-Time Navigator 🚶
- Live webcam object detection using YOLOv8
- Detects safety-critical objects: Person, Car, Chair, Stairs, Door, Stop Sign
- Distance estimation based on bounding box area
- Position announcements (left, center, right)
- Audio warnings for hazards

### Module B: Scene Assistant 🤖
- Press **SPACEBAR** to describe the current scene
- Uses LLaVA (Large Language and Vision Assistant) via Ollama
- Provides detailed scene descriptions focused on navigation assistance
- Identifies hazards and suggests safe paths

### Module C: Text Reader 📖
- Press **R** to read text from the current frame
- Uses EasyOCR for text detection and recognition
- Converts detected text to speech
- Useful for reading menus, signs, books

### Advanced Features 📊
- **Debug Logging**: Automatic saving of frames and detection data for failure analysis
- **Performance Metrics**: FPS tracking, inference time logging
- **Robustness Testing**: Simulate noise (Gaussian, motion blur, brightness)
- **Bias Testing**: Test model performance across different scenarios
- **Evaluation Tools**: Calculate mAP, IoU, F1-Score, Confusion Matrix

## 🛠️ Installation

### Prerequisites
- Python 3.9 or higher
- Webcam
- (Optional) CUDA-capable GPU for faster processing

### Step 1: Clone or Download
```bash
cd "IPCV A2"
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Install Ollama & LLaVA
1. Download Ollama from: https://ollama.ai
2. Install and start Ollama:
   ```bash
   ollama serve
   ```
3. Pull the LLaVA model:
   ```bash
   ollama pull llava
   ```

### Step 4: Download YOLOv8 Pre-trained Model
The pretrained model will auto-download on first run. For fine-tuned model, see DATASET_SETUP.md.

## 🚀 Quick Start

### Run the Main Application
```bash
python main.py
```

### Keyboard Controls
| Key | Action |
|-----|--------|
| **START Button** | Start/Stop camera |
| **SPACE** | Describe current scene (LLaVA) |
| **R** | Read text from image (OCR) |
| **D** | Toggle debug visualization |
| **A** | Toggle audio on/off |
| **L** | Toggle debug logging |
| **Q** | Quit application |

## 📁 Project Structure

```
IPCV A2/
├── main.py                      # Main GUI application
├── modules/
│   ├── navigator.py            # YOLOv8 real-time detection
│   ├── assistant.py            # LLaVA scene understanding
│   ├── reader.py               # OCR text reading
│   └── audio.py                # Text-to-speech
├── utils/
│   ├── config.py               # Configuration settings
│   ├── logger.py               # Debug & performance logging
│   └── distance_estimator.py  # Distance & position estimation
├── evaluate.py                 # Model evaluation script
├── bias_test.py                # Bias testing script
├── logs/                       # Auto-generated logs
│   ├── images/                 # Saved frames
│   ├── detections/             # Detection logs (JSON)
│   └── *.json/*.png            # Reports & visualizations
├── models/
│   └── yolov8_finetuned.pt    # Your fine-tuned model (after training)
├── test_images/
│   └── bias_test/              # Bias test images by category
├── requirements.txt
├── README.md
└── DATASET_SETUP.md            # Dataset download & training guide
```

## 📊 Evaluation & Testing

### 1. Run Model Evaluation
Calculate mAP, IoU, F1-Score, and generate visualizations:
```bash
python evaluate.py --images test_images --output logs
```

With ground truth labels (YOLO format):
```bash
python evaluate.py --images test_images --labels test_labels --output logs
```

Full validation with dataset.yaml:
```bash
python evaluate.py --data-yaml path/to/dataset.yaml
```

### 2. Run Bias Testing
Test model performance across different scenarios:
```bash
python bias_test.py --bias-dir test_images/bias_test --output logs
```

**Organize bias test images in folders:**
```
test_images/bias_test/
├── western_cities/      # Images from Western cities
├── asian_cities/        # Images from Asian cities
├── dark_lighting/       # Low-light conditions
├── bright_lighting/     # Bright/overexposed conditions
└── diverse_people/      # People with diverse skin tones
```

## 🔧 Configuration

Edit `utils/config.py` to customize:
- Model paths
- Detection thresholds
- Camera settings
- Ollama API endpoint
- TTS settings
- UI colors
- Debug logging options

### Important Settings:
```python
# Enable/disable debug logging
SAVE_DEBUG_LOGS = True
LOG_EVERY_N_FRAMES = 50

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.5
HAZARD_AREA_THRESHOLD = 0.40  # Trigger warning if bbox > 40% of screen

# Ollama settings
OLLAMA_MODEL = "llava"
OLLAMA_BASE_URL = "http://localhost:11434"
```

## 🎓 For Assignment: Critical Evaluation

### Data Logging for Failure Analysis
Debug logging is enabled by default. Every 50th frame is automatically saved to `logs/images/` with corresponding detection data in `logs/detections/`.

**To analyze failures:**
1. Run the app and navigate in various conditions (dark, bright, motion blur)
2. Check `logs/` folder for saved frames and JSON detection logs
3. Identify failure cases (missed detections, false positives)
4. Use these for your report's failure analysis section

### Performance Metrics
After running the app, check `logs/performance_report.json` for:
- Average FPS
- Inference times (YOLO, LLaVA, OCR)
- Processing latency

### Robustness Testing
The Navigator module includes noise simulation:
```python
from modules.navigator import Navigator

navigator = Navigator()
noisy_frame = navigator.apply_noise(frame, noise_type='gaussian')
# Test with 'gaussian', 'motion_blur', or 'brightness'
```

## 📈 Expected Outputs for Report

### 1. Evaluation Metrics
- `logs/evaluation_report.json` - Detailed metrics
- `logs/iou_distribution.png` - IoU histogram
- `logs/performance_metrics.png` - Precision/Recall/F1 bar chart
- `logs/confusion_matrix.png` - Confusion matrix heatmap

### 2. Bias Testing Results
- `logs/bias_test_results.csv` - Per-image confidence scores
- `logs/bias_confidence_comparison.png` - Confidence by category
- `logs/bias_detection_count.png` - Detection count by category
- `logs/bias_confidence_range.png` - Min/avg/max confidence range

### 3. Debug Logs
- `logs/images/*.jpg` - Saved frames showing failures
- `logs/detections/*.json` - Detection data with confidence scores

## 🐛 Troubleshooting

### Camera not opening
- Check camera permissions
- Try changing `CAMERA_INDEX` in `utils/config.py` (0, 1, 2...)

### Ollama connection error
```bash
# Start Ollama server
ollama serve

# Verify model is installed
ollama list

# Pull LLaVA if missing
ollama pull llava
```

### Low FPS / Slow performance
- Set `use_finetuned=False` in Navigator if using CPU
- Reduce camera resolution in `utils/config.py`
- Use smaller YOLOv8 model (yolov8n.pt is fastest)

### EasyOCR slow on first run
- EasyOCR downloads models on first use (~100MB)
- Subsequent runs will be faster

## 📚 Fine-tuning Guide

See **DATASET_SETUP.md** for:
- How to download hazard datasets (Roboflow)
- Dataset organization
- YOLOv8 training commands
- Transfer learning from COCO

## 🤝 Credits

- **YOLOv8**: Ultralytics
- **LLaVA**: Visual Instruction Tuning
- **Ollama**: Local LLM runtime
- **EasyOCR**: Jaided AI

## 📄 License

This project is for educational purposes (University Assignment).

## 👨‍💻 Author

University of Wollongong - Image Processing & Computer Vision Assignment

---

## 🌐 Web Demo (Streamlit)

### Try the Live Demo
🔗 **[VisionGuard Web App](https://your-app-url.streamlit.app)** _(Coming soon)_

### Run Locally
```bash
# Install Streamlit dependencies
pip install -r streamlit_requirements.txt

# Run the web app
streamlit run streamlit_app.py
```

### Deploy to Streamlit Cloud

1. **Push to GitHub:**
```bash
git init
git add .
git commit -m "Initial commit: VisionGuard AI Vision Assistant"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/visionguard.git
git push -u origin main
```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository: `YOUR_USERNAME/visionguard`
   - Set main file path: `streamlit_app.py`
   - Set Python version: 3.11
   - Advanced settings → Dependencies: Use `streamlit_requirements.txt`
   - Click "Deploy"!

3. **Your app will be live at:** `https://YOUR_USERNAME-visionguard-streamlit-app-xxxxx.streamlit.app`

### Web App Features
- ✅ Upload images for analysis
- ✅ Object detection with YOLOv8
- ✅ Text extraction with OCR
- ✅ Interactive UI with confidence controls
- ✅ No installation required - runs in browser
- ⚠️ Note: LLaVA scene description disabled (requires local Ollama server)

---

**Need help?** Check the troubleshooting section or review `utils/config.py` for configuration options.
