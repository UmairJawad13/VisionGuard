"""
Configuration file for VisionGuard Application
"""

import os

# ========================
# GENERAL SETTINGS
# ========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
IMAGES_LOG_DIR = os.path.join(LOGS_DIR, "images")
DETECTIONS_LOG_DIR = os.path.join(LOGS_DIR, "detections")
MODELS_DIR = os.path.join(BASE_DIR, "models")
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "test_images")

# ========================
# YOLO SETTINGS
# ========================
# Model paths
YOLO_PRETRAINED = "yolov8n.pt"  # Pre-trained COCO model
YOLO_FINETUNED = os.path.join(MODELS_DIR, "yolov8_finetuned.pt")  # Fine-tuned model

# Target classes for safety detection (COCO dataset indices)
SAFETY_CLASSES = {
    0: "person",
    2: "car",
    56: "chair",
    # Add custom classes after fine-tuning:
    # 80: "stairs",
    # 81: "pothole",
    # 82: "door",
    # 83: "stop_sign"
}

# Detection settings
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
HAZARD_AREA_THRESHOLD = 0.40  # If bbox covers >40% of screen, trigger warning

# ========================
# CAMERA SETTINGS
# ========================
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FPS = 30

# ========================
# DEBUG & LOGGING SETTINGS
# ========================
SAVE_DEBUG_LOGS = True  # Toggle to save frames and detection logs
LOG_EVERY_N_FRAMES = 50  # Save every 50th frame when debug is enabled
MAX_LOG_FILES = 100  # Maximum number of log files to keep

# ========================
# LLM SETTINGS (Ollama + LLaVA)
# ========================
OLLAMA_MODEL = "llava"  # Model name in Ollama
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama API endpoint
VQA_PROMPT = """You are assisting a visually impaired person. Describe this scene in detail, 
focusing on:
1. Any potential hazards or obstacles (people, vehicles, furniture, stairs)
2. The general environment and atmosphere
3. Navigation suggestions (safe paths, things to avoid)
Keep it concise but informative."""

# ========================
# OCR SETTINGS
# ========================
OCR_ENGINE = "easyocr"  # Options: "easyocr" or "pytesseract"
OCR_LANGUAGES = ['en']  # Languages for text detection
MIN_TEXT_CONFIDENCE = 0.3

# ========================
# TEXT-TO-SPEECH SETTINGS
# ========================
TTS_ENGINE = "pyttsx3"  # Offline TTS
TTS_RATE = 150  # Speaking rate (words per minute)
TTS_VOLUME = 0.9  # Volume (0.0 to 1.0)
TTS_VOICE_INDEX = 0  # Voice index (0 = default, try 1 for female voice)

# ========================
# UI SETTINGS
# ========================
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
UI_BG_COLOR = "#000000"  # Black background
UI_FG_COLOR = "#FFFF00"  # Yellow text (high contrast)
UI_BUTTON_COLOR = "#333333"
UI_BUTTON_HOVER = "#555555"

# ========================
# POSITION DETECTION
# ========================
# Screen divided into 3 zones: Left (0-0.33), Center (0.33-0.66), Right (0.66-1.0)
LEFT_ZONE = 0.33
RIGHT_ZONE = 0.66

# ========================
# EVALUATION SETTINGS
# ========================
EVAL_IOU_THRESHOLD = 0.5
EVAL_CONF_THRESHOLD = 0.25
EVAL_CLASSES = list(SAFETY_CLASSES.keys())

# ========================
# ROBUSTNESS TESTING
# ========================
NOISE_LEVELS = {
    "gaussian": (0, 25),  # Mean and standard deviation
    "motion_blur": 15,     # Kernel size
    "brightness": 0.5      # Brightness factor
}
