"""
Quick test script to verify all components are working
"""
import sys
import importlib


def test_imports():
    """Test if all required packages are installed"""
    print("="*60)
    print("Testing Package Imports")
    print("="*60)
    
    packages = {
        'cv2': 'opencv-python',
        'torch': 'torch',
        'ultralytics': 'ultralytics',
        'easyocr': 'easyocr',
        'pyttsx3': 'pyttsx3',
        'customtkinter': 'customtkinter',
        'PIL': 'Pillow',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'sklearn': 'scikit-learn',
        'requests': 'requests'
    }
    
    all_good = True
    
    for package, pip_name in packages.items():
        try:
            importlib.import_module(package)
            print(f"✓ {pip_name}")
        except ImportError:
            print(f"✗ {pip_name} - NOT INSTALLED")
            all_good = False
    
    print("="*60)
    
    if not all_good:
        print("\n[ERROR] Some packages are missing!")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✓ All packages installed successfully!")
    return True


def test_camera():
    """Test if camera is accessible"""
    print("\n" + "="*60)
    print("Testing Camera Access")
    print("="*60)
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ Camera could not be opened")
            print("  - Check if camera is connected")
            print("  - Check camera permissions")
            print("  - Try changing CAMERA_INDEX in utils/config.py")
            cap.release()
            return False
        
        ret, frame = cap.read()
        if not ret:
            print("✗ Could not read frame from camera")
            cap.release()
            return False
        
        print(f"✓ Camera is working!")
        print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
        cap.release()
        return True
        
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False


def test_yolo():
    """Test if YOLOv8 can be loaded"""
    print("\n" + "="*60)
    print("Testing YOLOv8 Model")
    print("="*60)
    
    try:
        from ultralytics import YOLO
        import torch
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {device}")
        
        if device == 'cpu':
            print("  [WARNING] No GPU detected. Performance will be slower.")
        
        print("Loading YOLOv8n model...")
        model = YOLO('yolov8n.pt')
        print("✓ YOLOv8 model loaded successfully!")
        print(f"  Model has {len(model.names)} classes")
        return True
        
    except Exception as e:
        print(f"✗ YOLOv8 test failed: {e}")
        return False


def test_ollama():
    """Test if Ollama is accessible"""
    print("\n" + "="*60)
    print("Testing Ollama Connection")
    print("="*60)
    
    try:
        import requests
        
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            print("✓ Ollama is running!")
            print(f"  Available models: {model_names}")
            
            if any('llava' in name for name in model_names):
                print("  ✓ LLaVA model is installed")
            else:
                print("  ✗ LLaVA model not found")
                print("  Run: ollama pull llava")
                return False
            
            return True
        else:
            print(f"✗ Ollama responded with status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to Ollama")
        print("  1. Install Ollama from https://ollama.ai")
        print("  2. Start Ollama: ollama serve")
        print("  3. Pull LLaVA: ollama pull llava")
        return False
    except Exception as e:
        print(f"✗ Ollama test failed: {e}")
        return False


def test_tts():
    """Test if text-to-speech is working"""
    print("\n" + "="*60)
    print("Testing Text-to-Speech")
    print("="*60)
    
    try:
        import pyttsx3
        
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        
        print(f"✓ TTS engine initialized!")
        print(f"  Available voices: {len(voices)}")
        
        # Test speech (without actually playing it)
        print("  Testing speech (silent)...")
        engine.say("Testing")
        # Don't run runAndWait to avoid audio during test
        
        print("✓ TTS is ready!")
        return True
        
    except Exception as e:
        print(f"✗ TTS test failed: {e}")
        return False


def test_modules():
    """Test if custom modules can be imported"""
    print("\n" + "="*60)
    print("Testing Custom Modules")
    print("="*60)
    
    modules_to_test = [
        'utils.config',
        'utils.logger',
        'utils.distance_estimator',
        'modules.audio',
        'modules.navigator',
        'modules.assistant',
        'modules.reader'
    ]
    
    all_good = True
    
    for module in modules_to_test:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except Exception as e:
            print(f"✗ {module} - {e}")
            all_good = False
    
    print("="*60)
    
    if not all_good:
        print("\n[ERROR] Some modules failed to import!")
        return False
    
    print("\n✓ All modules loaded successfully!")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print(" "*20 + "VisionGuard System Check")
    print("="*70 + "\n")
    
    results = {
        "Package Imports": test_imports(),
        "Custom Modules": test_modules(),
        "Camera": test_camera(),
        "YOLOv8": test_yolo(),
        "Ollama/LLaVA": test_ollama(),
        "Text-to-Speech": test_tts()
    }
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<50} {status}")
    
    print("="*70)
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All tests passed! You're ready to run VisionGuard!")
        print("\nRun the application with: python main.py")
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install packages: pip install -r requirements.txt")
        print("  - Start Ollama: ollama serve")
        print("  - Pull LLaVA: ollama pull llava")
        print("  - Check camera connections and permissions")
    
    print()


if __name__ == "__main__":
    main()
