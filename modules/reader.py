"""
OCR module for reading text from images using EasyOCR
"""

import cv2
import numpy as np
import time
from utils.config import OCR_ENGINE, OCR_LANGUAGES, MIN_TEXT_CONFIDENCE
from utils.logger import PerformanceLogger


class Reader:
    """Text reading module using OCR"""
    
    def __init__(self):
        self.ocr_engine = OCR_ENGINE
        self.languages = OCR_LANGUAGES
        self.min_confidence = MIN_TEXT_CONFIDENCE
        self.reader = None
        self.perf_logger = PerformanceLogger()
        
        # Initialize OCR engine
        self._initialize_ocr()
    
    def _initialize_ocr(self):
        """Initialize the OCR engine"""
        try:
            if self.ocr_engine == "easyocr":
                import easyocr
                print(f"[READER] Initializing EasyOCR with languages: {self.languages}")
                print("[READER] This may take a moment on first run (downloading models)...")
                self.reader = easyocr.Reader(self.languages, gpu=False)  # Set gpu=True if you have CUDA
                print("[READER] EasyOCR initialized successfully")
                
            elif self.ocr_engine == "pytesseract":
                import pytesseract
                self.reader = pytesseract
                print("[READER] Using Pytesseract")
                # Note: pytesseract requires Tesseract to be installed separately
                
            else:
                raise ValueError(f"Unknown OCR engine: {self.ocr_engine}")
                
        except ImportError as e:
            print(f"[READER ERROR] Failed to import OCR library: {e}")
            print(f"[READER] Please install: pip install {self.ocr_engine}")
            self.reader = None
        except Exception as e:
            print(f"[READER ERROR] Failed to initialize OCR: {e}")
            self.reader = None
    
    def read_text(self, frame, show_debug=False):
        """
        Detect and read text from frame
        
        Args:
            frame: Input image frame (numpy array)
            show_debug: If True, draw bounding boxes around detected text
        
        Returns:
            tuple: (annotated_frame, detected_texts)
                - annotated_frame: Frame with text boxes if show_debug=True
                - detected_texts: List of dictionaries with text and confidence
        """
        if self.reader is None:
            return frame, [{"text": "OCR not available", "confidence": 0.0}]
        
        start_time = time.time()
        detected_texts = []
        
        try:
            if self.ocr_engine == "easyocr":
                # EasyOCR returns: (bbox, text, confidence)
                results = self.reader.readtext(frame)
                
                for (bbox, text, confidence) in results:
                    if confidence >= self.min_confidence:
                        detected_texts.append({
                            'text': text,
                            'confidence': confidence,
                            'bbox': bbox
                        })
                        
                        # Draw on frame if debug mode
                        if show_debug:
                            # Convert bbox to integer points
                            points = np.array(bbox, dtype=np.int32)
                            cv2.polylines(frame, [points], True, (0, 255, 0), 2)
                            
                            # Draw text label
                            x, y = int(bbox[0][0]), int(bbox[0][1])
                            label = f"{text} ({confidence:.2f})"
                            cv2.putText(frame, label, (x, y - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            elif self.ocr_engine == "pytesseract":
                # Pytesseract approach
                import pytesseract
                
                # Get detailed text data
                data = pytesseract.image_to_data(frame, output_type=pytesseract.Output.DICT)
                
                for i in range(len(data['text'])):
                    text = data['text'][i].strip()
                    confidence = float(data['conf'][i]) / 100.0  # Convert to 0-1 range
                    
                    if text and confidence >= self.min_confidence:
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        
                        detected_texts.append({
                            'text': text,
                            'confidence': confidence,
                            'bbox': [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                        })
                        
                        # Draw on frame if debug mode
                        if show_debug:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            label = f"{text} ({confidence:.2f})"
                            cv2.putText(frame, label, (x, y - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Log inference time
            inference_time = time.time() - start_time
            self.perf_logger.log_inference_time("ocr", inference_time)
            
            print(f"[READER] Detected {len(detected_texts)} text regions in {inference_time:.2f}s")
            
        except Exception as e:
            print(f"[READER ERROR] OCR failed: {e}")
            detected_texts = [{"text": f"Error: {str(e)}", "confidence": 0.0}]
        
        return frame, detected_texts
    
    def format_text_for_speech(self, detected_texts):
        """
        Format detected text into a natural speech string
        
        Args:
            detected_texts: List of text detection dictionaries
        
        Returns:
            str: Formatted text for TTS
        """
        if not detected_texts:
            return "No text detected in the image."
        
        # Sort by vertical position (top to bottom)
        # This helps read text in a more natural order
        sorted_texts = sorted(detected_texts, key=lambda x: x['bbox'][0][1] if 'bbox' in x else 0)
        
        # Extract just the text
        text_lines = [item['text'] for item in sorted_texts]
        
        if not text_lines:
            return "No readable text found."
        
        # Join text with pauses
        full_text = ". ".join(text_lines)
        
        # Return just the text, no extra context to make it faster
        return full_text
    
    def preprocess_for_ocr(self, frame):
        """
        Preprocess image to improve OCR accuracy
        
        Args:
            frame: Input image
        
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to get black text on white background
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh)
        
        # Optional: upscale for better OCR
        # upscaled = cv2.resize(denoised, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        return denoised
    
    def is_available(self):
        """Check if OCR is available"""
        return self.reader is not None
