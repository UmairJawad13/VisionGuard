"""
Real-time navigation module using YOLOv8 for object detection
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
from utils.config import (
    YOLO_PRETRAINED, YOLO_FINETUNED, SAFETY_CLASSES, 
    CONFIDENCE_THRESHOLD, IOU_THRESHOLD, CAMERA_WIDTH, CAMERA_HEIGHT
)
from utils.distance_estimator import DistanceEstimator
from utils.logger import DebugLogger, PerformanceLogger
import os


class Navigator:
    """Real-time object detection and navigation assistance using YOLOv8"""
    
    def __init__(self, use_finetuned=False):
        """
        Initialize Navigator
        
        Args:
            use_finetuned: If True and fine-tuned model exists, use it. Otherwise use pretrained.
        """
        self.use_finetuned = use_finetuned
        self.model = None
        self.distance_estimator = DistanceEstimator(CAMERA_WIDTH, CAMERA_HEIGHT)
        self.debug_logger = DebugLogger()
        self.perf_logger = PerformanceLogger()
        
        # Detection settings
        self.conf_threshold = CONFIDENCE_THRESHOLD
        self.iou_threshold = IOU_THRESHOLD
        
        # Load model
        self._load_model()
        
        # Statistics
        self.total_frames = 0
        self.total_detections = 0
        
    def _load_model(self):
        """Load YOLOv8 model (pretrained or fine-tuned)"""
        try:
            # Check if fine-tuned model exists and is requested
            if self.use_finetuned and os.path.exists(YOLO_FINETUNED):
                model_path = YOLO_FINETUNED
                print(f"[NAVIGATOR] Loading fine-tuned model from {model_path}")
            else:
                model_path = YOLO_PRETRAINED
                print(f"[NAVIGATOR] Loading pretrained model: {model_path}")
            
            self.model = YOLO(model_path)
            
            # Check if CUDA is available
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"[NAVIGATOR] Using device: {device}")
            
            if device == 'cpu':
                print("[NAVIGATOR WARNING] Running on CPU. Performance may be slower.")
            
            print(f"[NAVIGATOR] Model loaded successfully")
            print(f"[NAVIGATOR] Model classes: {len(self.model.names)}")
            
        except Exception as e:
            print(f"[NAVIGATOR ERROR] Failed to load model: {e}")
            raise
    
    def process_frame(self, frame, show_debug=False):
        """
        Process a single frame and detect objects
        
        Args:
            frame: Input image frame (numpy array)
            show_debug: If True, draw bounding boxes and labels on frame
        
        Returns:
            tuple: (processed_frame, detections, analysis)
                - processed_frame: Frame with annotations if show_debug=True
                - detections: List of detection dictionaries
                - analysis: Analysis results from distance_estimator
        """
        start_time = time.time()
        
        # Run YOLO inference
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )[0]
        
        inference_time = time.time() - start_time
        self.perf_logger.log_inference_time("yolo", inference_time)
        
        # Parse detections
        detections = []
        for box in results.boxes:
            # Extract box information
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            
            # Get class name
            class_name = self.model.names[class_id]
            
            # Filter for safety classes (if using pretrained COCO model)
            # if not self.use_finetuned and class_id not in SAFETY_CLASSES:
            #     continue
            
            detection = {
                'class_id': class_id,
                'class_name': class_name,
                'confidence': confidence,
                'bbox': [int(x1), int(y1), int(x2), int(y2)]
            }
            
            detections.append(detection)
        
        # Analyze detections for distance and position
        analysis = self.distance_estimator.analyze_frame(detections)
        
        # Draw on frame if debug mode
        if show_debug:
            frame = self._draw_detections(frame, detections)
            
            # Draw FPS
            fps = self.perf_logger.get_average_fps()
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw detection count
            cv2.putText(frame, f"Objects: {len(detections)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Log frame if debug logging is enabled
        if self.debug_logger.enabled:
            metadata = {
                "inference_time": inference_time,
                "fps": self.perf_logger.get_average_fps(),
                "show_debug": show_debug
            }
            self.debug_logger.log_frame(frame, detections, metadata)
        
        # Update statistics
        self.total_frames += 1
        self.total_detections += len(detections)
        self.perf_logger.log_inference_time("frame_processing", time.time() - start_time)
        
        return frame, detections, analysis
    
    def _draw_detections(self, frame, detections):
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input image
            detections: List of detection dictionaries
        
        Returns:
            Annotated frame
        """
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Determine color based on distance
            distance_info = det.get('distance_info', {})
            if distance_info.get('is_hazard', False):
                color = (0, 0, 255)  # Red for hazards
            else:
                color = (0, 255, 0)  # Green for safe
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            position = det.get('position', '')
            distance = distance_info.get('distance_level', '')
            label = f"{class_name} {confidence:.2f}"
            
            if distance:
                label += f" ({distance})"
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw position indicator
            if position:
                cv2.putText(frame, position.upper(), (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def apply_noise(self, frame, noise_type='gaussian'):
        """
        Apply noise to frame for robustness testing
        
        Args:
            frame: Input frame
            noise_type: Type of noise ('gaussian', 'motion_blur', 'brightness')
        
        Returns:
            Noisy frame
        """
        from utils.config import NOISE_LEVELS
        
        if noise_type == 'gaussian':
            mean, std = NOISE_LEVELS['gaussian']
            noise = np.random.normal(mean, std, frame.shape).astype(np.uint8)
            noisy_frame = cv2.add(frame, noise)
            
        elif noise_type == 'motion_blur':
            kernel_size = NOISE_LEVELS['motion_blur']
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
            kernel = kernel / kernel_size
            noisy_frame = cv2.filter2D(frame, -1, kernel)
            
        elif noise_type == 'brightness':
            factor = NOISE_LEVELS['brightness']
            noisy_frame = cv2.convertScaleAbs(frame, alpha=factor, beta=0)
        
        else:
            noisy_frame = frame
        
        return noisy_frame
    
    def get_statistics(self):
        """Get detection statistics"""
        return {
            'total_frames': self.total_frames,
            'total_detections': self.total_detections,
            'avg_detections_per_frame': self.total_detections / self.total_frames if self.total_frames > 0 else 0,
            'performance': self.perf_logger.get_summary()
        }
    
    def save_performance_report(self, filepath):
        """Save performance report to file"""
        self.perf_logger.save_report(filepath)
