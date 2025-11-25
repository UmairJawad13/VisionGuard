"""
Debug logging system for failure analysis
"""

import os
import cv2
import json
from datetime import datetime
from utils.config import IMAGES_LOG_DIR, DETECTIONS_LOG_DIR, LOG_EVERY_N_FRAMES, SAVE_DEBUG_LOGS, MAX_LOG_FILES


class DebugLogger:
    """Logs frames and detection data for failure analysis"""
    
    def __init__(self):
        self.frame_count = 0
        self.log_count = 0
        self.enabled = SAVE_DEBUG_LOGS
        
        # Ensure log directories exist
        os.makedirs(IMAGES_LOG_DIR, exist_ok=True)
        os.makedirs(DETECTIONS_LOG_DIR, exist_ok=True)
        
        # Clean old logs if exceeding max
        self._clean_old_logs()
    
    def _clean_old_logs(self):
        """Remove oldest log files if exceeding MAX_LOG_FILES"""
        for log_dir in [IMAGES_LOG_DIR, DETECTIONS_LOG_DIR]:
            files = sorted([
                os.path.join(log_dir, f) for f in os.listdir(log_dir)
                if os.path.isfile(os.path.join(log_dir, f))
            ], key=os.path.getmtime)
            
            while len(files) > MAX_LOG_FILES:
                os.remove(files.pop(0))
    
    def log_frame(self, frame, detections, metadata=None):
        """
        Log a frame with its detection data
        
        Args:
            frame: The image frame (numpy array)
            detections: List of detection results from YOLO
            metadata: Additional metadata (lighting conditions, blur level, etc.)
        """
        if not self.enabled:
            return
        
        self.frame_count += 1
        
        # Only log every Nth frame
        if self.frame_count % LOG_EVERY_N_FRAMES != 0:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_id = f"{timestamp}_{self.log_count:04d}"
        
        # Save image
        image_path = os.path.join(IMAGES_LOG_DIR, f"{log_id}.jpg")
        cv2.imwrite(image_path, frame)
        
        # Prepare detection data
        detection_data = {
            "log_id": log_id,
            "timestamp": timestamp,
            "frame_number": self.frame_count,
            "image_path": image_path,
            "detections": [],
            "metadata": metadata or {}
        }
        
        # Extract detection information
        for detection in detections:
            det_info = {
                "class_id": int(detection.get('class_id', -1)),
                "class_name": detection.get('class_name', 'unknown'),
                "confidence": float(detection.get('confidence', 0.0)),
                "bbox": detection.get('bbox', []),  # [x1, y1, x2, y2]
                "bbox_area_percent": detection.get('bbox_area_percent', 0.0),
                "position": detection.get('position', 'center')
            }
            detection_data["detections"].append(det_info)
        
        # Calculate statistics
        detection_data["statistics"] = {
            "total_detections": len(detections),
            "avg_confidence": sum([d['confidence'] for d in detection_data["detections"]]) / len(detections) if detections else 0,
            "max_confidence": max([d['confidence'] for d in detection_data["detections"]]) if detections else 0,
            "min_confidence": min([d['confidence'] for d in detection_data["detections"]]) if detections else 0,
        }
        
        # Save detection log as JSON
        json_path = os.path.join(DETECTIONS_LOG_DIR, f"{log_id}.json")
        with open(json_path, 'w') as f:
            json.dump(detection_data, f, indent=2)
        
        self.log_count += 1
        print(f"[DEBUG LOG] Saved frame {self.frame_count} with {len(detections)} detections")
    
    def enable(self):
        """Enable debug logging"""
        self.enabled = True
        print("[DEBUG] Logging enabled")
    
    def disable(self):
        """Disable debug logging"""
        self.enabled = False
        print("[DEBUG] Logging disabled")
    
    def toggle(self):
        """Toggle debug logging"""
        self.enabled = not self.enabled
        status = "enabled" if self.enabled else "disabled"
        print(f"[DEBUG] Logging {status}")
        return self.enabled
    
    def get_summary(self):
        """Get logging summary statistics"""
        return {
            "total_frames_processed": self.frame_count,
            "frames_logged": self.log_count,
            "enabled": self.enabled
        }


class PerformanceLogger:
    """Logs performance metrics (FPS, latency, etc.)"""
    
    def __init__(self):
        self.metrics = {
            "yolo_inference_times": [],
            "llm_inference_times": [],
            "ocr_inference_times": [],
            "frame_processing_times": []
        }
    
    def log_inference_time(self, module, elapsed_time):
        """Log inference time for a specific module"""
        key = f"{module}_inference_times"
        if key in self.metrics:
            self.metrics[key].append(elapsed_time)
    
    def get_average_fps(self):
        """Calculate average FPS from frame processing times"""
        if not self.metrics["frame_processing_times"]:
            return 0
        avg_time = sum(self.metrics["frame_processing_times"]) / len(self.metrics["frame_processing_times"])
        return 1.0 / avg_time if avg_time > 0 else 0
    
    def get_summary(self):
        """Get performance summary"""
        summary = {}
        for key, values in self.metrics.items():
            if values:
                summary[key] = {
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
        return summary
    
    def save_report(self, filepath):
        """Save performance report to JSON"""
        summary = self.get_summary()
        summary["average_fps"] = self.get_average_fps()
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[PERFORMANCE] Report saved to {filepath}")
