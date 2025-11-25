"""
Distance estimation and position detection utilities
"""

import numpy as np
from utils.config import HAZARD_AREA_THRESHOLD, LEFT_ZONE, RIGHT_ZONE


class DistanceEstimator:
    """Simple distance and position estimation based on bounding box size"""
    
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_area = frame_width * frame_height
    
    def estimate_distance(self, bbox):
        """
        Estimate relative distance based on bounding box area
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box coordinates
        
        Returns:
            dict: {
                'distance_level': str ('very_close', 'close', 'medium', 'far'),
                'bbox_area_percent': float (percentage of frame covered),
                'is_hazard': bool (True if object is too close)
            }
        """
        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_area = bbox_width * bbox_height
        
        # Calculate percentage of frame covered
        area_percent = bbox_area / self.frame_area
        
        # Determine distance level
        if area_percent > 0.40:
            distance_level = "very_close"
            is_hazard = True
        elif area_percent > 0.25:
            distance_level = "close"
            is_hazard = True
        elif area_percent > 0.10:
            distance_level = "medium"
            is_hazard = False
        else:
            distance_level = "far"
            is_hazard = False
        
        return {
            'distance_level': distance_level,
            'bbox_area_percent': area_percent,
            'is_hazard': is_hazard
        }
    
    def get_position(self, bbox):
        """
        Determine object position relative to user (left, center, right)
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box coordinates
        
        Returns:
            str: 'left', 'center', or 'right'
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        
        # Normalize to 0-1 range
        normalized_x = center_x / self.frame_width
        
        if normalized_x < LEFT_ZONE:
            return "left"
        elif normalized_x > RIGHT_ZONE:
            return "right"
        else:
            return "center"
    
    def get_vertical_position(self, bbox):
        """
        Determine vertical position (top, middle, bottom)
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box coordinates
        
        Returns:
            str: 'top', 'middle', or 'bottom'
        """
        x1, y1, x2, y2 = bbox
        center_y = (y1 + y2) / 2
        
        # Normalize to 0-1 range
        normalized_y = center_y / self.frame_height
        
        if normalized_y < 0.33:
            return "top"
        elif normalized_y > 0.66:
            return "bottom"
        else:
            return "middle"
    
    def generate_warning_message(self, class_name, distance_info, position):
        """
        Generate a spoken warning message
        
        Args:
            class_name: Name of detected object
            distance_info: Dictionary from estimate_distance()
            position: String from get_position()
        
        Returns:
            str: Warning message to be spoken
        """
        distance_level = distance_info['distance_level']
        
        # Build message based on distance and position
        if distance_info['is_hazard']:
            if distance_level == "very_close":
                urgency = "STOP! "
            else:
                urgency = "Warning! "
            
            if position == "center":
                return f"{urgency}{class_name} directly in front of you!"
            else:
                return f"{urgency}{class_name} on your {position}!"
        else:
            # Non-hazardous announcements (informational)
            return f"{class_name} detected on your {position}"
    
    def analyze_frame(self, detections):
        """
        Analyze all detections in a frame and prioritize warnings
        
        Args:
            detections: List of detection dictionaries with bbox, class_name, etc.
        
        Returns:
            dict: {
                'priority_warning': str or None,
                'all_warnings': list of str,
                'hazard_count': int
            }
        """
        warnings = []
        hazards = []
        
        for det in detections:
            bbox = det.get('bbox', [])
            class_name = det.get('class_name', 'object')
            
            if not bbox:
                continue
            
            distance_info = self.estimate_distance(bbox)
            position = self.get_position(bbox)
            
            # Add distance and position to detection
            det['distance_info'] = distance_info
            det['position'] = position
            
            warning_msg = self.generate_warning_message(class_name, distance_info, position)
            warnings.append(warning_msg)
            
            if distance_info['is_hazard']:
                hazards.append({
                    'message': warning_msg,
                    'distance': distance_info['bbox_area_percent'],
                    'class': class_name
                })
        
        # Prioritize closest hazard
        priority_warning = None
        if hazards:
            # Sort by distance (area percentage - larger = closer)
            hazards.sort(key=lambda x: x['distance'], reverse=True)
            priority_warning = hazards[0]['message']
        
        return {
            'priority_warning': priority_warning,
            'all_warnings': warnings,
            'hazard_count': len(hazards),
            'total_objects': len(detections)
        }
