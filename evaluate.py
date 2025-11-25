"""
Evaluation script for calculating model performance metrics
Generates mAP, IoU, F1-Score, Confusion Matrix, and visualizations
"""

import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from collections import defaultdict
import pandas as pd

from utils.config import (
    YOLO_PRETRAINED, YOLO_FINETUNED, EVAL_IOU_THRESHOLD,
    EVAL_CONF_THRESHOLD, TEST_IMAGES_DIR, MODELS_DIR
)


class ModelEvaluator:
    """Comprehensive model evaluation and metrics calculation"""
    
    def __init__(self, model_path=None, ground_truth_path=None):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to YOLO model (uses finetuned if exists, else pretrained)
            ground_truth_path: Path to ground truth annotations (YOLO format)
        """
        # Load model
        if model_path and os.path.exists(model_path):
            self.model_path = model_path
        elif os.path.exists(YOLO_FINETUNED):
            self.model_path = YOLO_FINETUNED
        else:
            self.model_path = YOLO_PRETRAINED
        
        print(f"[EVAL] Loading model: {self.model_path}")
        self.model = YOLO(self.model_path)
        
        self.ground_truth_path = ground_truth_path
        self.results = {
            'predictions': [],
            'ground_truths': [],
            'metrics': {}
        }
    
    def calculate_iou(self, box1, box2):
        """
        Calculate IoU (Intersection over Union) between two bounding boxes
        
        Args:
            box1, box2: [x1, y1, x2, y2] format
        
        Returns:
            float: IoU score
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection area
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        inter_width = max(0, inter_x_max - inter_x_min)
        inter_height = max(0, inter_y_max - inter_y_min)
        inter_area = inter_width * inter_height
        
        # Calculate union area
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        iou = inter_area / union_area if union_area > 0 else 0
        return iou
    
    def evaluate_on_dataset(self, images_dir, labels_dir=None):
        """
        Evaluate model on a dataset
        
        Args:
            images_dir: Directory containing test images
            labels_dir: Directory containing ground truth labels (YOLO format)
        
        Returns:
            dict: Evaluation metrics
        """
        print(f"[EVAL] Evaluating on dataset: {images_dir}")
        
        image_files = list(Path(images_dir).glob("*.jpg")) + \
                     list(Path(images_dir).glob("*.png")) + \
                     list(Path(images_dir).glob("*.jpeg"))
        
        if not image_files:
            print("[EVAL WARNING] No images found in directory")
            return {}
        
        all_predictions = []
        all_ground_truths = []
        iou_scores = []
        
        for img_path in image_files:
            print(f"[EVAL] Processing: {img_path.name}")
            
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # Run inference
            results = self.model(image, conf=EVAL_CONF_THRESHOLD, verbose=False)[0]
            
            # Extract predictions
            predictions = []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                predictions.append({
                    'bbox': [x1, y1, x2, y2],
                    'class_id': class_id,
                    'confidence': confidence
                })
            
            # Load ground truth if available
            ground_truths = []
            if labels_dir:
                label_path = Path(labels_dir) / f"{img_path.stem}.txt"
                if label_path.exists():
                    ground_truths = self._load_yolo_labels(label_path, image.shape)
            
            # Calculate IoU for matching predictions
            if predictions and ground_truths:
                for pred in predictions:
                    best_iou = 0
                    for gt in ground_truths:
                        if pred['class_id'] == gt['class_id']:
                            iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                            best_iou = max(best_iou, iou)
                    iou_scores.append(best_iou)
            
            all_predictions.append(predictions)
            all_ground_truths.append(ground_truths)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_predictions, all_ground_truths, iou_scores)
        
        self.results['predictions'] = all_predictions
        self.results['ground_truths'] = all_ground_truths
        self.results['metrics'] = metrics
        
        return metrics
    
    def _load_yolo_labels(self, label_path, image_shape):
        """
        Load YOLO format labels
        
        Args:
            label_path: Path to label file
            image_shape: (height, width, channels)
        
        Returns:
            List of ground truth dictionaries
        """
        height, width = image_shape[:2]
        ground_truths = []
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                class_id = int(parts[0])
                x_center = float(parts[1]) * width
                y_center = float(parts[2]) * height
                box_width = float(parts[3]) * width
                box_height = float(parts[4]) * height
                
                # Convert to [x1, y1, x2, y2]
                x1 = x_center - box_width / 2
                y1 = y_center - box_height / 2
                x2 = x_center + box_width / 2
                y2 = y_center + box_height / 2
                
                ground_truths.append({
                    'bbox': [x1, y1, x2, y2],
                    'class_id': class_id
                })
        
        return ground_truths
    
    def _calculate_metrics(self, predictions, ground_truths, iou_scores):
        """Calculate comprehensive metrics"""
        metrics = {}
        
        # IoU statistics
        if iou_scores:
            metrics['mean_iou'] = np.mean(iou_scores)
            metrics['median_iou'] = np.median(iou_scores)
            metrics['iou_scores'] = iou_scores
        else:
            metrics['mean_iou'] = 0
            metrics['median_iou'] = 0
            metrics['iou_scores'] = []
        
        # Count predictions and ground truths
        total_predictions = sum(len(p) for p in predictions)
        total_ground_truths = sum(len(g) for g in ground_truths)
        
        metrics['total_predictions'] = total_predictions
        metrics['total_ground_truths'] = total_ground_truths
        
        # If ground truths available, calculate precision/recall
        if total_ground_truths > 0:
            # True positives: predictions with IoU > threshold
            tp = sum(1 for iou in iou_scores if iou > EVAL_IOU_THRESHOLD)
            fp = total_predictions - tp
            fn = total_ground_truths - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1_score'] = f1
            metrics['true_positives'] = tp
            metrics['false_positives'] = fp
            metrics['false_negatives'] = fn
        
        return metrics
    
    def run_ultralytics_val(self, data_yaml_path):
        """
        Run Ultralytics built-in validation (most accurate for mAP)
        
        Args:
            data_yaml_path: Path to dataset.yaml file
        
        Returns:
            Validation results
        """
        print(f"[EVAL] Running Ultralytics validation on: {data_yaml_path}")
        
        try:
            results = self.model.val(data=data_yaml_path, imgsz=640, conf=EVAL_CONF_THRESHOLD)
            
            # Extract key metrics
            metrics = {
                'mAP50': float(results.box.map50),  # mAP at IoU=0.5
                'mAP50-95': float(results.box.map),  # mAP at IoU=0.5:0.95
                'precision': float(results.box.mp),
                'recall': float(results.box.mr)
            }
            
            print(f"[EVAL] mAP@50: {metrics['mAP50']:.4f}")
            print(f"[EVAL] mAP@50-95: {metrics['mAP50-95']:.4f}")
            
            self.results['metrics'].update(metrics)
            return metrics
            
        except Exception as e:
            print(f"[EVAL ERROR] Ultralytics validation failed: {e}")
            return {}
    
    def generate_confusion_matrix(self, predictions, ground_truths, class_names):
        """
        Generate confusion matrix
        
        Args:
            predictions: List of prediction lists
            ground_truths: List of ground truth lists
            class_names: List of class names
        
        Returns:
            Confusion matrix array
        """
        y_true = []
        y_pred = []
        
        for preds, gts in zip(predictions, ground_truths):
            if not gts:
                continue
            
            for gt in gts:
                gt_class = gt['class_id']
                
                # Find best matching prediction
                best_match = None
                best_iou = 0
                
                for pred in preds:
                    if pred['class_id'] == gt_class:
                        iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                        if iou > best_iou:
                            best_iou = iou
                            best_match = pred
                
                if best_match and best_iou > EVAL_IOU_THRESHOLD:
                    y_true.append(gt_class)
                    y_pred.append(best_match['class_id'])
                else:
                    # Missed detection
                    y_true.append(gt_class)
                    y_pred.append(-1)  # No prediction
        
        if not y_true:
            print("[EVAL WARNING] No data for confusion matrix")
            return None
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        return cm
    
    def plot_confusion_matrix(self, cm, class_names, save_path='logs/confusion_matrix.png'):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[EVAL] Confusion matrix saved to {save_path}")
    
    def plot_metrics(self, save_dir='logs'):
        """Generate and save metric visualizations"""
        os.makedirs(save_dir, exist_ok=True)
        metrics = self.results['metrics']
        
        # 1. IoU distribution
        if 'iou_scores' in metrics and metrics['iou_scores']:
            plt.figure(figsize=(10, 6))
            plt.hist(metrics['iou_scores'], bins=20, edgecolor='black')
            plt.xlabel('IoU Score')
            plt.ylabel('Frequency')
            plt.title('Distribution of IoU Scores')
            plt.axvline(np.mean(metrics['iou_scores']), color='r', linestyle='--',
                       label=f"Mean: {np.mean(metrics['iou_scores']):.3f}")
            plt.legend()
            plt.savefig(os.path.join(save_dir, 'iou_distribution.png'), dpi=300)
            plt.close()
        
        # 2. Precision, Recall, F1 bar chart
        if 'precision' in metrics:
            fig, ax = plt.subplots(figsize=(8, 6))
            metrics_names = ['Precision', 'Recall', 'F1-Score']
            values = [metrics['precision'], metrics['recall'], metrics['f1_score']]
            colors = ['#2ecc71', '#3498db', '#e74c3c']
            
            bars = ax.bar(metrics_names, values, color=colors, edgecolor='black')
            ax.set_ylim(0, 1.0)
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Metrics')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'performance_metrics.png'), dpi=300)
            plt.close()
        
        print(f"[EVAL] Plots saved to {save_dir}")
    
    def save_report(self, filepath='logs/evaluation_report.json'):
        """Save detailed evaluation report"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        report = {
            'model_path': self.model_path,
            'metrics': self.results['metrics'],
            'settings': {
                'iou_threshold': EVAL_IOU_THRESHOLD,
                'conf_threshold': EVAL_CONF_THRESHOLD
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"[EVAL] Report saved to {filepath}")
    
    def print_summary(self):
        """Print evaluation summary"""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        
        metrics = self.results['metrics']
        
        print(f"\nModel: {self.model_path}")
        print(f"\nDetection Metrics:")
        print(f"  Total Predictions: {metrics.get('total_predictions', 0)}")
        print(f"  Total Ground Truths: {metrics.get('total_ground_truths', 0)}")
        
        if 'mean_iou' in metrics:
            print(f"\nIoU Metrics:")
            print(f"  Mean IoU: {metrics['mean_iou']:.4f}")
            print(f"  Median IoU: {metrics['median_iou']:.4f}")
        
        if 'mAP50' in metrics:
            print(f"\nmAP Metrics:")
            print(f"  mAP@50: {metrics['mAP50']:.4f}")
            print(f"  mAP@50-95: {metrics['mAP50-95']:.4f}")
        
        if 'precision' in metrics:
            print(f"\nClassification Metrics:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1_score']:.4f}")
            print(f"\n  True Positives: {metrics['true_positives']}")
            print(f"  False Positives: {metrics['false_positives']}")
            print(f"  False Negatives: {metrics['false_negatives']}")
        
        print("\n" + "=" * 60)


def main():
    """Main evaluation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate YOLO model performance')
    parser.add_argument('--model', type=str, help='Path to model file')
    parser.add_argument('--images', type=str, default=TEST_IMAGES_DIR,
                       help='Path to test images directory')
    parser.add_argument('--labels', type=str, help='Path to ground truth labels directory')
    parser.add_argument('--data-yaml', type=str, help='Path to dataset.yaml for full validation')
    parser.add_argument('--output', type=str, default='logs',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("VisionGuard - Model Evaluation")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model_path=args.model)
    
    # Run evaluation
    if args.data_yaml and os.path.exists(args.data_yaml):
        # Full Ultralytics validation (most accurate for mAP)
        evaluator.run_ultralytics_val(args.data_yaml)
    
    # Evaluate on test images
    if os.path.exists(args.images):
        evaluator.evaluate_on_dataset(args.images, args.labels)
    else:
        print(f"[EVAL WARNING] Test images directory not found: {args.images}")
    
    # Generate visualizations
    evaluator.plot_metrics(save_dir=args.output)
    
    # Save report
    evaluator.save_report(os.path.join(args.output, 'evaluation_report.json'))
    
    # Print summary
    evaluator.print_summary()


if __name__ == "__main__":
    main()
