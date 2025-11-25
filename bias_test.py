"""
Bias Testing Module
Tests model performance across different demographic/geographic scenarios
"""

import os
import cv2
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO

from utils.config import YOLO_PRETRAINED, YOLO_FINETUNED, TEST_IMAGES_DIR, EVAL_CONF_THRESHOLD


class BiasTestRunner:
    """Test model for bias across different scenarios"""
    
    def __init__(self, model_path=None):
        """
        Initialize bias tester
        
        Args:
            model_path: Path to YOLO model
        """
        if model_path and os.path.exists(model_path):
            self.model_path = model_path
        elif os.path.exists(YOLO_FINETUNED):
            self.model_path = YOLO_FINETUNED
        else:
            self.model_path = YOLO_PRETRAINED
        
        print(f"[BIAS TEST] Loading model: {self.model_path}")
        self.model = YOLO(self.model_path)
        
        self.results = []
    
    def test_on_category(self, images_dir, category_name):
        """
        Test model on a specific category of images
        
        Args:
            images_dir: Directory containing test images
            category_name: Name of the category (e.g., 'western_cities', 'asian_cities')
        
        Returns:
            dict: Results for this category
        """
        print(f"\n[BIAS TEST] Testing category: {category_name}")
        print(f"[BIAS TEST] Images directory: {images_dir}")
        
        if not os.path.exists(images_dir):
            print(f"[BIAS TEST ERROR] Directory not found: {images_dir}")
            return None
        
        image_files = list(Path(images_dir).glob("*.jpg")) + \
                     list(Path(images_dir).glob("*.png")) + \
                     list(Path(images_dir).glob("*.jpeg"))
        
        if not image_files:
            print(f"[BIAS TEST WARNING] No images found in {images_dir}")
            return None
        
        print(f"[BIAS TEST] Found {len(image_files)} images")
        
        category_results = []
        
        for img_path in image_files:
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"[BIAS TEST WARNING] Could not read: {img_path}")
                continue
            
            # Run inference
            results = self.model(image, conf=EVAL_CONF_THRESHOLD, verbose=False)[0]
            
            # Extract detections
            detections = []
            confidence_scores = []
            
            for box in results.boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                confidence = float(box.conf[0])
                
                detections.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence
                })
                confidence_scores.append(confidence)
            
            # Record results
            result = {
                'image_path': str(img_path),
                'image_name': img_path.name,
                'category': category_name,
                'num_detections': len(detections),
                'detections': detections,
                'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
                'max_confidence': max(confidence_scores) if confidence_scores else 0.0,
                'min_confidence': min(confidence_scores) if confidence_scores else 0.0
            }
            
            category_results.append(result)
            self.results.append(result)
            
            print(f"  ✓ {img_path.name}: {len(detections)} detections, "
                  f"avg conf: {result['avg_confidence']:.3f}")
        
        # Calculate category statistics
        if category_results:
            avg_detections = np.mean([r['num_detections'] for r in category_results])
            avg_confidence = np.mean([r['avg_confidence'] for r in category_results])
            
            print(f"\n[BIAS TEST] Category '{category_name}' Summary:")
            print(f"  Images tested: {len(category_results)}")
            print(f"  Avg detections per image: {avg_detections:.2f}")
            print(f"  Avg confidence: {avg_confidence:.3f}")
        
        return category_results
    
    def test_bias_folders(self, base_dir=None):
        """
        Test on multiple bias categories organized in folders
        
        Expected folder structure:
        bias_test/
          ├── western_cities/
          ├── asian_cities/
          ├── dark_lighting/
          ├── bright_lighting/
          └── ...
        
        Args:
            base_dir: Base directory containing category folders
        
        Returns:
            dict: Results grouped by category
        """
        if base_dir is None:
            base_dir = os.path.join(TEST_IMAGES_DIR, "bias_test")
        
        print(f"\n[BIAS TEST] Scanning for bias test folders in: {base_dir}")
        
        if not os.path.exists(base_dir):
            print(f"[BIAS TEST ERROR] Bias test directory not found: {base_dir}")
            print(f"[BIAS TEST] Please create folder structure with test categories")
            return {}
        
        # Find all subdirectories
        categories = [d for d in os.listdir(base_dir) 
                     if os.path.isdir(os.path.join(base_dir, d))]
        
        if not categories:
            print(f"[BIAS TEST WARNING] No category folders found in {base_dir}")
            return {}
        
        print(f"[BIAS TEST] Found {len(categories)} categories: {categories}")
        
        all_results = {}
        
        for category in categories:
            category_dir = os.path.join(base_dir, category)
            results = self.test_on_category(category_dir, category)
            if results:
                all_results[category] = results
        
        return all_results
    
    def compare_categories(self, results_by_category):
        """
        Compare performance across different categories
        
        Args:
            results_by_category: Dict of category_name -> list of results
        
        Returns:
            DataFrame: Comparison statistics
        """
        comparison_data = []
        
        for category, results in results_by_category.items():
            if not results:
                continue
            
            stats = {
                'Category': category,
                'Num_Images': len(results),
                'Avg_Detections': np.mean([r['num_detections'] for r in results]),
                'Avg_Confidence': np.mean([r['avg_confidence'] for r in results]),
                'Std_Confidence': np.std([r['avg_confidence'] for r in results]),
                'Max_Confidence': max([r['max_confidence'] for r in results]),
                'Min_Confidence': min([r['min_confidence'] for r in results])
            }
            
            comparison_data.append(stats)
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def save_results_csv(self, filepath='logs/bias_test_results.csv'):
        """Save detailed results to CSV"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Flatten results for CSV
        csv_data = []
        for result in self.results:
            row = {
                'image_name': result['image_name'],
                'category': result['category'],
                'num_detections': result['num_detections'],
                'avg_confidence': result['avg_confidence'],
                'max_confidence': result['max_confidence'],
                'min_confidence': result['min_confidence']
            }
            
            # Add detection details
            for i, det in enumerate(result['detections'][:5]):  # Limit to 5 detections
                row[f'det_{i+1}_class'] = det['class_name']
                row[f'det_{i+1}_conf'] = det['confidence']
            
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(filepath, index=False)
        print(f"[BIAS TEST] Results saved to {filepath}")
        
        return df
    
    def save_json_report(self, filepath='logs/bias_test_report.json'):
        """Save complete results as JSON"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        report = {
            'model_path': self.model_path,
            'total_images_tested': len(self.results),
            'results': self.results
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"[BIAS TEST] JSON report saved to {filepath}")
    
    def plot_comparison(self, results_by_category, save_dir='logs'):
        """Generate comparison visualizations"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Get comparison DataFrame
        df = self.compare_categories(results_by_category)
        
        if df.empty:
            print("[BIAS TEST WARNING] No data to plot")
            return
        
        # 1. Average Confidence by Category
        plt.figure(figsize=(12, 6))
        bars = plt.bar(df['Category'], df['Avg_Confidence'], 
                      yerr=df['Std_Confidence'], capsize=5,
                      color='skyblue', edgecolor='black')
        plt.xlabel('Category', fontsize=12)
        plt.ylabel('Average Confidence', fontsize=12)
        plt.title('Model Confidence Across Different Categories', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'bias_confidence_comparison.png'), dpi=300)
        plt.close()
        
        # 2. Number of Detections by Category
        plt.figure(figsize=(12, 6))
        plt.bar(df['Category'], df['Avg_Detections'],
               color='lightcoral', edgecolor='black')
        plt.xlabel('Category', fontsize=12)
        plt.ylabel('Average Detections per Image', fontsize=12)
        plt.title('Detection Count Across Different Categories', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'bias_detection_count.png'), dpi=300)
        plt.close()
        
        # 3. Confidence range (min to max) by category
        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(len(df))
        ax.scatter(x, df['Max_Confidence'], label='Max Confidence', 
                  color='green', s=100, alpha=0.6, marker='^')
        ax.scatter(x, df['Avg_Confidence'], label='Avg Confidence',
                  color='blue', s=100, alpha=0.6, marker='o')
        ax.scatter(x, df['Min_Confidence'], label='Min Confidence',
                  color='red', s=100, alpha=0.6, marker='v')
        
        for i in x:
            ax.plot([i, i], [df['Min_Confidence'].iloc[i], df['Max_Confidence'].iloc[i]],
                   'k-', alpha=0.3, linewidth=1)
        
        ax.set_xticks(x)
        ax.set_xticklabels(df['Category'], rotation=45, ha='right')
        ax.set_xlabel('Category', fontsize=12)
        ax.set_ylabel('Confidence Score', fontsize=12)
        ax.set_title('Confidence Score Range by Category', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'bias_confidence_range.png'), dpi=300)
        plt.close()
        
        print(f"[BIAS TEST] Comparison plots saved to {save_dir}")
        
        # Print comparison table
        print("\n[BIAS TEST] Category Comparison:")
        print(df.to_string(index=False))


def main():
    """Main bias testing script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run bias testing on model')
    parser.add_argument('--model', type=str, help='Path to model file')
    parser.add_argument('--bias-dir', type=str, 
                       default=os.path.join(TEST_IMAGES_DIR, 'bias_test'),
                       help='Directory containing bias test categories')
    parser.add_argument('--output', type=str, default='logs',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("VisionGuard - Bias Testing")
    print("=" * 60)
    
    # Initialize tester
    tester = BiasTestRunner(model_path=args.model)
    
    # Run tests on all categories
    results_by_category = tester.test_bias_folders(base_dir=args.bias_dir)
    
    if not results_by_category:
        print("\n[BIAS TEST] No results to analyze.")
        print(f"[BIAS TEST] Please add test images to: {args.bias_dir}")
        print("[BIAS TEST] Organize images in subfolders by category, e.g.:")
        print("  - western_cities/")
        print("  - asian_cities/")
        print("  - dark_lighting/")
        print("  - bright_lighting/")
        return
    
    # Generate comparison plots
    tester.plot_comparison(results_by_category, save_dir=args.output)
    
    # Save results
    tester.save_results_csv(os.path.join(args.output, 'bias_test_results.csv'))
    tester.save_json_report(os.path.join(args.output, 'bias_test_report.json'))
    
    print("\n" + "=" * 60)
    print("Bias testing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
