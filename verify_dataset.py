"""
Quick verification script for dataset structure
"""
import os
import yaml
from pathlib import Path


def verify_dataset(data_yaml_path='datasets/hazards/data.yaml'):
    """Verify dataset structure and contents"""
    
    print("="*60)
    print("Dataset Verification")
    print("="*60)
    
    # Check if data.yaml exists
    if not os.path.exists(data_yaml_path):
        print(f"\n[ERROR] Dataset configuration not found: {data_yaml_path}")
        print("\n[HELP] Please create the dataset according to DATASET_SETUP.md")
        print("\nExpected structure:")
        print("  datasets/hazards/")
        print("    ├── data.yaml")
        print("    ├── train/")
        print("    │   ├── images/")
        print("    │   └── labels/")
        print("    ├── valid/")
        print("    │   ├── images/")
        print("    │   └── labels/")
        print("    └── test/")
        print("        ├── images/")
        print("        └── labels/")
        return False
    
    # Load data.yaml
    try:
        with open(data_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"\n[ERROR] Could not parse data.yaml: {e}")
        return False
    
    print(f"\n✓ Found dataset configuration: {data_yaml_path}")
    
    # Display configuration
    print(f"\nDataset Configuration:")
    print(f"  Path: {data.get('path', 'Not specified')}")
    print(f"  Number of classes: {data.get('nc', 'Not specified')}")
    print(f"  Classes: {data.get('names', 'Not specified')}")
    
    # Get base path
    base_path = Path(data_yaml_path).parent
    if 'path' in data and data['path'] != './':
        base_path = Path(data['path'])
    
    # Check splits
    all_valid = True
    for split in ['train', 'val', 'test']:
        if split not in data:
            # Try alternate name
            if split == 'val' and 'valid' in data:
                split_key = 'valid'
            else:
                print(f"\n[WARNING] '{split}' split not defined in data.yaml")
                continue
        else:
            split_key = split
        
        print(f"\n{split.upper()} Split:")
        
        # Get image directory
        img_path_str = data[split_key]
        
        # Handle different path formats
        if 'images' in img_path_str:
            img_dir = base_path / img_path_str
            lbl_dir = base_path / img_path_str.replace('images', 'labels')
        else:
            img_dir = base_path / img_path_str / 'images'
            lbl_dir = base_path / img_path_str / 'labels'
        
        # Check if directories exist
        if not img_dir.exists():
            print(f"  [ERROR] Image directory not found: {img_dir}")
            all_valid = False
            continue
        
        if not lbl_dir.exists():
            print(f"  [ERROR] Label directory not found: {lbl_dir}")
            all_valid = False
            continue
        
        print(f"  ✓ Image directory: {img_dir}")
        print(f"  ✓ Label directory: {lbl_dir}")
        
        # Count files
        img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpeg'))
        lbl_files = list(lbl_dir.glob('*.txt'))
        
        print(f"  Images: {len(img_files)}")
        print(f"  Labels: {len(lbl_files)}")
        
        # Check if counts match
        if len(img_files) != len(lbl_files):
            print(f"  [WARNING] Mismatch between images and labels!")
            all_valid = False
        else:
            print(f"  ✓ Counts match")
        
        # Verify a few label files
        if lbl_files:
            sample_label = lbl_files[0]
            try:
                with open(sample_label, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        parts = lines[0].strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            if class_id < data['nc']:
                                print(f"  ✓ Label format appears valid")
                            else:
                                print(f"  [WARNING] Class ID {class_id} exceeds nc={data['nc']}")
                        else:
                            print(f"  [WARNING] Label format may be incorrect")
            except Exception as e:
                print(f"  [ERROR] Could not read sample label: {e}")
    
    # Final verdict
    print("\n" + "="*60)
    if all_valid:
        print("✓ Dataset verification PASSED")
        print("\nYou can now train the model with:")
        print(f"  python train_model.py --data {data_yaml_path}")
    else:
        print("✗ Dataset verification FAILED")
        print("\nPlease fix the issues above before training")
    print("="*60)
    
    return all_valid


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify dataset structure')
    parser.add_argument('--data', type=str, default='datasets/hazards/data.yaml',
                       help='Path to dataset.yaml file')
    
    args = parser.parse_args()
    
    verify_dataset(args.data)
