#!/usr/bin/env python3
"""
Preprocess FPUS23 dataset with median blur despeckling.

This script applies SOTA ultrasound preprocessing to reduce speckle noise
while preserving edge information critical for object detection.

Based on:
  - YOLOv7-FPUS23 study (2024): "Preprocessed Images with median blur"
  - Wang et al. (2022): Fetal cardiac detection with preprocessing
  - Multiple ultrasound imaging papers (2022-2025)

Usage:
  python preprocess_fpus23_despeckle.py \
    --input-dir dataset/fpus23_yolo/images \
    --output-dir dataset/fpus23_yolo_despeckled/images \
    --kernel-size 5 \
    --splits train val test

Output:
  - Despeckled images saved to output-dir
  - Directory structure preserved
  - Original annotations work with despeckled images
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List
import shutil

import cv2
import numpy as np
from tqdm import tqdm


def despeckle_image(
    img_path: Path,
    output_path: Path,
    kernel_size: int = 5,
    preserve_color: bool = True
) -> None:
    """
    Apply median blur to reduce speckle noise.
    
    Args:
        img_path: Path to input image
        output_path: Path to save despeckled image
        kernel_size: Median filter kernel (3, 5, or 7)
        preserve_color: If True, apply to each channel independently
    """
    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Warning: Could not read {img_path}")
        return
    
    # Apply median blur
    if preserve_color and len(img.shape) == 3:
        # Apply to each channel independently
        despeckled = np.zeros_like(img)
        for c in range(img.shape[2]):
            despeckled[:, :, c] = cv2.medianBlur(img[:, :, c], kernel_size)
    else:
        despeckled = cv2.medianBlur(img, kernel_size)
    
    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), despeckled)


def process_split(
    input_dir: Path,
    output_dir: Path,
    split: str,
    kernel_size: int
) -> int:
    """
    Process all images in a split (train/val/test).
    
    Returns:
        Number of images processed
    """
    split_dir = input_dir / split
    if not split_dir.exists():
        print(f"Warning: {split_dir} does not exist, skipping")
        return 0
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(split_dir.glob(ext))
        image_files.extend(split_dir.glob(ext.upper()))
    
    if not image_files:
        print(f"Warning: No images found in {split_dir}")
        return 0
    
    print(f"\nProcessing {split} set: {len(image_files)} images")
    
    # Process each image
    for img_path in tqdm(image_files, desc=f"Despeckling {split}"):
        output_path = output_dir / split / img_path.name
        despeckle_image(img_path, output_path, kernel_size)
    
    return len(image_files)


def copy_labels(input_dir: Path, output_dir: Path, splits: List[str]) -> None:
    """
    Copy label files from input to output directory.
    
    Labels don't need preprocessing, just copy them.
    """
    for split in splits:
        label_input = input_dir.parent / 'labels' / split
        label_output = output_dir.parent / 'labels' / split
        
        if not label_input.exists():
            print(f"Warning: {label_input} does not exist, skipping labels for {split}")
            continue
        
        print(f"\nCopying labels for {split} set...")
        label_output.mkdir(parents=True, exist_ok=True)
        
        for label_file in label_input.glob('*.txt'):
            shutil.copy2(label_file, label_output / label_file.name)


def create_data_yaml(output_dir: Path, original_yaml: Path) -> None:
    """
    Create new data.yaml pointing to despeckled images.
    """
    import yaml
    
    # Read original yaml
    with open(original_yaml) as f:
        data = yaml.safe_load(f)
    
    # Update paths to point to despeckled directory
    despeckled_root = output_dir.parent
    data['path'] = str(despeckled_root)
    data['train'] = 'images/train'
    data['val'] = 'images/val'
    if 'test' in data:
        data['test'] = 'images/test'
    
    # Add note about preprocessing
    data['_preprocessing'] = 'median_blur_kernel_5'
    data['_note'] = 'Images preprocessed with median blur for speckle reduction'
    
    # Save new yaml
    output_yaml = despeckled_root / 'data.yaml'
    with open(output_yaml, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"\n✅ Created new data.yaml: {output_yaml}")


def main():
    ap = argparse.ArgumentParser(
        description='Preprocess FPUS23 with median blur despeckling',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    ap.add_argument('--input-dir', type=str, required=True,
                    help='Input images directory (e.g., dataset/fpus23_yolo/images)')
    ap.add_argument('--output-dir', type=str, required=True,
                    help='Output images directory (e.g., dataset/fpus23_yolo_despeckled/images)')
    ap.add_argument('--kernel-size', type=int, default=5, choices=[3, 5, 7],
                    help='Median blur kernel size (3=mild, 5=moderate, 7=strong)')
    ap.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                    help='Dataset splits to process')
    ap.add_argument('--data-yaml', type=str, default=None,
                    help='Path to original data.yaml (optional, for creating new yaml)')
    
    args = ap.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    print("=" * 80)
    print("🔬 FPUS23 Despeckling Preprocessing")
    print("=" * 80)
    print(f"Input:        {input_dir}")
    print(f"Output:       {output_dir}")
    print(f"Kernel Size:  {args.kernel_size}")
    print(f"Splits:       {', '.join(args.splits)}")
    print("=" * 80)
    print("\nPreprocessing Method: Median Blur")
    print("Purpose: Reduce speckle noise while preserving edges")
    print("SOTA Reference: YOLOv7-FPUS23 (2024), Fetal cardiac detection (2022)")
    print("=" * 80 + "\n")
    
    # Process each split
    total_processed = 0
    for split in args.splits:
        n_processed = process_split(input_dir, output_dir, split, args.kernel_size)
        total_processed += n_processed
    
    # Copy labels
    print("\n" + "=" * 80)
    copy_labels(input_dir, output_dir, args.splits)
    
    # Create new data.yaml if original provided
    if args.data_yaml:
        original_yaml = Path(args.data_yaml)
        if original_yaml.exists():
            create_data_yaml(output_dir, original_yaml)
        else:
            print(f"\nWarning: Original data.yaml not found: {original_yaml}")
    
    print("\n" + "=" * 80)
    print("✅ Preprocessing Complete!")
    print("=" * 80)
    print(f"Total images processed: {total_processed}")
    print(f"\nOutput structure:")
    print(f"  {output_dir.parent}/")
    print(f"  ├── images/")
    for split in args.splits:
        print(f"  │   ├── {split}/  (despeckled)")
    print(f"  ├── labels/")
    for split in args.splits:
        print(f"  │   ├── {split}/  (copied from original)")
    if args.data_yaml:
        print(f"  └── data.yaml  (updated paths)")
    print()
    print("Next steps:")
    print(f"  1. Train YOLO with despeckled data:")
    print(f"     python train_yolo_fpus23.py \\")
    print(f"       --data {output_dir.parent}/data.yaml \\")
    print(f"       --model yolo11n.pt \\")
    print(f"       --epochs 100")
    print()
    print(f"  2. Compare with non-despeckled baseline")
    print(f"  3. Expected improvement: +2-4% mAP on small objects (Arms/Legs)")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
