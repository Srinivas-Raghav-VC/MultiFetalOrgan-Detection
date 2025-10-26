#!/usr/bin/env python3
"""
Focused Analysis of boxes/annotation Directory
===============================================

This script specifically analyzes the boxes/annotation directory to understand
how many unique IMAGES have annotations, not just counting bounding boxes.

Usage:
    python analyze_boxes_directory.py
"""

import argparse
from pathlib import Path
from lxml import etree
from collections import defaultdict, Counter
import json


def analyze_boxes_directory(dataset_root):
    """Analyze ONLY the boxes/annotation directory"""
    print("=" * 80)
    print("ANALYZING boxes/annotation DIRECTORY")
    print("=" * 80)

    boxes_dir = dataset_root / 'boxes' / 'annotation'

    if not boxes_dir.exists():
        print(f"ERROR: boxes/annotation directory not found: {boxes_dir}")
        return None

    xml_files = list(boxes_dir.rglob('annotations.xml'))
    print(f"\nFound {len(xml_files)} XML files in boxes/annotation/")

    # Track statistics
    total_images_with_boxes = 0
    total_bounding_boxes = 0
    total_images_without_boxes = 0
    stream_details = {}
    class_counts = Counter()
    all_annotated_images = set()  # Track unique images across all streams

    for xml_file in xml_files:
        stream_name = xml_file.parent.name
        print(f"\nProcessing: {stream_name}")

        root = etree.parse(str(xml_file)).getroot()
        images = root.findall('.//image')

        images_with_boxes = 0
        images_without_boxes = 0
        box_count = 0
        stream_class_counts = Counter()

        for img in images:
            img_name = img.get('name') or img.get('id')
            boxes = img.findall('.//box')

            if boxes:
                images_with_boxes += 1
                all_annotated_images.add(img_name)  # Track unique image

                for box in boxes:
                    box_count += 1
                    label = box.get('label', '').strip().lower()
                    class_counts[label] += 1
                    stream_class_counts[label] += 1
            else:
                images_without_boxes += 1

        stream_details[stream_name] = {
            'total_images': len(images),
            'images_with_boxes': images_with_boxes,
            'images_without_boxes': images_without_boxes,
            'total_boxes': box_count,
            'class_distribution': dict(stream_class_counts),
            'xml_path': str(xml_file)
        }

        total_images_with_boxes += images_with_boxes
        total_images_without_boxes += images_without_boxes
        total_bounding_boxes += box_count

        print(f"   Total images in XML: {len(images)}")
        print(f"   Images WITH boxes: {images_with_boxes}")
        print(f"   Images WITHOUT boxes: {images_without_boxes}")
        print(f"   Total bounding boxes: {box_count}")
        if stream_class_counts:
            print(f"   Classes: {dict(stream_class_counts)}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - boxes/annotation Directory")
    print("=" * 80)
    print(f"Total streams: {len(stream_details)}")
    print(f"\nIMAGE COUNTS:")
    print(f"   Total images referenced in XMLs: {total_images_with_boxes + total_images_without_boxes}")
    print(f"   Images WITH annotations: {total_images_with_boxes}")
    print(f"   Images WITHOUT annotations: {total_images_without_boxes}")
    print(f"   Unique annotated images: {len(all_annotated_images)}")
    print(f"\nBOUNDING BOX COUNTS:")
    print(f"   Total bounding boxes: {total_bounding_boxes}")
    print(f"\nCLASS DISTRIBUTION:")
    for label, count in class_counts.most_common():
        print(f"   {label}: {count} boxes")

    return {
        'total_streams': len(stream_details),
        'total_images_in_xmls': total_images_with_boxes + total_images_without_boxes,
        'images_with_boxes': total_images_with_boxes,
        'images_without_boxes': total_images_without_boxes,
        'unique_annotated_images': len(all_annotated_images),
        'total_bounding_boxes': total_bounding_boxes,
        'class_distribution': dict(class_counts),
        'stream_details': stream_details
    }


def check_image_existence(dataset_root, boxes_analysis):
    """Check which annotated images actually exist on disk"""
    print("\n" + "=" * 80)
    print("CHECKING IMAGE FILE EXISTENCE")
    print("=" * 80)

    four_poses_dir = dataset_root / 'four_poses'

    if not four_poses_dir.exists():
        print(f"ERROR: four_poses directory not found: {four_poses_dir}")
        return None

    total_found = 0
    total_missing = 0
    missing_examples = []

    for stream_name, details in boxes_analysis['stream_details'].items():
        stream_dir = four_poses_dir / stream_name

        if not stream_dir.exists():
            print(f"\nWARNING: {stream_name}: Directory doesn't exist!")
            continue

        # Parse XML again to check image existence
        xml_file = Path(details['xml_path'])
        root = etree.parse(str(xml_file)).getroot()

        found = 0
        missing = 0

        for img in root.findall('.//image'):
            boxes = img.findall('.//box')
            if not boxes:
                continue  # Skip images without annotations

            img_name = img.get('name') or img.get('id')
            img_path = stream_dir / img_name

            if img_path.exists():
                found += 1
            else:
                missing += 1
                if len(missing_examples) < 10:
                    missing_examples.append(f"{stream_name}/{img_name}")

        total_found += found
        total_missing += missing

        if missing > 0:
            print(f"\nWARNING: {stream_name}: {missing} annotated images missing")

    print("\n" + "=" * 80)
    print(f"ANNOTATED IMAGES ON DISK:")
    print(f"   Found: {total_found}")
    print(f"   Missing: {total_missing}")

    if missing_examples:
        print(f"\nWARNING: Missing image examples:")
        for example in missing_examples:
            print(f"      {example}")

    print("=" * 80)

    return {
        'found': total_found,
        'missing': total_missing,
        'missing_examples': missing_examples
    }


def simulate_split(boxes_analysis):
    """Simulate the train/val/test split that prepare_fpus23.py would create"""
    print("\n" + "=" * 80)
    print("SIMULATING TRAIN/VAL/TEST SPLIT")
    print("=" * 80)

    streams = list(boxes_analysis['stream_details'].keys())

    # Simulate group-based split (80/10/10)
    import random
    random.seed(42)
    shuffled = streams.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_streams = shuffled[:n_train]
    val_streams = shuffled[n_train:n_train+n_val]
    test_streams = shuffled[n_train+n_val:]

    print(f"\nSplit by stream:")
    print(f"   Train: {len(train_streams)} streams")
    print(f"   Val: {len(val_streams)} streams")
    print(f"   Test: {len(test_streams)} streams")

    # Count images per split
    train_count = sum(boxes_analysis['stream_details'][s]['images_with_boxes']
                      for s in train_streams)
    val_count = sum(boxes_analysis['stream_details'][s]['images_with_boxes']
                    for s in val_streams)
    test_count = sum(boxes_analysis['stream_details'][s]['images_with_boxes']
                     for s in test_streams)

    print(f"\nExpected image counts:")
    print(f"   Train: {train_count} images")
    print(f"   Val: {val_count} images")
    print(f"   Test: {test_count} images")
    print(f"   Total: {train_count + val_count + test_count} images")

    return {
        'train': train_count,
        'val': val_count,
        'test': test_count,
        'train_streams': train_streams,
        'val_streams': val_streams,
        'test_streams': test_streams
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze boxes/annotation directory')
    parser.add_argument('--dataset-root', type=str,
                       default=r"C:\Users\Srinivas's G14\Downloads\SAE_2\FPUS23_Dataset\Dataset",
                       help='Path to dataset root')
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)

    if not dataset_root.exists():
        print(f"ERROR: Dataset root not found: {dataset_root}")
        return

    print("=" * 80)
    print("FPUS23 boxes/annotation FOCUSED ANALYSIS")
    print("=" * 80)
    print(f"Dataset root: {dataset_root}\n")

    # Step 1: Analyze boxes/annotation directory
    boxes_analysis = analyze_boxes_directory(dataset_root)

    if not boxes_analysis:
        return

    # Step 2: Check image file existence
    existence_check = check_image_existence(dataset_root, boxes_analysis)

    # Step 3: Simulate split
    split_simulation = simulate_split(boxes_analysis)

    # Save detailed results
    output = {
        'boxes_analysis': boxes_analysis,
        'existence_check': existence_check,
        'split_simulation': split_simulation
    }

    output_file = Path('boxes_directory_analysis.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nDetailed analysis saved to {output_file}")

    # Final summary
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print(f"1. Total annotated images in boxes/: {boxes_analysis['images_with_boxes']}")
    print(f"2. Unique annotated images: {boxes_analysis['unique_annotated_images']}")
    print(f"3. Images found on disk: {existence_check['found']}")
    print(f"4. Images missing from disk: {existence_check['missing']}")
    print(f"5. Expected after 80/10/10 split:")
    print(f"   - Train: {split_simulation['train']}")
    print(f"   - Val: {split_simulation['val']}")
    print(f"   - Test: {split_simulation['test']}")

    if existence_check['found'] < 1000:
        print("\nWARNING: Very small dataset!")
        print(f"   Only {existence_check['found']} annotated images exist on disk.")

    print("=" * 80)


if __name__ == '__main__':
    main()
