#!/usr/bin/env python3
"""
Comprehensive FPUS23 Dataset Analysis Script
============================================

This script analyzes why only ~887 images are being processed
when XMLs reference 9,455 images.

Usage:
    python analyze_dataset.py --dataset-root "C:/Users/Srinivas's G14/Downloads/SAE_2/FPUS23_Dataset/Dataset"
"""

import argparse
from pathlib import Path
from lxml import etree
from collections import defaultdict, Counter
import json


def analyze_xmls(dataset_root):
    """Analyze all XML files to count image references"""
    print("="*80)
    print("STEP 1: ANALYZING XML FILES")
    print("="*80)

    boxes_dir = dataset_root / 'boxes' / 'annotation'
    annos_dir = dataset_root / 'annos' / 'annotation'

    xml_files = []
    if boxes_dir.exists():
        xml_files.extend(list(boxes_dir.rglob('annotations.xml')))
    if annos_dir.exists():
        xml_files.extend(list(annos_dir.rglob('annotations.xml')))

    print(f"\nFound {len(xml_files)} XML files")

    total_images_in_xmls = 0
    images_with_boxes = 0
    images_without_boxes = 0
    stream_stats = {}

    for xml_file in xml_files:
        stream_name = xml_file.parent.name
        root = etree.parse(str(xml_file)).getroot()
        images = root.findall('.//image')

        with_boxes = 0
        without_boxes = 0

        for img in images:
            boxes = img.findall('.//box')
            if boxes:
                with_boxes += 1
            else:
                without_boxes += 1

        stream_stats[stream_name] = {
            'total': len(images),
            'with_boxes': with_boxes,
            'without_boxes': without_boxes,
            'xml_path': str(xml_file)
        }

        total_images_in_xmls += len(images)
        images_with_boxes += with_boxes
        images_without_boxes += without_boxes

    print(f"\nTotal image references in XMLs: {total_images_in_xmls}")
    print(f"  - With boxes: {images_with_boxes}")
    print(f"  - Without boxes: {images_without_boxes}")

    return stream_stats, total_images_in_xmls


def check_image_files(dataset_root, stream_stats):
    """Check which image files actually exist"""
    print("\n" + "="*80)
    print("STEP 2: CHECKING ACTUAL IMAGE FILES")
    print("="*80)

    four_poses = dataset_root / 'four_poses'

    total_pngs = 0
    missing_count = 0
    found_count = 0

    results = {}

    for stream_name, stats in stream_stats.items():
        xml_file = Path(stats['xml_path'])

        # Check if corresponding four_poses directory exists
        stream_dir = four_poses / stream_name

        if not stream_dir.exists():
            print(f"\n⚠️  {stream_name}: Directory doesn't exist!")
            results[stream_name] = {
                'expected': stats['total'],
                'found': 0,
                'missing': stats['total'],
                'exists': False
            }
            missing_count += stats['total']
            continue

        # Count PNG files
        png_files = list(stream_dir.glob('*.png'))
        total_pngs += len(png_files)

        # Parse XML and check which images exist
        root = etree.parse(str(xml_file)).getroot()
        images = root.findall('.//image')

        found = 0
        missing = 0
        missing_examples = []

        for img in images:
            img_name = img.get('name') or img.get('id')
            img_path = stream_dir / img_name

            if img_path.exists():
                found += 1
            else:
                missing += 1
                if len(missing_examples) < 3:
                    missing_examples.append(img_name)

        results[stream_name] = {
            'expected': len(images),
            'found': found,
            'missing': missing,
            'exists': True,
            'png_count': len(png_files),
            'missing_examples': missing_examples
        }

        found_count += found
        missing_count += missing

        if missing > 0:
            print(f"\n⚠️  {stream_name}:")
            print(f"   Expected: {len(images)} images")
            print(f"   Found: {found} images ({len(png_files)} total PNGs in dir)")
            print(f"   Missing: {missing} images")
            if missing_examples:
                print(f"   Examples: {missing_examples}")

    print("\n" + "="*80)
    print(f"TOTAL IMAGES:")
    print(f"  Referenced in XMLs: {found_count + missing_count}")
    print(f"  Actually found: {found_count}")
    print(f"  Missing: {missing_count}")
    print(f"  Total PNGs in four_poses: {total_pngs}")
    print("="*80)

    return results


def analyze_class_distribution(dataset_root, stream_stats):
    """Analyze class distribution in annotations"""
    print("\n" + "="*80)
    print("STEP 3: CLASS DISTRIBUTION ANALYSIS")
    print("="*80)

    class_counts = Counter()

    for stream_name, stats in stream_stats.items():
        xml_file = Path(stats['xml_path'])
        root = etree.parse(str(xml_file)).getroot()

        for img in root.findall('.//image'):
            for box in img.findall('.//box'):
                label = box.get('label', '').strip().lower()
                class_counts[label] += 1

    print("\nAnnotation counts by class:")
    for label, count in class_counts.most_common():
        print(f"  {label}: {count}")

    return class_counts


def analyze_split_behavior(dataset_root, stream_stats):
    """Understand how train/val/test split works"""
    print("\n" + "="*80)
    print("STEP 4: ANALYZING SPLIT BEHAVIOR")
    print("="*80)

    streams = list(stream_stats.keys())

    print(f"\nTotal streams: {len(streams)}")
    print("\nStream names:")
    for stream in sorted(streams):
        print(f"  - {stream}")

    # Simulate 80/10/10 split by stream
    import random
    random.seed(42)
    shuffled = streams.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_streams = set(shuffled[:n_train])
    val_streams = set(shuffled[n_train:n_train+n_val])
    test_streams = set(shuffled[n_train+n_val:])

    print(f"\nSimulated split (by stream):")
    print(f"  Train: {len(train_streams)} streams")
    print(f"  Val: {len(val_streams)} streams")
    print(f"  Test: {len(test_streams)} streams")

    # Count images per split
    train_count = sum(stream_stats[s]['with_boxes'] for s in train_streams)
    val_count = sum(stream_stats[s]['with_boxes'] for s in val_streams)
    test_count = sum(stream_stats[s]['with_boxes'] for s in test_streams)

    print(f"\nExpected image counts:")
    print(f"  Train: {train_count} images")
    print(f"  Val: {val_count} images")
    print(f"  Test: {test_count} images")
    print(f"  Total: {train_count + val_count + test_count} images")

    return {
        'train': train_count,
        'val': val_count,
        'test': test_count
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze FPUS23 dataset')
    parser.add_argument('--dataset-root', type=str,
                       default=r"C:\Users\Srinivas's G14\Downloads\SAE_2\FPUS23_Dataset\Dataset",
                       help='Path to dataset root')
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)

    if not dataset_root.exists():
        print(f"❌ Dataset root not found: {dataset_root}")
        return

    print("="*80)
    print("FPUS23 DATASET ANALYSIS")
    print("="*80)
    print(f"Dataset root: {dataset_root}\n")

    # Step 1: Analyze XMLs
    stream_stats, total_xml_refs = analyze_xmls(dataset_root)

    # Step 2: Check actual image files
    image_results = check_image_files(dataset_root, stream_stats)

    # Step 3: Class distribution
    class_counts = analyze_class_distribution(dataset_root, stream_stats)

    # Step 4: Split behavior
    split_stats = analyze_split_behavior(dataset_root, stream_stats)

    # Save results
    output = {
        'total_xml_references': total_xml_refs,
        'streams': stream_stats,
        'image_check': image_results,
        'class_distribution': dict(class_counts),
        'expected_split': split_stats
    }

    output_file = Path('dataset_analysis.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n✅ Analysis complete! Results saved to {output_file}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total streams: {len(stream_stats)}")
    print(f"Total image references in XMLs: {total_xml_refs}")

    total_found = sum(r['found'] for r in image_results.values())
    total_missing = sum(r['missing'] for r in image_results.values())

    print(f"Images that actually exist: {total_found}")
    print(f"Images missing from disk: {total_missing}")
    print(f"\nExpected train/val/test split:")
    print(f"  Train: {split_stats['train']}")
    print(f"  Val: {split_stats['val']}")
    print(f"  Test: {split_stats['test']}")

    if total_found < 1000:
        print("\n⚠️  WARNING: Dataset is very small!")
        print(f"   Only {total_found} images with annotations exist.")
        print(f"   This will limit training performance significantly.")

    print("="*80)


if __name__ == '__main__':
    main()
