#!/usr/bin/env python3
"""
Evaluate trained YOLO models on FPUS23 test set with comprehensive metrics.

This script provides:
  - COCO mAP metrics (mAP50, mAP50-95, per-class AP)
  - Size-stratified analysis (small/medium/large objects)
  - FROC curves (Free-Response ROC for medical imaging)
  - Confusion matrices
  - Speed benchmarking (FPS, latency)
  - Multi-IoU evaluation (IoU 0.5, 0.75, 0.9)

Example Usage:
  python eval_yolo_fpus23.py \
    --weights runs/detect/fpus23/weights/best.pt \
    --data fpus23_yolo/data.yaml \
    --split test \
    --save-dir results/yolo11n

Output:
  - metrics.json: All numerical metrics
  - size_stratified.csv: Performance by object size
  - froc_curve.png: Free-Response ROC plot
  - confusion_matrix.png: Confusion matrix visualization
  - predictions.json: All predictions for analysis
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
import seaborn as sns


# Object size thresholds (COCO standard)
SMALL_AREA = 32**2   # < 1024 px²
MEDIUM_AREA = 96**2  # < 9216 px²
# > 9216 px² = LARGE

CLASS_NAMES = ['Head', 'Abdomen', 'Arms', 'Legs']


def evaluate_at_multiple_ious(
    coco_gt: COCO,
    predictions: List[Dict],
    iou_thresholds: List[float] = [0.5, 0.75, 0.9]
) -> Dict[str, float]:
    """
    Evaluate predictions at multiple IoU thresholds.

    Args:
        coco_gt: Ground truth COCO object
        predictions: List of prediction dicts
        iou_thresholds: IoU thresholds to evaluate at

    Returns:
        Dict mapping "mAP@{iou}" to mAP value
    """
    results = {}

    # Create COCO detections object
    coco_dt = coco_gt.loadRes(predictions)

    for iou_thr in iou_thresholds:
        evaluator = COCOeval(coco_gt, coco_dt, 'bbox')
        evaluator.params.iouThrs = [iou_thr]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

        # Get mAP at this IoU
        map_value = evaluator.stats[0]  # AP @ IoU=threshold
        results[f'mAP@{iou_thr:.2f}'] = float(map_value)

    return results


def compute_size_stratified_metrics(
    coco_gt: COCO,
    predictions: List[Dict]
) -> pd.DataFrame:
    """
    Compute metrics stratified by object size (small/medium/large).

    Args:
        coco_gt: Ground truth COCO object
        predictions: List of prediction dicts

    Returns:
        DataFrame with columns: [class, size_category, AP, num_objects]
    """
    coco_dt = coco_gt.loadRes(predictions)

    rows = []

    for cat_id in coco_gt.getCatIds():
        cat_info = coco_gt.loadCats(cat_id)[0]
        cat_name = cat_info['name']

        # Get all ground truth annotations for this class
        ann_ids = coco_gt.getAnnIds(catIds=[cat_id])
        anns = coco_gt.loadAnns(ann_ids)

        # Stratify by size
        for size_cat, (min_area, max_area) in [
            ('small', (0, SMALL_AREA)),
            ('medium', (SMALL_AREA, MEDIUM_AREA)),
            ('large', (MEDIUM_AREA, float('inf')))
        ]:
            # Filter annotations by size
            size_anns = [a for a in anns if min_area <= a['area'] < max_area]

            if not size_anns:
                continue

            # Evaluate on this subset
            evaluator = COCOeval(coco_gt, coco_dt, 'bbox')
            evaluator.params.catIds = [cat_id]
            evaluator.params.areaRng = [[min_area, max_area]]
            evaluator.params.areaRngLbl = [size_cat]

            try:
                evaluator.evaluate()
                evaluator.accumulate()
                evaluator.summarize()

                ap = evaluator.stats[0]  # AP @ IoU=0.5:0.95
                ap50 = evaluator.stats[1]  # AP @ IoU=0.5

            except:
                ap = 0.0
                ap50 = 0.0

            rows.append({
                'class': cat_name,
                'size_category': size_cat,
                'AP': float(ap),
                'AP50': float(ap50),
                'num_objects': len(size_anns)
            })

    return pd.DataFrame(rows)


def compute_froc_curve(
    coco_gt: COCO,
    predictions: List[Dict],
    confidence_thresholds: np.ndarray = np.linspace(0.01, 0.99, 50)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Free-Response ROC (FROC) curve.

    FROC is standard for medical imaging evaluation:
    - X-axis: False Positives Per Image (FPPI)
    - Y-axis: Sensitivity (recall)

    Args:
        coco_gt: Ground truth COCO object
        predictions: List of prediction dicts
        confidence_thresholds: Thresholds to sweep

    Returns:
        (fppi, sensitivity, thresholds) arrays
    """
    num_images = len(coco_gt.getImgIds())
    total_gt = len(coco_gt.getAnnIds())

    fppi_list = []
    sensitivity_list = []

    coco_dt = coco_gt.loadRes(predictions)

    for conf_thr in confidence_thresholds:
        # Filter predictions by confidence
        filtered_preds = [p for p in predictions if p['score'] >= conf_thr]

        if not filtered_preds:
            fppi_list.append(0.0)
            sensitivity_list.append(0.0)
            continue

        # Evaluate at this threshold
        coco_dt_filtered = coco_gt.loadRes(filtered_preds)
        evaluator = COCOeval(coco_gt, coco_dt_filtered, 'bbox')
        evaluator.params.iouThrs = [0.5]  # Standard IoU=0.5

        try:
            evaluator.evaluate()
            evaluator.accumulate()

            # Get true positives and false positives
            precision = evaluator.eval['precision'][0, :, :, 0, 2]  # IoU=0.5, all areas, maxDets=100
            recall = evaluator.eval['recall'][0, :, 0, 2]  # IoU=0.5, all areas, maxDets=100

            # Compute FPPI
            # FP = (1 - precision) * TP
            # FPPI = FP / num_images
            tp = len([p for p in filtered_preds if any(
                evaluator.evalImgs[i]['dtMatches'][0][evaluator.evalImgs[i]['dtIds'].index(p['id'])] > 0
                for i in range(len(evaluator.evalImgs))
                if evaluator.evalImgs[i] and p['id'] in evaluator.evalImgs[i]['dtIds']
            )])
            fp = len(filtered_preds) - tp
            fppi = fp / num_images

            # Sensitivity = TP / (TP + FN) = TP / total_gt
            sensitivity = tp / total_gt if total_gt > 0 else 0.0

        except:
            fppi = len(filtered_preds) / num_images
            sensitivity = 0.0

        fppi_list.append(fppi)
        sensitivity_list.append(sensitivity)

    return np.array(fppi_list), np.array(sensitivity_list), confidence_thresholds


def benchmark_speed(
    model: YOLO,
    test_images: List[str],
    imgsz: int = 640,
    warmup_runs: int = 10,
    benchmark_runs: int = 100
) -> Dict[str, float]:
    """
    Benchmark inference speed.

    Args:
        model: YOLO model
        test_images: List of test image paths
        imgsz: Input image size
        warmup_runs: Number of warmup iterations
        benchmark_runs: Number of timed iterations

    Returns:
        Dict with FPS, latency, etc.
    """
    # Warmup
    for _ in range(warmup_runs):
        _ = model.predict(test_images[0], imgsz=imgsz, verbose=False)

    # Benchmark
    start_time = time.time()
    for img_path in test_images[:benchmark_runs]:
        _ = model.predict(img_path, imgsz=imgsz, verbose=False)
    end_time = time.time()

    total_time = end_time - start_time
    avg_latency = total_time / min(benchmark_runs, len(test_images))
    fps = 1.0 / avg_latency

    return {
        'fps': fps,
        'latency_ms': avg_latency * 1000,
        'total_time_s': total_time,
        'num_images': min(benchmark_runs, len(test_images))
    }


def main():
    ap = argparse.ArgumentParser(
        description='Comprehensive YOLO evaluation on FPUS23',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    ap.add_argument('--weights', type=str, required=True,
                    help='Path to trained YOLO weights (.pt file)')
    ap.add_argument('--data', type=str, required=True,
                    help='Path to data.yaml')
    ap.add_argument('--split', type=str, default='test',
                    choices=['val', 'test'],
                    help='Dataset split to evaluate on')
    ap.add_argument('--imgsz', type=int, default=768,
                    help='Input image size for inference')
    ap.add_argument('--conf-thr', type=float, default=0.001,
                    help='Confidence threshold for predictions')
    ap.add_argument('--iou-thr', type=float, default=0.6,
                    help='IoU threshold for NMS')
    ap.add_argument('--save-dir', type=str, default='results/eval',
                    help='Directory to save results')
    ap.add_argument('--device', type=str, default='',
                    help='CUDA device (e.g., 0) or cpu')
    ap.add_argument('--benchmark-speed', action='store_true',
                    help='Run speed benchmark')

    args = ap.parse_args()

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("📊 YOLO Evaluation on FPUS23")
    print("=" * 80)
    print(f"Weights:    {args.weights}")
    print(f"Data:       {args.data}")
    print(f"Split:      {args.split}")
    print(f"Image Size: {args.imgsz}px")
    print(f"Save Dir:   {save_dir}")
    print("=" * 80 + "\n")

    # Load model
    print("Loading model...")
    model = YOLO(args.weights)

    # Run evaluation
    print(f"\nEvaluating on {args.split} set...")
    results = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf_thr,
        iou=args.iou_thr,
        device=args.device,
        plots=True,
        save_json=True,
        verbose=True
    )

    # Extract metrics
    metrics = {
        'mAP50': float(results.box.map50),
        'mAP50-95': float(results.box.map),
        'mAP75': float(results.box.map75),
        'per_class_AP50': {CLASS_NAMES[i]: float(ap) for i, ap in enumerate(results.box.ap50)},
        'per_class_AP': {CLASS_NAMES[i]: float(ap) for i, ap in enumerate(results.box.ap)},
    }

    print("\n" + "=" * 80)
    print("📈 Overall Metrics:")
    print("=" * 80)
    print(f"mAP50:      {metrics['mAP50']:.4f}")
    print(f"mAP50-95:   {metrics['mAP50-95']:.4f}")
    print(f"mAP75:      {metrics['mAP75']:.4f}")
    print("\nPer-Class AP50:")
    for cls_name, ap in metrics['per_class_AP50'].items():
        print(f"  {cls_name:12s}: {ap:.4f}")
    print("=" * 80 + "\n")

    # Load predictions for advanced analysis
    pred_json = Path(results.save_dir) / 'predictions.json'
    if pred_json.exists():
        with open(pred_json) as f:
            predictions = json.load(f)

        # Load ground truth
        from ultralytics.data.utils import check_det_dataset
        data_dict = check_det_dataset(args.data)
        gt_json = Path(data_dict[args.split]).parent / 'annotations' / f'{args.split}.json'

        if gt_json.exists():
            coco_gt = COCO(str(gt_json))

            # Multi-IoU evaluation
            print("Computing multi-IoU metrics...")
            multi_iou = evaluate_at_multiple_ious(coco_gt, predictions, [0.5, 0.75, 0.9])
            metrics['multi_iou'] = multi_iou

            print("\n" + "=" * 80)
            print("📊 Multi-IoU Metrics:")
            print("=" * 80)
            for key, value in multi_iou.items():
                print(f"{key}: {value:.4f}")
            print("=" * 80 + "\n")

            # Size-stratified analysis
            print("Computing size-stratified metrics...")
            size_metrics = compute_size_stratified_metrics(coco_gt, predictions)
            size_metrics.to_csv(save_dir / 'size_stratified.csv', index=False)

            print("\n" + "=" * 80)
            print("📏 Size-Stratified Metrics:")
            print("=" * 80)
            print(size_metrics.to_string(index=False))
            print("=" * 80 + "\n")

            # FROC curve
            print("Computing FROC curve...")
            fppi, sensitivity, thresholds = compute_froc_curve(coco_gt, predictions)

            # Plot FROC
            plt.figure(figsize=(10, 6))
            plt.plot(fppi, sensitivity, 'b-', linewidth=2)
            plt.xlabel('False Positives Per Image (FPPI)', fontsize=12)
            plt.ylabel('Sensitivity (Recall)', fontsize=12)
            plt.title('Free-Response ROC (FROC) Curve', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_dir / 'froc_curve.png', dpi=300)
            print(f"FROC curve saved to: {save_dir / 'froc_curve.png'}")

            # Save FROC data
            froc_df = pd.DataFrame({
                'fppi': fppi,
                'sensitivity': sensitivity,
                'threshold': thresholds
            })
            froc_df.to_csv(save_dir / 'froc_data.csv', index=False)

    # Speed benchmark
    if args.benchmark_speed:
        print("\n" + "=" * 80)
        print("⚡ Speed Benchmark")
        print("=" * 80)

        # Get test images
        from ultralytics.data.utils import check_det_dataset
        data_dict = check_det_dataset(args.data)
        test_dir = Path(data_dict[args.split])
        test_images = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))

        if test_images:
            speed_metrics = benchmark_speed(model, [str(p) for p in test_images], args.imgsz)
            metrics['speed'] = speed_metrics

            print(f"FPS:         {speed_metrics['fps']:.2f}")
            print(f"Latency:     {speed_metrics['latency_ms']:.2f} ms")
            print(f"Tested on:   {speed_metrics['num_images']} images")
            print("=" * 80 + "\n")

    # Save all metrics
    with open(save_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Evaluation complete! Results saved to: {save_dir}")
    print(f"   - metrics.json: All numerical metrics")
    print(f"   - size_stratified.csv: Performance by object size")
    if pred_json.exists():
        print(f"   - froc_curve.png: Free-Response ROC plot")
        print(f"   - froc_data.csv: FROC data for plotting")
    print()


if __name__ == '__main__':
    main()
