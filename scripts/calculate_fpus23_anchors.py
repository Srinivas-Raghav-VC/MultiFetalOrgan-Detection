#!/usr/bin/env python3
"""
Calculate Optimal Anchors for FPUS23 Dataset
=============================================

Uses K-means clustering to find optimal anchor sizes based on your actual
bounding box distribution. Run this ONCE before training to get custom anchors.

What changed (engineering fix):
- Added argparse and support for reading YOLO data.yaml (preferred)
- Auto-detect COCO layout produced by prepare_fpus23.py (flat JSONs)
- Fallback to classic COCO layout (annotations/ + images/)

Expected improvement: +3-5% AP for Arms/Legs

Usage (preferred):
    python scripts/calculate_fpus23_anchors.py \
      --data /path/to/fpus23_yolo/data.yaml \
      --num-clusters 9

Output:
    - Optimal anchors printed to console
    - anchors.yaml file created for YOLO training
    - Visualization saved to outputs/anchor_analysis.png

Author: FPUS23 custom YOLO implementation (Oct 2025)
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from collections import Counter
import yaml
import argparse

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


def load_bbox_sizes(coco_json_path):
    """
    Extract all bounding box widths and heights from COCO JSON.

    Args:
        coco_json_path: Path to COCO format annotations JSON

    Returns:
        sizes: Array of (width, height) for all bboxes
        class_sizes: Dict mapping class_id to list of (width, height)
    """
    print(f"Loading bboxes from: {coco_json_path}")

    with open(coco_json_path, 'r') as f:
        data = json.load(f)

    sizes = []
    class_sizes = {cat['id']: [] for cat in data['categories']}

    for ann in data['annotations']:
        w, h = ann['bbox'][2], ann['bbox'][3]  # COCO format: [x, y, w, h]

        if w > 0 and h > 0:  # Skip invalid boxes
            sizes.append([w, h])
            class_id = ann['category_id']
            class_sizes[class_id].append([w, h])

    sizes = np.array(sizes)
    print(f"  Loaded {len(sizes)} bounding boxes")

    return sizes, class_sizes


def parse_simple_yaml(yaml_path: Path):
    """Parse minimal fields from Ultralytics-style data.yaml without requiring PyYAML.

    We still try yaml.safe_load if available, but gracefully fall back to a tiny parser.
    Returns dict with keys: path, train, val, test (if present).
    """
    try:
        import yaml as _yaml  # type: ignore
        with open(yaml_path, "r") as f:
            data = _yaml.safe_load(f) or {}
        return {
            "path": str(data.get("path", "")),
            "train": str(data.get("train", "")),
            "val": str(data.get("val", "")),
            "test": str(data.get("test", "")),
        }
    except Exception:
        # ultra-minimal parser
        meta = {"path": "", "train": "", "val": "", "test": ""}
        for ln in Path(yaml_path).read_text().splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            if ":" in ln:
                k, v = ln.split(":", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k in meta:
                    meta[k] = v
        return meta


def collect_yolo_bbox_sizes_from_yaml(data_yaml: Path):
    """Collect absolute bbox sizes (in pixels) by reading YOLO labels via data.yaml.

    - Reads root from `path:` and split rel paths (expects `train:`)
    - For each label file in labels/train, loads the paired image to get W,H
    - Converts normalized (w,h) → absolute pixels
    Returns: np.ndarray of shape (N, 2)
    """
    meta = parse_simple_yaml(data_yaml)
    if not meta.get("path"):
        raise FileNotFoundError(f"Could not parse 'path:' from {data_yaml}")
    root = Path(meta["path"]).expanduser()
    # Derive labels/train from train rel path
    train_rel = meta.get("train") or "images/train"
    train_images_dir = root / train_rel
    train_labels_dir = root / train_rel.replace("images/", "labels/")
    if not train_labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {train_labels_dir}")

    sizes = []
    n_labels = 0
    for lab in sorted(train_labels_dir.glob("*.txt")):
        n_labels += 1
        # Map to image path (prefer .png, fallback to common types)
        stem = lab.stem
        img_path = None
        for ext in (".png", ".jpg", ".jpeg", ".bmp"):
            p = train_images_dir / f"{stem}{ext}"
            if p.exists():
                img_path = p
                break
        if img_path is None or Image is None:
            continue
        try:
            with Image.open(img_path) as im:
                W, H = im.size
        except Exception:
            continue
        try:
            lines = [ln.strip() for ln in lab.read_text().splitlines() if ln.strip()]
        except Exception:
            continue
        for ln in lines:
            parts = ln.split()
            if len(parts) != 5:
                continue
            # cid = int(parts[0])
            try:
                w = float(parts[3]) * W
                h = float(parts[4]) * H
            except Exception:
                continue
            if w > 0 and h > 0:
                sizes.append([w, h])

    sizes = np.array(sizes, dtype=float)
    if sizes.size == 0:
        raise RuntimeError("No YOLO bounding boxes found to compute anchors.")
    print(f"  Parsed {len(sizes)} YOLO boxes from {n_labels} label files")
    return sizes


def calculate_anchors(sizes, n_anchors=9, n_clusters_per_head=3):
    """
    Use K-means clustering to find optimal anchor sizes.

    Args:
        sizes: Array of (width, height) bbox sizes
        n_anchors: Total number of anchors (default: 9)
        n_clusters_per_head: Anchors per detection head (default: 3)

    Returns:
        anchors_p2: Small anchors for P2 head (1/4 resolution)
        anchors_p3: Medium anchors for P3 head (1/8 resolution)
        anchors_p4: Large anchors for P4 head (1/16 resolution)
    """
    print(f"\nRunning K-means clustering with {n_anchors} clusters...")

    # K-means clustering
    kmeans = KMeans(n_clusters=n_anchors, random_state=42, n_init=10)
    kmeans.fit(sizes)

    # Sort by area (small to large)
    anchors = kmeans.cluster_centers_
    anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]

    # Round to integers
    anchors = np.round(anchors).astype(int)

    # Group into detection heads (P2, P3, P4)
    anchors_p2 = anchors[:n_clusters_per_head]  # Smallest (for Arms/Legs)
    anchors_p3 = anchors[n_clusters_per_head:2*n_clusters_per_head]  # Medium (for Head)
    anchors_p4 = anchors[2*n_clusters_per_head:]  # Largest (for Abdomen)

    return anchors_p2, anchors_p3, anchors_p4


def analyze_class_distributions(class_sizes, categories):
    """
    Analyze bounding box distributions per class.

    Args:
        class_sizes: Dict mapping class_id to list of (width, height)
        categories: List of category dicts from COCO JSON

    Returns:
        stats: Dict with per-class statistics
    """
    print("\nPer-Class Bounding Box Statistics:")
    print("=" * 80)

    stats = {}

    for cat in categories:
        class_id = cat['id']
        class_name = cat['name']

        if class_id not in class_sizes or len(class_sizes[class_id]) == 0:
            continue

        sizes = np.array(class_sizes[class_id])
        widths = sizes[:, 0]
        heights = sizes[:, 1]
        areas = widths * heights
        aspect_ratios = widths / (heights + 1e-6)

        stats[class_name] = {
            'count': len(sizes),
            'mean_width': np.mean(widths),
            'mean_height': np.mean(heights),
            'mean_area': np.mean(areas),
            'mean_aspect_ratio': np.mean(aspect_ratios),
            'std_width': np.std(widths),
            'std_height': np.std(heights),
        }

        print(f"\n{class_name} ({len(sizes)} boxes):")
        print(f"  Size (WxH):     {stats[class_name]['mean_width']:.1f} x {stats[class_name]['mean_height']:.1f} "
              f"± ({stats[class_name]['std_width']:.1f}, {stats[class_name]['std_height']:.1f})")
        print(f"  Area:           {stats[class_name]['mean_area']:.1f} px²")
        print(f"  Aspect ratio:   {stats[class_name]['mean_aspect_ratio']:.2f} (W/H)")

    return stats


def visualize_anchors(sizes, anchors_p2, anchors_p3, anchors_p4, output_path):
    """
    Visualize anchor boxes overlaid on bbox distribution.

    Args:
        sizes: Array of (width, height) for all bboxes
        anchors_p2, anchors_p3, anchors_p4: Anchor arrays
        output_path: Where to save visualization
    """
    print(f"\nCreating visualization...")

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Scatter plot of all bboxes
    ax.scatter(sizes[:, 0], sizes[:, 1], alpha=0.3, s=10, c='lightgray', label='All bboxes')

    # Plot anchors
    all_anchors = np.vstack([anchors_p2, anchors_p3, anchors_p4])
    colors = ['red', 'red', 'red', 'green', 'green', 'green', 'blue', 'blue', 'blue']
    labels = ['P2 (tiny)', None, None, 'P3 (small)', None, None, 'P4 (medium)', None, None]

    for i, (anchor, color, label) in enumerate(zip(all_anchors, colors, labels)):
        ax.scatter(anchor[0], anchor[1], s=300, c=color, marker='x', linewidths=3,
                  label=label, zorder=10)
        ax.text(anchor[0], anchor[1] + 2, f"{anchor[0]}x{anchor[1]}",
               ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Width (pixels)', fontsize=12)
    ax.set_ylabel('Height (pixels)', fontsize=12)
    ax.set_title('FPUS23 Optimal Anchors (K-means Clustering)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  Saved to: {output_path}")


def save_anchors_yaml(anchors_p2, anchors_p3, anchors_p4, output_path):
    """
    Save anchors in YAML format for YOLO training.

    Args:
        anchors_p2, anchors_p3, anchors_p4: Anchor arrays
        output_path: Where to save YAML file
    """
    # Flatten anchors to YOLO format: [w1, h1, w2, h2, ...]
    anchors_dict = {
        'anchors': [
            anchors_p2.flatten().tolist(),  # P2
            anchors_p3.flatten().tolist(),  # P3
            anchors_p4.flatten().tolist(),  # P4
        ]
    }

    with open(output_path, 'w') as f:
        yaml.dump(anchors_dict, f, default_flow_style=False)

    print(f"\n✅ Anchors saved to: {output_path}")
    print("\nTo use in training, add to your data.yaml or training config:")
    print("anchors:")
    print(f"  - {anchors_dict['anchors'][0]}  # P2 - tiny objects")
    print(f"  - {anchors_dict['anchors'][1]}  # P3 - small objects")
    print(f"  - {anchors_dict['anchors'][2]}  # P4 - medium objects")


def main():
    """Main execution"""
    print("=" * 80)
    print("FPUS23 Optimal Anchor Calculator")
    print("=" * 80)

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None,
                    help="Path to YOLO data.yaml (preferred)")
    ap.add_argument("--coco-root", type=str, default=None,
                    help="Path to COCO root (e.g., .../fpus23_coco)")
    ap.add_argument("--annotations", type=str, default=None,
                    help="Path to COCO train.json (overrides --coco-root)")
    ap.add_argument("--images", type=str, default=None,
                    help="Path to COCO train images (overrides --coco-root)")
    ap.add_argument("--num-clusters", type=int, default=9)
    ap.add_argument("--out", type=str, default="outputs/fpus23_anchors.yaml")
    args = ap.parse_args()

    output_dir = Path(args.out).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    sizes = None
    categories = None
    class_sizes = None

    # 1) Preferred: gather from YOLO labels via data.yaml
    if args.data:
        try:
            print(f"\n📄 Reading YOLO labels via {args.data}")
            sizes = collect_yolo_bbox_sizes_from_yaml(Path(args.data))
        except Exception as e:
            print(f"⚠️  Failed to parse YOLO labels from data.yaml: {e}")

        # Derive COCO root for (optional) per-class stats visualization
        try:
            meta = parse_simple_yaml(Path(args.data))
            yolo_root = Path(meta.get("path", "")).expanduser()
            coco_root_guess = yolo_root.parent / "fpus23_coco"
        except Exception:
            coco_root_guess = None
    else:
        coco_root_guess = None

    # 2) If sizes still None, try COCO JSON (flat layout created by prepare_fpus23.py)
    train_json = None
    if sizes is None:
        if args.annotations:
            train_json = Path(args.annotations)
        elif args.coco_root:
            cr = Path(args.coco_root)
            if (cr / "train.json").exists():
                train_json = cr / "train.json"
            elif (cr / "annotations" / "train.json").exists():
                train_json = cr / "annotations" / "train.json"
        elif coco_root_guess and (coco_root_guess / "train.json").exists():
            train_json = coco_root_guess / "train.json"
        elif coco_root_guess and (coco_root_guess / "annotations" / "train.json").exists():
            train_json = coco_root_guess / "annotations" / "train.json"

        if train_json and train_json.exists():
            sizes, class_sizes = load_bbox_sizes(train_json)
            with open(train_json, "r") as f:
                categories = json.load(f).get("categories", [])
        else:
            print("\n❌ Could not locate COCO annotations (train.json). Provide --data or --coco-root/--annotations.")
            return

    # Calculate optimal anchors
    anchors_p2, anchors_p3, anchors_p4 = calculate_anchors(sizes, n_anchors=args.num_clusters)

    print("\n" + "=" * 80)
    print("FPUS23 Optimal Anchors (K-means)")
    print("=" * 80)
    print(f"\nP2 (1/4 resolution - tiny objects like Arms/Legs):")
    for i, anchor in enumerate(anchors_p2):
        print(f"  Anchor {i+1}: {anchor[0]:3d} x {anchor[1]:3d}  (area: {anchor[0]*anchor[1]:5d} px²)")

    print(f"\nP3 (1/8 resolution - small objects like Head/organs):")
    for i, anchor in enumerate(anchors_p3):
        print(f"  Anchor {i+1}: {anchor[0]:3d} x {anchor[1]:3d}  (area: {anchor[0]*anchor[1]:5d} px²)")

    print(f"\nP4 (1/16 resolution - medium objects like Abdomen):")
    for i, anchor in enumerate(anchors_p4):
        print(f"  Anchor {i+1}: {anchor[0]:3d} x {anchor[1]:3d}  (area: {anchor[0]*anchor[1]:5d} px²)")

    # Save outputs
    anchor_yaml_path = Path(args.out)
    save_anchors_yaml(anchors_p2, anchors_p3, anchors_p4, anchor_yaml_path)

    # Visualize (only if we have sizes from somewhere)
    try:
        viz_path = output_dir / 'anchor_analysis.png'
        visualize_anchors(sizes, anchors_p2, anchors_p3, anchors_p4, viz_path)
    except Exception:
        pass

    print("\n" + "=" * 80)
    print("✅ Anchor calculation complete!")
    print("=" * 80)
    print(f"\nExpected improvement: +3-5% AP for small objects (Arms/Legs)")
    print(f"\nNext steps:")
    print(f"  1. Review the visualization: {output_dir/'anchor_analysis.png'}")
    print(f"  2. Use anchors YAML: {anchor_yaml_path}")
    print(f"  3. Train with --custom-anchors {anchor_yaml_path}")


if __name__ == '__main__':
    main()
