#!/usr/bin/env python3
"""
Convert COCO (balanced) to YOLO format for training.

Typical usage on Colab after running the balancer:

  python tools/coco_to_yolo.py \
    --coco-json /content/fpus23_project/dataset/fpus23_coco/train_balanced.json \
    --images-dir /content/fpus23_project/dataset/fpus23_coco/images_balanced/train \
    --out-yolo-root /content/fpus23_project/dataset/fpus23_yolo_balanced \
    --orig-data-yaml /content/fpus23_project/dataset/fpus23_yolo/data.yaml

This will create:
  - out-yolo-root/images/train (copied balanced images)
  - out-yolo-root/labels/train (generated YOLO labels)
  - out-yolo-root/data.yaml (train uses balanced images; val/test from original yaml)

Notes:
  - COCO JSON is expected to contain width/height in the 'images' entries.
  - Categories should be 0-based ids consistent with original YOLO export.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil


def parse_simple_yaml(yaml_path: Path) -> dict:
    try:
        import yaml  # type: ignore
        return yaml.safe_load(Path(yaml_path).read_text()) or {}
    except Exception:
        meta = {}
        for ln in Path(yaml_path).read_text().splitlines():
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            if ':' in ln:
                k, v = ln.split(':', 1)
                meta[k.strip()] = v.strip().strip('\"\'')
        return meta


def write_data_yaml(out_yaml: Path, yolo_root: Path, orig_yaml: Path | None) -> None:
    root_str = str(yolo_root.resolve()).replace('\\', '/')
    path_line = f"path: \"{root_str}\"\n"

    train_line = "train: images/train\n"
    val_line = "val: images/val\n"
    test_line = "test: images/test\n"
    names_line = "names: ['Head','Abdomen','Arms','Legs']\n"  # default fallback
    nc_line = "nc: 4\n"

    if orig_yaml and Path(orig_yaml).exists():
        meta = parse_simple_yaml(Path(orig_yaml))
        # Respect original val/test if possible
        val_rel = meta.get('val', 'images/val')
        test_rel = meta.get('test', 'images/test')
        # Keep the relative paths (they will be resolved under new root at runtime)
        val_line = f"val: {val_rel}\n"
        test_line = f"test: {test_rel}\n"
        # Names/nc if present
        if 'names' in meta:
            import yaml  # type: ignore
            names_line = f"names: {yaml.safe_dump(meta['names'], default_flow_style=True).strip()}\n"
        if 'nc' in meta:
            nc_line = f"nc: {meta['nc']}\n"

    out_yaml.write_text(path_line + train_line + val_line + test_line + "\n" + nc_line + names_line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--coco-json', type=str, required=True, help='Path to COCO JSON (balanced)')
    ap.add_argument('--images-dir', type=str, required=True, help='Directory with images referenced by the JSON')
    ap.add_argument('--out-yolo-root', type=str, required=True, help='Output YOLO root directory')
    ap.add_argument('--orig-data-yaml', type=str, default=None, help='Original YOLO data.yaml to copy val/test')
    args = ap.parse_args()

    coco_json = Path(args.coco_json)
    img_dir = Path(args.images_dir)
    out_root = Path(args.out_yolo_root)
    out_images = out_root / 'images' / 'train'
    out_labels = out_root / 'labels' / 'train'

    if not coco_json.exists():
        raise FileNotFoundError(f'COCO JSON not found: {coco_json}')
    if not img_dir.exists():
        raise FileNotFoundError(f'Images dir not found: {img_dir}')

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    data = json.loads(coco_json.read_text())
    images = {img['id']: img for img in data.get('images', [])}
    ann_by_img = {}
    for ann in data.get('annotations', []):
        ann_by_img.setdefault(ann['image_id'], []).append(ann)

    # Copy images + write labels
    n_images = 0
    n_labels = 0
    for img_id, img in images.items():
        file_name = img['file_name']
        width = img.get('width')
        height = img.get('height')
        if not width or not height:
            # skip if dimensions unknown
            continue
        src = img_dir / file_name
        if not src.exists():
            # try lowercase extension fallback
            src_alt = img_dir / file_name.lower()
            if src_alt.exists():
                src = src_alt
            else:
                print(f"Warning: missing image {src}")
                continue
        dst = out_images / src.name
        if not dst.exists():
            try:
                shutil.copy(src, dst)
            except Exception:
                continue
        n_images += 1
        # labels
        lines = []
        for ann in ann_by_img.get(img_id, []):
            cid = int(ann['category_id'])
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
            xcen = (x + w / 2.0) / width
            ycen = (y + h / 2.0) / height
            wn = w / width
            hn = h / height
            lines.append(f"{cid} {xcen:.6f} {ycen:.6f} {wn:.6f} {hn:.6f}")
        if lines:
            (out_labels / (dst.stem + '.txt')).write_text('\n'.join(lines))
            n_labels += 1

    print(f"Copied images: {n_images}")
    print(f"Wrote label files: {n_labels}")

    # Write data.yaml
    out_yaml = out_root / 'data.yaml'
    write_data_yaml(out_yaml, out_root, Path(args.orig_data_yaml) if args.orig_data_yaml else None)
    print(f"Wrote {out_yaml}")


if __name__ == '__main__':
    main()

