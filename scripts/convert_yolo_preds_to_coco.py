#!/usr/bin/env python3
"""
Convert YOLO txt predictions to COCO detection results JSON using GT image IDs.

Assumes one txt per image in <pred-dir>, with lines:
  <cls> <x_center> <y_center> <width> <height> <conf>
All coords normalized to [0,1].

Usage:
  python scripts/convert_yolo_preds_to_coco.py \
    --gt fpus23_complete_project/dataset/fpus23_coco/test.json \
    --images fpus23_complete_project/dataset/fpus23_coco/test \
    --pred-dir path/to/yolo/preds \
    --out preds_coco.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from PIL import Image
from pycocotools.coco import COCO


def main():
    ap = argparse.ArgumentParser(description='Convert YOLO txt predictions to COCO results')
    ap.add_argument('--gt', type=str, required=True)
    ap.add_argument('--images', type=str, required=True)
    ap.add_argument('--pred-dir', type=str, required=True)
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    coco = COCO(args.gt)
    name2id = {img['file_name']: img['id'] for img in coco.dataset['images']}
    img_dir = Path(args.images)
    pred_dir = Path(args.pred_dir)
    out = []
    for img_name, img_id in name2id.items():
        lab = pred_dir / (Path(img_name).stem + '.txt')
        img_path = img_dir / img_name
        if not lab.exists() or not img_path.exists():
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        for line in lab.read_text().splitlines():
            p = line.strip().split()
            if len(p) < 6:
                # if no conf present, skip
                continue
            cls = int(p[0]); xc = float(p[1])*w; yc = float(p[2])*h
            bw = float(p[3])*w; bh = float(p[4])*h; conf = float(p[5])
            x = xc - bw/2; y = yc - bh/2
            out.append({'image_id': int(img_id), 'category_id': int(cls), 'bbox': [float(x), float(y), float(bw), float(bh)], 'score': float(conf)})
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out} with {len(out)} detections")


if __name__ == '__main__':
    main()

