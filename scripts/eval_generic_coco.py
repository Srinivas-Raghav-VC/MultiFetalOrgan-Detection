#!/usr/bin/env python3
"""
Evaluate arbitrary COCO-format detection predictions against FPUS23 COCO ground truth.

Usage:
  python scripts/eval_generic_coco.py \
    --gt fpus23_complete_project/dataset/fpus23_coco/test.json \
    --pred my_model_predictions.json

Outputs AP50-95 / AP50, and saves optional per-class AP.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def main():
    ap = argparse.ArgumentParser(description='COCOeval for arbitrary predictions')
    ap.add_argument('--gt', type=str, required=True, help='Ground truth COCO JSON')
    ap.add_argument('--pred', type=str, required=True, help='Predictions COCO results JSON')
    ap.add_argument('--save', type=str, default=None, help='Optional path to save summary JSON')
    args = ap.parse_args()

    coco_gt = COCO(args.gt)
    preds = json.loads(Path(args.pred).read_text())
    coco_dt = coco_gt.loadRes(preds)
    ev = COCOeval(coco_gt, coco_dt, iouType='bbox')
    ev.evaluate(); ev.accumulate(); ev.summarize()
    ap5095 = float(ev.stats[0]*100.0); ap50 = float(ev.stats[1]*100.0)
    print(f"AP50-95 {ap5095:.2f} | AP50 {ap50:.2f}")
    per_class = {}
    for cat in coco_gt.loadCats(coco_gt.getCatIds()):
        e = COCOeval(coco_gt, coco_dt, iouType='bbox'); e.params.catIds=[cat['id']]
        e.evaluate(); e.accumulate(); e.summarize()
        per_class[cat['name']] = {'AP50-95': float(e.stats[0]*100.0), 'AP50': float(e.stats[1]*100.0)}
    out = {'AP50-95': ap5095, 'AP50': ap50, 'per_class': per_class}
    if args.save:
        Path(args.save).write_text(json.dumps(out, indent=2))
        print(f"Wrote {args.save}")


if __name__ == '__main__':
    main()

