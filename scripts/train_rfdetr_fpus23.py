#!/usr/bin/env python3
"""
Fine-tune RF-DETR on FPUS23 using the rfdetr Python package.

This script prepares an RF-DETR-friendly COCO directory layout and launches training.

RF-DETR expects:
  dataset_dir/
    train/  (images + _annotations.coco.json)
    valid/  (images + _annotations.coco.json)
    test/   (images + _annotations.coco.json)

Example:
  python scripts/train_rfdetr_fpus23.py \
    --project-root fpus23_complete_project \
    --epochs 20 --batch 4 --grad-accum 4 --lr 1e-4 --resolution 672

Checkpoint directory: <project-root>/models/rfdetr_finetuned
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path


def build_rfdetr_view(project_root: Path) -> Path:
    """Create the RF-DETR directory view from our COCO export.
    Copies images and writes _annotations.coco.json per split; renames val->valid.
    """
    coco_root = project_root / 'dataset' / 'fpus23_coco'
    out = project_root / 'dataset' / 'fpus23_coco_rfdetr'
    out.mkdir(parents=True, exist_ok=True)
    mapping = {
        'train': 'train',
        'val': 'valid',
        'test': 'test',
    }
    for src_split, dst_split in mapping.items():
        src_img_dir = coco_root / src_split
        dst_img_dir = out / dst_split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        # Copy images lazily (skip if exists)
        for p in src_img_dir.glob('*.png'):
            target = dst_img_dir / p.name
            if not target.exists():
                shutil.copy(p, target)
        # Copy JSON with expected name
        src_json = coco_root / f'{src_split}.json'
        dst_json = dst_img_dir / '_annotations.coco.json'
        shutil.copy(src_json, dst_json)
    return out


def main():
    ap = argparse.ArgumentParser(description='Train RF-DETR on FPUS23 (COCO)')
    ap.add_argument('--project-root', type=str, default='fpus23_complete_project')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--grad-accum', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--resolution', type=int, default=None, help='Input resolution (divisible by 56). Optional.')
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--ema', type=int, default=1)
    args = ap.parse_args()

    PROJECT = Path(args.project_root)
    dataset_dir = build_rfdetr_view(PROJECT)
    out_dir = PROJECT / 'models' / 'rfdetr_finetuned'
    out_dir.mkdir(parents=True, exist_ok=True)

    from rfdetr import RFDETRBase
    model = RFDETRBase()
    train_kwargs = dict(
        dataset_dir=str(dataset_dir),
        epochs=int(args.epochs),
        batch_size=int(args.batch),
        grad_accum_steps=int(args.grad_accum),
        lr=float(args.lr),
        output_dir=str(out_dir),
        weight_decay=float(args.weight_decay),
        use_ema=bool(args.ema),
    )
    if args.resolution:
        train_kwargs['resolution'] = int(args.resolution)
    print(f"[RF-DETR] Training with: {train_kwargs}")
    model.train(**train_kwargs)
    print(f"Saved RF-DETR fine-tuned checkpoints to {out_dir}")


if __name__ == '__main__':
    main()

