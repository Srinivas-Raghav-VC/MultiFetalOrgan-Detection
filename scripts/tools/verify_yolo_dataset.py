#!/usr/bin/env python3
"""
Quick sanity checker for a YOLO dataset produced by prepare_fpus23.py.

What it does:
- Parses data.yaml to find the YOLO root and split folders
- Counts images and labels per split; reports missing label/image issues
- Optionally renders a small grid of overlays for visual spot‑check

Usage:
  python tools/verify_yolo_dataset.py \
    --data path/to/fpus23_yolo/data.yaml \
    --split val \
    --vis-out tmp/vis_val --limit 16
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import sys

try:
    from PIL import Image, ImageDraw
except Exception:
    Image = None


def parse_simple_yaml(fp: Path) -> dict:
    """Parse the minimal fields we use from Ultralytics-style data.yaml without PyYAML.

    Expects lines like:
      path: /abs/path/to/fpus23_yolo
      train: images/train
      val: images/val
      test: images/test
      names: ['Head', 'Abdomen', 'Arms', 'Legs']
    """
    text = fp.read_text(encoding='utf-8')
    def get_line(key: str) -> str | None:
        m = re.search(rf"^{key}\s*:\s*(.+)$", text, flags=re.M)
        return m.group(1).strip() if m else None

    root = get_line('path')
    train = get_line('train')
    val = get_line('val')
    test = get_line('test')
    names = get_line('names')
    return {
        'path': root.strip('"\'') if root else None,
        'train': train.strip('"\'') if train else None,
        'val': val.strip('"\'') if val else None,
        'test': test.strip('"\'') if test else None,
        'names': names,
    }


def load_yolo_split(root: Path, split_rel: str) -> tuple[list[Path], list[Path]]:
    images = sorted((root / split_rel).glob('*.png')) + \
             sorted((root / split_rel).glob('*.jpg')) + \
             sorted((root / split_rel).glob('*.jpeg'))
    # Convert e.g. images/train -> labels/train
    labels_rel = split_rel.replace('images/', 'labels/')
    labels = [root / labels_rel / (p.stem + '.txt') for p in images]
    return images, labels


def overlay_bbox(img: Image.Image, lines: list[str]) -> Image.Image:
    draw = ImageDraw.Draw(img)
    W, H = img.size
    for ln in lines:
        p = ln.strip().split()
        if len(p) != 5:
            continue
        _, x, y, w, h = p
        x = float(x) * W; y = float(y) * H
        w = float(w) * W; h = float(h) * H
        x0 = x - w/2; y0 = y - h/2; x1 = x + w/2; y1 = y + h/2
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    ap.add_argument('--vis-out', type=str, default=None)
    ap.add_argument('--limit', type=int, default=16)
    args = ap.parse_args()

    data_yaml = Path(args.data)
    meta = parse_simple_yaml(data_yaml)
    if not meta.get('path'):
        print('Could not parse path: from data.yaml. Aborting.', file=sys.stderr)
        sys.exit(2)

    root = Path(meta['path'])
    split_rel = {'train': meta['train'], 'val': meta['val'], 'test': meta['test']}[args.split]
    if not split_rel:
        print(f'Missing split path for {args.split} in data.yaml', file=sys.stderr)
        sys.exit(2)

    images, labels = load_yolo_split(root, split_rel)
    missing_labels = [p for p, l in zip(images, labels) if not l.exists()]
    print(f"Split: {args.split}")
    print(f" - Images: {len(images)}")
    print(f" - Labels: {len(labels)}")
    print(f" - Missing label files: {len(missing_labels)}")

    if args.vis_out and Image is not None:
        out_dir = Path(args.vis_out)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, (img_p, lab_p) in enumerate(zip(images, labels)):
            if i >= args.limit:
                break
            try:
                im = Image.open(img_p).convert('RGB')
                if lab_p.exists():
                    lines = lab_p.read_text().splitlines()
                    im = overlay_bbox(im, lines)
                im.save(out_dir / img_p.name)
            except Exception:
                continue
        print(f" - Wrote overlays to: {out_dir}")
    elif args.vis_out and Image is None:
        print('PIL not available; skipping visualization')


if __name__ == '__main__':
    main()

