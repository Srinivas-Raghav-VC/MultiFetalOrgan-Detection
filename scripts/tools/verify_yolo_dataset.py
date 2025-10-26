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


def load_yolo_split(root: Path, split_rel: str) -> tuple[list[Path], list[Path], list[Path]]:
    img_dir = root / split_rel
    # Case-insensitive globbing by enumerating common variants
    patterns = ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']
    images: list[Path] = []
    for pat in patterns:
        images += sorted(img_dir.glob(pat))
    # Convert e.g. images/train -> labels/train
    labels_rel = split_rel.replace('images/', 'labels/')
    lab_dir = root / labels_rel
    label_files = sorted(lab_dir.glob('*.txt'))
    # Map labels for images (missing image labels will be ignored in this mapping)
    img_to_lab = [root / labels_rel / (p.stem + '.txt') for p in images]
    return images, img_to_lab, label_files


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

    print(f"YOLO root: {root}")
    print(f"Split rel : {split_rel}")
    images, mapped_labels, all_labels = load_yolo_split(root, split_rel)
    img_dir = root / split_rel
    lab_dir = root / split_rel.replace('images/', 'labels/')
    # Extension breakdown for debugging
    by_ext = {}
    for p in images:
        by_ext[p.suffix] = by_ext.get(p.suffix, 0) + 1
    missing_labels = [p for p, l in zip(images, mapped_labels) if not l.exists()]
    # Also detect labels that don't have a corresponding image
    image_stems = {p.stem for p in images}
    orphan_labels = [l for l in all_labels if l.stem not in image_stems]
    print(f"Split: {args.split}")
    print(f" - Images: {len(images)}")
    print(f" - Labels: {len(all_labels)}")
    print(f" - Missing label files: {len(missing_labels)}")
    print(f" - Orphan labels (no image): {len(orphan_labels)}")
    print(f" - Image dir: {img_dir}")
    print(f" - Label dir: {lab_dir}")
    if by_ext:
        print(" - Image ext breakdown:")
        for k, v in sorted(by_ext.items()):
            print(f"    {k}: {v}")

    if args.vis_out and Image is not None:
        out_dir = Path(args.vis_out)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, (img_p, lab_p) in enumerate(zip(images, mapped_labels)):
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
