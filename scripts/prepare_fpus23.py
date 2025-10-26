#!/usr/bin/env python3
"""
FPUS23 Preparation: CVAT XML -> YOLO -> COCO with leakage-safe splits.

Usage:
  python scripts/prepare_fpus23.py \
    --dataset-root FPUS23 \
    --project-root fpus23_complete_project \
    --group-split 1 \
    --group-depth 1

Outputs under <project-root>/dataset:
  - fpus23_yolo/{images,labels}/{train,val,test} + data.yaml
  - fpus23_coco/{train.json,val.json,test.json} and copied images
  - plots/dataset_class_distribution.png
"""
from __future__ import annotations
import argparse
import os
import shutil
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional, Tuple, List, Dict

from lxml import etree
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


CLASSES = ['Head', 'Abdomen', 'Arms', 'Legs']
CLASS_TO_ID = {n: i for i, n in enumerate(CLASSES)}


def normalize_label(name: str) -> Optional[str]:
    if not name:
        return None
    t = name.strip().lower()
    mapping = {
        'head': 'Head',
        'abdomen': 'Abdomen', 'belly': 'Abdomen',
        'arm': 'Arms', 'arms': 'Arms',
        'leg': 'Legs', 'legs': 'Legs', 'lower limb': 'Legs', 'lower limbs': 'Legs'
    }
    return mapping.get(t)


def find_dataset_root(candidate: Path) -> Path:
    # If candidate contains many XMLs, assume it is the root
    if candidate.exists():
        n_xml = sum(1 for _ in candidate.rglob('*.xml'))
        if n_xml >= 10:
            return candidate
    # Probe common siblings
    for c in [Path('FPUS23'), Path('FPUS23_Dataset')/ 'Dataset', Path('FPUS23_Dataset')]:
        if c.exists():
            n_xml = sum(1 for _ in c.rglob('*.xml'))
            if n_xml >= 10:
                return c
    raise FileNotFoundError(f"Could not find FPUS23 dataset with XMLs starting from {candidate}")


def collect_pairs(root: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    for xml in root.rglob('*.xml'):
        img = xml.with_suffix('.png')
        if img.exists():
            pairs.append((xml, img))
    return pairs


def find_cvat_aggregated_xmls(root: Path) -> List[Path]:
    """Return CVAT aggregated XMLs from both annos/ and boxes/ directories"""
    xs = []
    # Check annos/annotation/<stream>/annotations.xml
    if (root / 'annos' / 'annotation').exists():
        xs.extend(list((root / 'annos' / 'annotation').rglob('annotations.xml')))
    # Check boxes/annotation/<stream>/annotations.xml
    if (root / 'boxes' / 'annotation').exists():
        xs.extend(list((root / 'boxes' / 'annotation').rglob('annotations.xml')))
    return xs


def map_images_dir_for_xml(xml_path: Path) -> Optional[Path]:
    """Given .../annos/annotation/<stream>/annotations.xml OR .../boxes/annotation/<stream>/annotations.xml,
    map to images dir .../four_poses/<stream>/"""
    parts = list(xml_path.parts)
    try:
        # Try to find 'annos' or 'boxes' in path
        annotation_dir = None
        if 'annos' in parts:
            i = parts.index('annos')
            annotation_dir = 'annos'
        elif 'boxes' in parts:
            i = parts.index('boxes')
            annotation_dir = 'boxes'

        if annotation_dir and parts[i+1] == 'annotation':
            stream = parts[i+2]
            base = Path(*parts[:i])
            candidate = base / 'four_poses' / stream
            return candidate if candidate.exists() else None
    except Exception:
        return None
    return None


def parse_cvat_aggregated(xml_file: Path) -> List[Dict]:
    """Parse CVAT aggregated annotations.xml -> list of {'name': str, 'boxes': [(label, xtl, ytl, xbr, ybr)]} per image."""
    out = []
    rt = etree.parse(str(xml_file)).getroot()
    for img in rt.findall('.//image'):
        name = img.attrib.get('name') or img.attrib.get('id')
        boxes = []
        for b in img.findall('.//box'):
            label = b.attrib.get('label', '')
            try:
                xtl = float(b.attrib.get('xtl')); ytl = float(b.attrib.get('ytl'))
                xbr = float(b.attrib.get('xbr')); ybr = float(b.attrib.get('ybr'))
            except Exception:
                continue
            boxes.append((label, xtl, ytl, xbr, ybr))
        out.append({'name': name, 'boxes': boxes})
    return out


def group_id_from_path(p: Path, depth: int) -> str:
    cur = p.parent
    for _ in range(max(0, depth-1)):
        if cur.parent:
            cur = cur.parent
    return cur.name


def split_pairs(pairs: List[Tuple[Path, Path]], group_split: bool, depth: int):
    import random
    random.seed(42)
    if group_split:
        from collections import defaultdict
        g = defaultdict(list)
        for xml, img in pairs:
            gid = group_id_from_path(xml, depth)
            g[gid].append((xml, img))
        groups = list(g.keys())
        random.shuffle(groups)
        n = len(groups)
        n_tr = int(0.8*n); n_va = int(0.1*n)
        tr = set(groups[:n_tr]); va = set(groups[n_tr:n_tr+n_va])
        train = [x for key in tr for x in g[key]]
        val = [x for key in va for x in g[key]]
        test = [x for key in set(groups) - tr - va for x in g[key]]
    else:
        random.shuffle(pairs)
        n = len(pairs)
        n_tr = int(0.8*n); n_va = int(0.1*n)
        train = pairs[:n_tr]
        val = pairs[n_tr:n_tr+n_va]
        test = pairs[n_tr+n_va:]
    return train, val, test


def convert_xml_to_yolo(xml_file: Path, output_label: Path, img_w: int, img_h: int,
                        drop_stats: dict) -> bool:
    try:
        root = etree.parse(str(xml_file)).getroot()
        yolo_lines = []
        voc_objects = root.findall('.//object')
        if voc_objects:
            for obj in voc_objects:
                name_elem = obj.find('name')
                canon = normalize_label(name_elem.text if name_elem is not None else '')
                if canon is None or canon not in CLASS_TO_ID:
                    continue
                cid = CLASS_TO_ID[canon]
                b = obj.find('bndbox')
                if b is None:
                    continue
                xmin = float(b.find('xmin').text); ymin = float(b.find('ymin').text)
                xmax = float(b.find('xmax').text); ymax = float(b.find('ymax').text)
                xcen = ((xmin+xmax)/2)/img_w; ycen = ((ymin+ymax)/2)/img_h
                w = (xmax-xmin)/img_w; h = (ymax-ymin)/img_h
                if w <= 0 or h <= 0:
                    drop_stats['zero'] += 1; continue
                pre = (xcen, ycen, w, h)
                xcen = max(0, min(1, xcen)); ycen = max(0, min(1, ycen))
                w = max(0, min(1, w)); h = max(0, min(1, h))
                if pre != (xcen, ycen, w, h):
                    drop_stats['oob'] += 1
                yolo_lines.append(f"{cid} {xcen:.6f} {ycen:.6f} {w:.6f} {h:.6f}")
        else:
            for box in root.findall('.//image/box'):
                canon = normalize_label(box.attrib.get('label', ''))
                if canon is None or canon not in CLASS_TO_ID:
                    continue
                cid = CLASS_TO_ID[canon]
                xtl = float(box.attrib.get('xtl')); ytl = float(box.attrib.get('ytl'))
                xbr = float(box.attrib.get('xbr')); ybr = float(box.attrib.get('ybr'))
                xcen = ((xtl+xbr)/2)/img_w; ycen = ((ytl+ybr)/2)/img_h
                w = (xbr-xtl)/img_w; h = (ybr-ytl)/img_h
                if w <= 0 or h <= 0:
                    drop_stats['zero'] += 1; continue
                pre = (xcen, ycen, w, h)
                xcen = max(0, min(1, xcen)); ycen = max(0, min(1, ycen))
                w = max(0, min(1, w)); h = max(0, min(1, h))
                if pre != (xcen, ycen, w, h):
                    drop_stats['oob'] += 1
                yolo_lines.append(f"{cid} {xcen:.6f} {ycen:.6f} {w:.6f} {h:.6f}")
        if yolo_lines:
            output_label.parent.mkdir(parents=True, exist_ok=True)
            output_label.write_text("\n".join(yolo_lines))
            return True
        return False
    except Exception:
        return False


def yolo_to_coco(yolo_root: Path, split: str, out_json: Path, one_based: bool = False):
    import json
    from PIL import Image
    images_dir = yolo_root / 'images' / split
    labels_dir = yolo_root / 'labels' / split
    cat_ids = list(range(1, len(CLASSES)+1)) if one_based else list(range(len(CLASSES)))
    id_map = {i: cid for i, cid in enumerate(cat_ids)}
    coco = {
        'images': [],
        'annotations': [],
        'categories': [{'id': id_map[i], 'name': n, 'supercategory': 'fetal'} for i, n in enumerate(CLASSES)]
    }
    img_id = 1; ann_id = 1
    for img_file in sorted(images_dir.glob('*.png')):
        with Image.open(img_file) as im:
            w, h = im.size
        coco['images'].append({'id': img_id, 'file_name': img_file.name, 'width': w, 'height': h})
        dest = out_json.parent / img_file.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy(img_file, dest)
        lab = labels_dir / (img_file.stem + '.txt')
        if lab.exists():
            for line in lab.read_text().splitlines():
                p = line.strip().split()
                if len(p) != 5:
                    continue
                cls = int(p[0]); xcen = float(p[1])*w; ycen = float(p[2])*h
                bw = float(p[3])*w; bh = float(p[4])*h
                x = xcen - bw/2; y = ycen - bh/2
                coco['annotations'].append({'id': ann_id, 'image_id': img_id, 'category_id': id_map[cls],
                                            'bbox': [x, y, bw, bh], 'area': bw*bh, 'iscrowd': 0})
                ann_id += 1
        img_id += 1
    out_json.write_text(__import__('json').dumps(coco, indent=2))


def main():
    ap = argparse.ArgumentParser(description='Prepare FPUS23 dataset (XML->YOLO->COCO)')
    ap.add_argument('--dataset-root', type=str, default='FPUS23', help='Root folder containing PNGs and XMLs')
    ap.add_argument('--project-root', type=str, default='fpus23_complete_project', help='Output project root')
    ap.add_argument('--group-split', type=int, default=1, help='Leakage-safe grouped split (default=1)')
    ap.add_argument('--group-depth', type=int, default=1, help='How many directory levels to group by (default=1)')
    ap.add_argument('--coco-one-based', type=int, default=0, help='Use 1-based category IDs in COCO (default 0 uses 0-based)')
    # Optional minority-class intensity augmentations (bbox-safe)
    ap.add_argument('--augment-minority', type=int, default=0, help='Enable minority-class intensity augs for train (default=0)')
    ap.add_argument('--minority-classes', type=str, default='Head,Legs', help='Comma-separated classes to augment more')
    ap.add_argument('--aug-ratio', type=float, default=0.3, help='Target extra samples as fraction of minority images (default 0.3)')
    ap.add_argument('--clahe', type=int, default=1, help='Use CLAHE (default=1)')
    ap.add_argument('--gamma', type=int, default=1, help='Use gamma jitter (default=1)')
    ap.add_argument('--speckle', type=int, default=1, help='Use speckle noise (default=1)')
    args = ap.parse_args()

    DATASET = find_dataset_root(Path(args.dataset_root))
    PROJECT = Path(args.project_root)
    (PROJECT / 'dataset').mkdir(parents=True, exist_ok=True)
    (PROJECT / 'plots').mkdir(parents=True, exist_ok=True)

    # Stats
    xmls = sorted(DATASET.rglob('*.xml'))
    imgs = sorted(DATASET.rglob('*.png'))
    print(f"Dataset: {DATASET}")
    print(f" - PNG images: {len(imgs)}")
    print(f" - XML files : {len(xmls)}")

    # Class distribution from XMLs
    class_counts = defaultdict(int); unknown = Counter(); total = 0
    for xf in tqdm(xmls, desc='Scanning labels'):
        try:
            rt = etree.parse(str(xf)).getroot()
            voc = rt.findall('.//object')
            if voc:
                for obj in voc:
                    nm = obj.find('name')
                    if nm is not None:
                        c = normalize_label(nm.text)
                        if c in CLASSES:
                            class_counts[c] += 1; total += 1
                        elif nm.text:
                            unknown[nm.text.strip()] += 1
            else:
                for b in rt.findall('.//image/box'):
                    c = normalize_label(b.attrib.get('label', ''))
                    if c in CLASSES:
                        class_counts[c] += 1; total += 1
                    else:
                        lb = b.attrib.get('label', '')
                        if lb:
                            unknown[lb.strip()] += 1
        except Exception:
            continue
    print("Class distribution:")
    for k, v in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        pct = v/total*100 if total else 0
        print(f" - {k}: {v} ({pct:.1f}%)")
    # Plot
    if class_counts:
        plt.figure(figsize=(6,3.5))
        ks, vs = list(class_counts.keys()), list(class_counts.values())
        plt.bar(ks, vs, color='steelblue')
        plt.title('FPUS23 Class Distribution (XML)')
        plt.tight_layout()
        plt.savefig(PROJECT / 'plots' / 'dataset_class_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
    if unknown:
        print('Unmapped labels (top 10):')
        for lb, ct in unknown.most_common(10):
            print(f" - '{lb}': {ct}")

    # Attempt simple xml<->image pairing; fall back to CVAT aggregated if too few pairs
    pairs = collect_pairs(DATASET)
    YOLO_ROOT = PROJECT / 'dataset' / 'fpus23_yolo'
    for s in ['train','val','test']:
        (YOLO_ROOT / 'images' / s).mkdir(parents=True, exist_ok=True)
        (YOLO_ROOT / 'labels' / s).mkdir(parents=True, exist_ok=True)
    drop = {'zero': 0, 'oob': 0, 'written': 0}

    if len(pairs) > 1000:
        # Normal per-image XML path
        train, val, test = split_pairs(pairs, bool(args.group_split), args.group_depth)
        print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

        def process(split_pairs, split_name):
            ok = 0
            for xml, img in tqdm(split_pairs, desc=f'YOLO {split_name}'):
                try:
                    with Image.open(img) as im:
                        w, h = im.size
                except Exception:
                    continue
                shutil.copy(img, YOLO_ROOT / 'images' / split_name / img.name)
                out_lab = YOLO_ROOT / 'labels' / split_name / (img.stem + '.txt')
                if convert_xml_to_yolo(xml, out_lab, w, h, drop):
                    ok += 1; drop['written'] += 1
            return ok

        tr_ok = process(train, 'train'); va_ok = process(val, 'val'); te_ok = process(test, 'test')
        print(f"YOLO conversion: train {tr_ok}/{len(train)}, val {va_ok}/{len(val)}, test {te_ok}/{len(test)}")
    else:
        # CVAT aggregated mode
        print("Detected CVAT aggregated annotations; building list per image ...")
        xmls = find_cvat_aggregated_xmls(DATASET)
        items = []  # list of (image_path, [(label, xtl, ytl, xbr, ybr)])
        for xf in xmls:
            images_dir = map_images_dir_for_xml(xf)
            if images_dir is None:
                continue
            recs = parse_cvat_aggregated(xf)
            for r in recs:
                img_name = Path(r['name']).name
                img_path = images_dir / img_name
                if img_path.exists() and r['boxes']:
                    items.append((img_path, r['boxes']))
        print(f"Found {len(items)} annotated images in CVAT aggregated xmls")
        # Split per image
        import random
        random.seed(42)
        random.shuffle(items)
        n = len(items); n_tr = int(0.8*n); n_va = int(0.1*n)
        splits = {
            'train': items[:n_tr],
            'val': items[n_tr:n_tr+n_va],
            'test': items[n_tr+n_va:]
        }
        # Write images and labels
        for split_name, lst in splits.items():
            ok = 0
            for img_path, boxes in tqdm(lst, desc=f'YOLO {split_name}'):
                try:
                    with Image.open(img_path) as im:
                        w, h = im.size
                except Exception:
                    continue
                dest_img = YOLO_ROOT / 'images' / split_name / img_path.name
                if not dest_img.exists():
                    shutil.copy(img_path, dest_img)
                yolo_lines = []
                for (label, xtl, ytl, xbr, ybr) in boxes:
                    canon = normalize_label(label)
                    if canon not in CLASSES:
                        continue
                    cid = CLASS_TO_ID[canon]
                    xcen = ((xtl + xbr)/2)/w; ycen = ((ytl + ybr)/2)/h
                    bw = (xbr - xtl)/w; bh = (ybr - ytl)/h
                    if bw <= 0 or bh <= 0:
                        drop['zero'] += 1; continue
                    pre = (xcen, ycen, bw, bh)
                    xcen = max(0, min(1, xcen)); ycen = max(0, min(1, ycen))
                    bw = max(0, min(1, bw)); bh = max(0, min(1, bh))
                    if pre != (xcen, ycen, bw, bh):
                        drop['oob'] += 1
                    yolo_lines.append(f"{cid} {xcen:.6f} {ycen:.6f} {bw:.6f} {bh:.6f}")
                if yolo_lines:
                    out_lab = YOLO_ROOT / 'labels' / split_name / (img_path.stem + '.txt')
                    out_lab.write_text('\n'.join(yolo_lines))
                    drop['written'] += 1; ok += 1
            print(f"YOLO conversion ({split_name}): {ok}/{len(lst)} annotations")
    print(f" - Dropped zero-area: {drop['zero']} | Clamped OOB: {drop['oob']} | Labels written: {drop['written']}")

    # Optional minority-class augmentation (intensity only, bbox-safe)
    if args.augment_minority:
        print("Applying minority-class intensity augmentations on train split ...")
        min_cls = [c.strip() for c in args.minority_classes.split(',') if c.strip() in CLASSES]
        # Build per-image class presence for train
        labels_dir = YOLO_ROOT / 'labels' / 'train'
        images_dir = YOLO_ROOT / 'images' / 'train'
        imgs_by_cls = {c: [] for c in min_cls}
        for lab in labels_dir.glob('*.txt'):
            try:
                with open(lab) as f:
                    lines = [ln.strip().split()[0] for ln in f if ln.strip()]
                if not lines:
                    continue
                classes_in = {CLASSES[int(cid)] for cid in lines}
                for c in min_cls:
                    if c in classes_in:
                        imgs_by_cls[c].append(lab.stem)
            except Exception:
                continue
        # Aug functions
        def aug_clahe(img):
            if len(img.shape) == 3 and img.shape[2] == 3:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                cla = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                cl = cla.apply(l)
                limg = cv2.merge((cl,a,b))
                return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            else:
                cla = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                return cla.apply(img)
        def aug_gamma(img):
            gamma = float(np.random.uniform(0.8, 1.2))
            inv = 1.0/gamma
            table = np.array([((i/255.0)**inv)*255 for i in np.arange(256)]).astype('uint8')
            if img.ndim == 2:
                return cv2.LUT(img, table)
            return cv2.LUT(img, table)
        def aug_speckle(img):
            noise = np.random.normal(0, 0.02, img.shape).astype(np.float32)
            out = img.astype(np.float32)/255.0
            out = out + out*noise
            out = np.clip(out, 0.0, 1.0)
            return (out*255.0).astype(np.uint8)
        aug_fns = []
        if args.clahe: aug_fns.append(aug_clahe)
        if args.gamma: aug_fns.append(aug_gamma)
        if args.speckle: aug_fns.append(aug_speckle)
        # For each minority class, add aug copies up to ratio
        for c in min_cls:
            stems = imgs_by_cls.get(c, [])
            target_add = int(len(stems) * args.aug_ratio)
            added = 0
            for stem in stems:
                if added >= target_add:
                    break
                img_path = images_dir / f"{stem}.png"
                lab_path = labels_dir / f"{stem}.txt"
                if not img_path.exists() or not lab_path.exists():
                    continue
                img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                fn = np.random.choice(aug_fns) if aug_fns else None
                if fn is None:
                    break
                aug_img = fn(img)
                out_stem = f"{stem}_aug_{c.lower()}_{added}"
                out_img = images_dir / f"{out_stem}.png"
                out_lab = labels_dir / f"{out_stem}.txt"
                cv2.imwrite(str(out_img), aug_img)
                shutil.copy(lab_path, out_lab)
                added += 1
            print(f"Augmented class {c}: +{added} images (target {target_add})")

    # data.yaml
    data_yaml = (
        f"path: {YOLO_ROOT}\n"
        "train: images/train\nval: images/val\ntest: images/test\n\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n"
    )
    (YOLO_ROOT / 'data.yaml').write_text(data_yaml)
    print(f"Wrote {YOLO_ROOT/'data.yaml'}")

    # COCO
    COCO_ROOT = PROJECT / 'dataset' / 'fpus23_coco'
    COCO_ROOT.mkdir(parents=True, exist_ok=True)
    for s in ['train','val','test']:
        (COCO_ROOT / s).mkdir(parents=True, exist_ok=True)
    for s in ['train','val','test']:
        yolo_to_coco(YOLO_ROOT, s, COCO_ROOT / f'{s}.json', one_based=bool(args.coco_one_based))
        print(f"Wrote {COCO_ROOT/(s+'.json')}")


if __name__ == '__main__':
    main()
