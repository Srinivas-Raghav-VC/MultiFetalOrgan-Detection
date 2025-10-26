#!/usr/bin/env python3
"""
Fine-tune DINO-DETR (HuggingFace) on FPUS23 COCO.

Example:
  python scripts/train_dinodetr_fpus23.py \
    --project-root fpus23_complete_project \
    --model IDEA-Research/dino-detr-resnet-50 \
    --epochs 30 --batch 8 --lr 5e-5

Saves model to <project-root>/models/dinodetr_fpus23
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import Dataset
from PIL import Image

from transformers import (
    AutoModelForObjectDetection,
    AutoImageProcessor,
    TrainingArguments,
    Trainer,
)
from pycocotools.coco import COCO


CLASSES = ['Head', 'Abdomen', 'Arms', 'Legs']


class CocoFPUS23(Dataset):
    def __init__(self, images_dir: Path, annot_json: Path, processor: AutoImageProcessor):
        self.images_dir = images_dir
        self.coco = COCO(str(annot_json))
        self.ids = self.coco.getImgIds()
        self.processor = processor

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_meta = self.coco.loadImgs(self.ids[idx])[0]
        img_path = self.images_dir / img_meta['file_name']
        image = Image.open(img_path).convert('RGB')
        ann_ids = self.coco.getAnnIds(imgIds=[img_meta['id']])
        anns = self.coco.loadAnns(ann_ids)
        boxes = [a['bbox'] for a in anns]  # [x,y,w,h]
        classes = [a['category_id'] for a in anns]  # assumed 0..N-1
        inputs = self.processor(images=image, annotations={
            'image_id': img_meta['id'],
            'annotations': [{'bbox': b, 'category_id': c} for b, c in zip(boxes, classes)]
        }, return_tensors='pt')
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs


def main():
    ap = argparse.ArgumentParser(description='Fine-tune DINO-DETR on FPUS23')
    ap.add_argument('--project-root', type=str, default='fpus23_complete_project')
    ap.add_argument('--model', type=str, default='IDEA-Research/dino-detr-resnet-50')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=5e-5)
    # Optional explicit balanced paths
    ap.add_argument('--train-json', type=str, default=None, help='Optional path to train JSON (e.g., train_balanced.json)')
    ap.add_argument('--val-json', type=str, default=None, help='Optional path to val JSON')
    ap.add_argument('--train-images', type=str, default=None, help='Optional path to train images directory (e.g., images_balanced/train)')
    ap.add_argument('--val-images', type=str, default=None, help='Optional path to val images directory')
    args = ap.parse_args()

    ROOT = Path(args.project_root)
    COCO_ROOT = ROOT / 'dataset' / 'fpus23_coco'
    IMG_ROOT = ROOT / 'dataset' / 'fpus23_coco'
    OUT = ROOT / 'models' / 'dinodetr_fpus23'
    OUT.mkdir(parents=True, exist_ok=True)

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForObjectDetection.from_pretrained(args.model, num_labels=len(CLASSES), ignore_mismatched_sizes=True)

    train_json = Path(args.train_json) if args.train_json else (COCO_ROOT / 'train.json')
    val_json = Path(args.val_json) if args.val_json else (COCO_ROOT / 'val.json')
    train_images = Path(args.train_images) if args.train_images else (IMG_ROOT / 'train')
    val_images = Path(args.val_images) if args.val_images else (IMG_ROOT / 'val')

    train_ds = CocoFPUS23(train_images, train_json, processor)
    val_ds = CocoFPUS23(val_images, val_json, processor)

    args_tr = TrainingArguments(
        output_dir=str(OUT),
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=max(1, args.batch//2),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=1e-4,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        logging_steps=50,
        save_total_limit=3,
        remove_unused_columns=False,
    )

    def collate_fn(batch):
        keys = batch[0].keys()
        out = {}
        for k in keys:
            if k == 'labels':
                out[k] = [b[k] for b in batch]
            else:
                out[k] = torch.stack([b[k] for b in batch])
        return out

    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        tokenizer=processor,
    )

    # Auto-resume if a checkpoint exists
    trainer.train(resume_from_checkpoint=True)
    model.save_pretrained(str(OUT))
    processor.save_pretrained(str(OUT))
    print(f"Saved DINO-DETR fine-tuned model to {OUT}")


if __name__ == '__main__':
    main()
