# FPUS23 End-to-End Colab Run

This guide runs the entire FPUS23 pipeline in Google Colab using a Drive ZIP.

## Requirements
- Google account and Colab access
- Your FPUS23 dataset ZIP on Drive (shared with "Anyone with the link can view")

## 1) Environment
```bash
!nvidia-smi
!pip -q install ultralytics==8.* gdown lxml pillow opencv-python-headless matplotlib tqdm pycocotools seaborn
```

## 2) Download and unzip dataset
```bash
!gdown --id 'YOUR_FILE_ID' -O /content/FPUS23_Dataset.zip
!unzip -q -o /content/FPUS23_Dataset.zip -d /content/
# Result: /content/FPUS23_Dataset or /content/FPUS23_Dataset/Dataset
```

## 3) Get this repository into Colab
```bash
!git clone https://github.com/Srinivas-Raghav-VC/MultiFetalOrgan-Detection.git /content/repo
%cd /content/repo
```

## 4) Prepare dataset (XML → YOLO → COCO)
```bash
%cd /content/repo/scripts
!python prepare_fpus23.py \
  --dataset-root "/content/FPUS23_Dataset" \
  --project-root /content/fpus23_complete_project \
  --group-split 1 \
  --dry-run 0
```

## 5) Verify dataset (counts + overlay sanity)
```bash
!python ../tools/verify_yolo_dataset.py \
  --data /content/fpus23_complete_project/dataset/fpus23_yolo/data.yaml \
  --split val \
  --vis-out /content/tmp/vis_val \
  --limit 24
```

## 6) Train (sanity, then full)
```bash
# Sanity 10 epochs
!python train_yolo_fpus23.py \
  --data /content/fpus23_complete_project/dataset/fpus23_yolo/data.yaml \
  --model yolo11s.pt \
  --epochs 10 \
  --batch 16 \
  --imgsz 896 \
  --lr 0.01 \
  --cls 0.8 \
  --rect \
  --workers 2 \
  --despeckle

# Full run
!python train_yolo_fpus23.py \
  --data /content/fpus23_complete_project/dataset/fpus23_yolo/data.yaml \
  --model yolo11s.pt \
  --epochs 100 \
  --batch 16 \
  --imgsz 896 \
  --lr 0.01 \
  --cls 0.8 \
  --rect \
  --workers 2 \
  --despeckle
```

## 7) Evaluate on test
```bash
!python eval_yolo_fpus23.py \
  --weights runs/detect/fpus23/weights/best.pt \
  --data /content/fpus23_complete_project/dataset/fpus23_yolo/data.yaml \
  --split test \
  --save-dir /content/results/yolo11s
```

## Notes
- `prepare_fpus23.py` writes a YAML‑safe `path:` (handles spaces/apostrophes).
- `yolo_to_coco` now copies images into split subfolders (train/val/test).
- On Windows, prefer `--workers 2`.

