# 🔧 Fix Dataset Preparation Issue in Colab

## ⚠️ Issue Detected

The dataset preparation found **0 images** because the XML structure doesn't match what the script expects.

---

## ✅ Quick Fix (Run in Colab)

### **Step 1:** Pull the latest code with `--project` fix

```python
# Re-clone with latest updates
!rm -rf /content/fpus23
!git clone https://github.com/Srinivas-Raghav-VC/MultiFetalOrgan-Detection.git /content/fpus23
```

### **Step 2:** Check dataset structure

```python
# Let's see what the actual structure is
!ls -la /content/FPUS23_Dataset/Dataset/
!find /content/FPUS23_Dataset -name "*.xml" -type f | head -10
!find /content/FPUS23_Dataset -name "*.png" -type f | head -10
```

### **Step 3:** Debug the dataset preparation

```python
# Run with verbose output to see what's happening
%cd /content/fpus23_project

!python /content/fpus23/scripts/prepare_fpus23.py \
    --dataset-root /content/FPUS23_Dataset/Dataset \
    --project-root /content/fpus23_project_debug \
    --group-split 1 \
    --group-depth 1
```

---

## 🔍 Alternative: Check Dataset Format

The FPUS23 dataset might be structured differently. Let's check:

```python
import os
from pathlib import Path

dataset_root = Path('/content/FPUS23_Dataset')

# Find all XMLs
xmls = list(dataset_root.rglob('*.xml'))
print(f"Found {len(xmls)} XML files")

# Show first few
for xml in xmls[:5]:
    print(f"  {xml}")

# Find all PNGs
pngs = list(dataset_root.rglob('*.png'))
print(f"\nFound {len(pngs)} PNG files")

# Show first few
for png in pngs[:5]:
    print(f"  {png}")

# Check directory structure
for item in dataset_root.rglob('*'):
    if item.is_dir() and 'annos' in item.name.lower():
        print(f"\nFound annotations dir: {item}")
        print(f"  Contents: {list(item.iterdir())[:5]}")
```

---

## 🚀 Temporary Workaround: Use Simpler Training

If dataset prep keeps failing, you can train directly with YOLO format if you have it:

```python
# Option 1: If you already have YOLO format data
%cd /content/fpus23

!python scripts/train_yolo_fpus23_phase1.py \
    --data /path/to/your/data.yaml \
    --model yolo11n.pt \
    --epochs 100 \
    --batch 16 \
    --imgsz 768 \
    --project /content/drive/MyDrive/FPUS23_YOLO_Training \
    --name fpus23_training
```

---

## 📊 Expected Dataset Structure

The script expects one of these formats:

### **Format 1: VOC XML (per-image)**
```
FPUS23_Dataset/
├── image1.png
├── image1.xml  (VOC format)
├── image2.png
├── image2.xml
└── ...
```

### **Format 2: CVAT Aggregated**
```
FPUS23_Dataset/
├── annos/
│   └── annotation/
│       ├── stream1/
│       │   └── annotations.xml  (CVAT format)
│       └── stream2/
│           └── annotations.xml
└── four_poses/
    ├── stream1/
    │   ├── image1.png
    │   └── image2.png
    └── stream2/
        ├── image3.png
        └── image4.png
```

---

## 🔨 Manual Fix Script

If the automatic prep doesn't work, here's a manual conversion:

```python
# Save this as /content/manual_dataset_prep.py
import os
import json
from pathlib import Path
from PIL import Image
import shutil

# Paths
dataset_root = Path('/content/FPUS23_Dataset')
output_root = Path('/content/fpus23_manual')

# Create YOLO structure
(output_root / 'images' / 'train').mkdir(parents=True, exist_ok=True)
(output_root / 'images' / 'val').mkdir(parents=True, exist_ok=True)
(output_root / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
(output_root / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

# Find all PNGs
all_images = list(dataset_root.rglob('*.png'))
print(f"Found {len(all_images)} images")

# Split 80/20
from sklearn.model_selection import train_test_split
train_imgs, val_imgs = train_test_split(all_images, test_size=0.2, random_state=42)

print(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}")

# Copy images (you'll need to create labels separately)
for split, imgs in [('train', train_imgs), ('val', val_imgs)]:
    for img_path in imgs:
        dest = output_root / 'images' / split / img_path.name
        shutil.copy(img_path, dest)

print("✅ Images copied. Now you need to add labels.")
```

Then run:
```python
!python /content/manual_dataset_prep.py
```

---

## 💡 Share Your Dataset Structure

To help debug, share the output of:

```python
!ls -R /content/FPUS23_Dataset/Dataset | head -50
```

This will show the actual structure so we can fix the prepare script.

---

## 🔄 After Fixing

Once dataset is ready, re-run training:

```python
%cd /content/fpus23

!python colab_train_with_drive.py
```

Or manually:

```python
!python scripts/train_yolo_fpus23_phase1.py \
    --data /content/fpus23_project/dataset/fpus23_yolo/data.yaml \
    --model yolo11n.pt \
    --epochs 100 \
    --batch 16 \
    --imgsz 768 \
    --project /content/drive/MyDrive/FPUS23_YOLO_Training \
    --name fpus23_colab_drive
```
