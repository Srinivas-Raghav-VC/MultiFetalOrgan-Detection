SOTA (Despeckle + Optimized) Notebooks

These three Colab notebooks run a streamlined, high‑leverage pipeline:
- Preprocess: despeckle (median blur, k=5) all splits to reduce ultrasound speckle noise
- Train with “optimized” settings (cos LR, cls weight, deterministic loaders)
- Save every epoch to Drive and auto‑resume

Run order (recommended):
1) `notebooks_sota/yolo_sota_all_in_one_v2.ipynb`  ← single, comprehensive pipeline (fixed JSON)
2) `notebooks_sota/yolo_smart_ensemble.ipynb`    (optional extra boost)
3) `notebooks_sota/rtdetr_sota_despeckle.ipynb`
4) `notebooks_sota/dinodetr_sota_despeckle.ipynb`

Outputs
- YOLO → `/content/drive/MyDrive/FPUS23_runs/fpus23_yolo_sota_*`
- RT‑DETR → `/content/drive/MyDrive/FPUS23_runs/rtdetr_sota_despeckled`
- DINO‑DETR → `/content/drive/MyDrive/FPUS23_runs/dinodetr_sota_despeckled`

Notes
- Despeckling uses `New folder/scripts/preprocess_fpus23_despeckle.py` (median blur k=5). Adjust k=3 or k=7 if needed.
- If you already created `/content/fpus23_project/dataset/fpus23_yolo_despeckled`, preprocessing will simply reuse it.
- The smart ensemble notebook trains two specialists (Abdomen+Head and Arms+Legs) and fuses with Weighted Boxes Fusion (WBF).
