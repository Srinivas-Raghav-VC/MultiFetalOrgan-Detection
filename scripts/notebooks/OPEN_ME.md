FPUS23 Colab Notebooks (v2)

Use the v2 notebooks below. They include:
- Drive mount + robust git clone/pull to `/content/fpus23`
- Per-epoch checkpoints saved to Drive
- Auto-resume on rerun (Ultralytics / scripts handle resuming)
- End-of-run compact summary cells

YOLO:
- `notebooks/yolo/baseline_v2.ipynb` — YOLO11 n/s/m on original dataset
- `notebooks/yolo/optimized_v2.ipynb` — YOLO11 n/s/m on balanced dataset
- `notebooks/yolo/noise_v2.ipynb` — YOLO11 n/s/m with stronger noise augs (optional denoiser)

RT‑DETR:
- `notebooks/rtdetr/baseline_v2.ipynb`
- `notebooks/rtdetr/optimized_v2.ipynb`
- `notebooks/rtdetr/noise_v2.ipynb`

DINO‑DETR:
- `notebooks/dinodetr/baseline_v2.ipynb`
- `notebooks/dinodetr/optimized_v2.ipynb`
- `notebooks/dinodetr/noise_v2.ipynb`

Notes
- Place your dataset at `/content/FPUS23_Dataset/Dataset` on Colab, or run the provided prepare script via the notebooks.
- YOLO runs are saved under `/content/drive/MyDrive/FPUS23_runs/<run_name>` with per-epoch checkpoints.
- RT‑DETR/DINO notebooks save metrics to `<run>/metrics_val.json` and resume automatically.
