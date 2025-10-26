#!/usr/bin/env python3
"""
Google Colab Setup Script for FPUS23 YOLO Training
===================================================

This script automates the entire setup process on Google Colab:
1. Downloads dataset from Google Drive
2. Extracts and validates dataset structure
3. Clones GitHub repository
4. Installs dependencies
5. Prepares dataset (CVAT XML -> YOLO -> COCO)
6. Runs preprocessing steps (anchors, balancing)
7. Starts YOLO training

Usage in Colab:
    !python colab_setup.py --github-repo <your-repo-url> --drive-file-id 1LL-r2hNiP6C190UBSE4v1FFCF3OQT9N3

Author: FPUS23 Colab Setup (Oct 2025)
"""

import argparse
import subprocess
import sys
from pathlib import Path
import os
import shutil


def run_command(cmd, description, check=True, shell=False):
    """Execute shell command with error handling"""
    print(f"\n{'='*80}")
    print(f"▶ {description}")
    print(f"{'='*80}")
    print(f"Command: {cmd if isinstance(cmd, str) else ' '.join(cmd)}")
    print()

    try:
        if shell:
            result = subprocess.run(cmd, shell=True, check=check, text=True,
                                   capture_output=False)
        else:
            result = subprocess.run(cmd, check=check, text=True,
                                   capture_output=False)
        print(f"✅ {description} - COMPLETED")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        if check:
            raise
        return e


def download_from_gdrive(file_id, output_path):
    """Download file from Google Drive using gdown"""
    print(f"\n📥 Downloading dataset from Google Drive...")
    print(f"   File ID: {file_id}")
    print(f"   Output:  {output_path}")

    # Install gdown if not available
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        subprocess.run([sys.executable, "-m", "pip", "install", "gdown", "-q"],
                      check=True)
        import gdown

    # Download
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(output_path), quiet=False)

    print(f"✅ Dataset downloaded: {output_path}")


def extract_dataset(zip_path, extract_to):
    """Extract ZIP dataset"""
    print(f"\n📦 Extracting dataset...")
    print(f"   From: {zip_path}")
    print(f"   To:   {extract_to}")

    import zipfile

    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get total file count for progress
        total_files = len(zip_ref.namelist())
        print(f"   Extracting {total_files} files...")

        zip_ref.extractall(extract_to)

    print(f"✅ Dataset extracted to: {extract_to}")
    return extract_to


def validate_dataset(dataset_path):
    """Validate FPUS23 dataset structure"""
    print(f"\n🔍 Validating dataset structure...")

    dataset_path = Path(dataset_path)

    # Check for expected directories
    required = ['annos', 'four_poses']
    found = []
    missing = []

    for req in required:
        if (dataset_path / req).exists():
            found.append(req)
        else:
            missing.append(req)

    # Count XML and PNG files
    xml_count = len(list(dataset_path.rglob('*.xml')))
    png_count = len(list(dataset_path.rglob('*.png')))

    print(f"\n   Directories found: {found}")
    if missing:
        print(f"   ⚠️  Directories missing: {missing}")
    print(f"   XML files: {xml_count}")
    print(f"   PNG files: {png_count}")

    if xml_count < 10:
        print(f"\n   ⚠️  Warning: Very few XML annotations found ({xml_count})")
        print(f"      Expected structure:")
        print(f"        FPUS23_Dataset/")
        print(f"          Dataset/")
        print(f"            annos/")
        print(f"            four_poses/")

        # Try to find the actual dataset directory
        print(f"\n   🔎 Searching for dataset in subdirectories...")
        for subdir in dataset_path.rglob('*'):
            if subdir.is_dir():
                sub_xml = len(list(subdir.rglob('*.xml')))
                if sub_xml >= 10:
                    print(f"      ✅ Found dataset at: {subdir} ({sub_xml} XMLs)")
                    return subdir

    return dataset_path


def setup_colab_environment():
    """Install required packages for Colab"""
    print(f"\n📦 Installing dependencies...")

    packages = [
        "ultralytics",  # YOLO
        "lxml",         # XML parsing
        "scikit-learn", # K-means for anchors
        "matplotlib",   # Plotting
        "tqdm",         # Progress bars
        "opencv-python" # Image processing
    ]

    for package in packages:
        print(f"   Installing {package}...")
        subprocess.run([sys.executable, "-m", "pip", "install", package, "-q"],
                      check=True)

    print(f"✅ All dependencies installed")


def clone_github_repo(repo_url, target_dir):
    """Clone GitHub repository"""
    print(f"\n📂 Cloning GitHub repository...")
    print(f"   URL:    {repo_url}")
    print(f"   Target: {target_dir}")

    target_dir = Path(target_dir)

    if target_dir.exists():
        print(f"   ⚠️  Directory already exists, removing...")
        shutil.rmtree(target_dir)

    subprocess.run(["git", "clone", repo_url, str(target_dir)], check=True)

    print(f"✅ Repository cloned to: {target_dir}")


def prepare_dataset(dataset_root, project_root, scripts_dir):
    """Run prepare_fpus23.py to convert XML -> YOLO -> COCO"""
    print(f"\n🔧 Preparing dataset (XML -> YOLO -> COCO)...")

    prepare_script = Path(scripts_dir) / "prepare_fpus23.py"

    if not prepare_script.exists():
        print(f"   ❌ Error: prepare_fpus23.py not found at {prepare_script}")
        return False

    cmd = [
        sys.executable,
        str(prepare_script),
        "--dataset-root", str(dataset_root),
        "--project-root", str(project_root),
        "--group-split", "1",
        "--group-depth", "1"
    ]

    run_command(cmd, "Dataset preparation (XML -> YOLO -> COCO)")

    return True


def calculate_anchors(scripts_dir, data_yaml):
    """Run calculate_fpus23_anchors.py"""
    print(f"\n⚓ Calculating custom anchors...")

    anchor_script = Path(scripts_dir) / "calculate_fpus23_anchors.py"

    if not anchor_script.exists():
        print(f"   ⚠️  Anchor script not found, skipping...")
        return None

    cmd = [
        sys.executable,
        str(anchor_script),
        "--data", str(data_yaml),
        "--num-clusters", "9"
    ]

    try:
        run_command(cmd, "Custom anchor calculation")
        return Path("outputs/fpus23_anchors.yaml")
    except:
        print(f"   ⚠️  Anchor calculation failed, will use default COCO anchors")
        return None


def balance_dataset(scripts_dir):
    """Run balance_fpus23_dataset.py"""
    print(f"\n⚖️  Balancing dataset...")

    balance_script = Path(scripts_dir) / "balance_fpus23_dataset.py"

    if not balance_script.exists():
        print(f"   ⚠️  Balance script not found, skipping...")
        return False

    try:
        run_command([sys.executable, str(balance_script)],
                   "Dataset balancing")
        return True
    except:
        print(f"   ⚠️  Dataset balancing failed, will use unbalanced data")
        return False


def start_training(scripts_dir, data_yaml, custom_anchors=None, balanced=False):
    """Start YOLO training"""
    print(f"\n🚀 Starting YOLO training...")

    train_script = Path(scripts_dir) / "train_yolo_fpus23_phase1.py"

    if not train_script.exists():
        print(f"   ❌ Training script not found at {train_script}")
        return False

    cmd = [
        sys.executable,
        str(train_script),
        "--data", str(data_yaml),
        "--model", "yolo11n.pt",
        "--epochs", "100",
        "--batch", "16",
        "--imgsz", "768",
        "--name", "fpus23_colab_phase1"
    ]

    if custom_anchors and Path(custom_anchors).exists():
        cmd.extend(["--custom-anchors", str(custom_anchors)])

    if balanced:
        balanced_json = "fpus23_coco/annotations/train_balanced.json"
        if Path(balanced_json).exists():
            cmd.extend(["--balanced-data", balanced_json])

    run_command(cmd, "YOLO Phase 1 Training")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Setup FPUS23 YOLO training on Google Colab"
    )

    parser.add_argument(
        "--github-repo",
        type=str,
        required=True,
        help="GitHub repository URL (e.g., https://github.com/user/repo.git)"
    )

    parser.add_argument(
        "--drive-file-id",
        type=str,
        default="1LL-r2hNiP6C190UBSE4v1FFCF3OQT9N3",
        help="Google Drive file ID for dataset ZIP"
    )

    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip dataset download (use existing)"
    )

    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training (just setup)"
    )

    args = parser.parse_args()

    print("="*80)
    print("FPUS23 YOLO TRAINING - GOOGLE COLAB SETUP")
    print("="*80)

    # Define paths (Colab-friendly)
    colab_root = Path("/content")
    repo_dir = colab_root / "fpus23_yolo_training"
    dataset_zip = colab_root / "FPUS23_Dataset.zip"
    dataset_dir = colab_root / "FPUS23_Dataset"
    project_dir = colab_root / "fpus23_project"

    # Step 1: Install dependencies
    setup_colab_environment()

    # Step 2: Clone GitHub repo
    clone_github_repo(args.github_repo, repo_dir)

    # Step 3: Download dataset
    if not args.skip_download:
        if not dataset_zip.exists():
            download_from_gdrive(args.drive_file_id, dataset_zip)
        else:
            print(f"✅ Dataset ZIP already exists: {dataset_zip}")

        # Step 4: Extract dataset
        if not dataset_dir.exists():
            extract_dataset(dataset_zip, dataset_dir)
        else:
            print(f"✅ Dataset already extracted: {dataset_dir}")

    # Step 5: Validate dataset
    actual_dataset = validate_dataset(dataset_dir)

    # Step 6: Change to working directory
    os.chdir(project_dir)
    print(f"\n📁 Changed working directory to: {project_dir}")

    # Step 7: Prepare dataset
    scripts_dir = repo_dir / "scripts"
    if prepare_dataset(actual_dataset, project_dir, scripts_dir):
        print(f"✅ Dataset preparation complete")
    else:
        print(f"❌ Dataset preparation failed")
        sys.exit(1)

    # Step 8: Calculate anchors
    data_yaml = project_dir / "dataset" / "fpus23_yolo" / "data.yaml"
    anchors_yaml = calculate_anchors(scripts_dir, data_yaml)

    # Step 9: Balance dataset
    balanced = balance_dataset(scripts_dir)

    # Step 10: Start training
    if not args.skip_training:
        start_training(scripts_dir, data_yaml, anchors_yaml, balanced)
    else:
        print(f"\n⏭️  Skipping training (--skip-training flag)")
        print(f"\nTo start training manually, run:")
        print(f"  cd {project_dir}")
        print(f"  python {scripts_dir}/train_yolo_fpus23_phase1.py \\")
        print(f"    --data {data_yaml} \\")
        print(f"    --epochs 100 --batch 16 --imgsz 768")

    print("\n" + "="*80)
    print("✅ COLAB SETUP COMPLETE!")
    print("="*80)
    print(f"\nProject directory: {project_dir}")
    print(f"Scripts directory: {scripts_dir}")
    print(f"Dataset YOLO: {project_dir}/dataset/fpus23_yolo")
    print(f"Dataset COCO: {project_dir}/dataset/fpus23_coco")
    print(f"\nTraining results will be saved to: {project_dir}/runs/detect/")


if __name__ == "__main__":
    main()
