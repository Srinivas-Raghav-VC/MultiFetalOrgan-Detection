#!/usr/bin/env python3
"""
Train Denoising Autoencoder for Ultrasound Preprocessing
=========================================================

Trains a SOTA denoising autoencoder for ultrasound speckle noise reduction.
Based on ArXiv 2403.02750v1 (March 2024) showing autoencoders outperform
traditional filters (median, Gaussian, bilateral) by 15-20%.

Expected improvement: +2-3% mAP over median blur preprocessing

Usage:
    python scripts/train_denoising_autoencoder.py \
        --images-dir fpus23_coco/images/train \
        --epochs 50 \
        --batch-size 16 \
        --device cuda

    # Resume from checkpoint
    python scripts/train_denoising_autoencoder.py \
        --resume checkpoints/denoiser_last.pt

    # Inference only
    python scripts/train_denoising_autoencoder.py \
        --inference \
        --checkpoint checkpoints/denoiser_best.pt \
        --input-image test_image.png

Author: FPUS23 custom YOLO implementation (Oct 2025)
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))

from denoising_autoencoder import (
    UltrasoundDenoiser,
    NoisyUltrasoundDataset,
    train_denoiser
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train denoising autoencoder for ultrasound preprocessing'
    )

    # Data
    parser.add_argument('--images-dir', type=str, default='fpus23_coco/images/train',
                       help='Directory containing ultrasound images')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split ratio (default: 0.1)')

    # Training
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Training device (cuda/cpu)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='DataLoader workers (default: 4)')

    # Model
    parser.add_argument('--base-channels', type=int, default=64,
                       help='Base number of channels (default: 64)')
    parser.add_argument('--noise-factor', type=float, default=0.3,
                       help='Speckle noise intensity (default: 0.3)')
    parser.add_argument('--image-size', type=int, default=640,
                       help='Training image size (default: 640)')

    # Checkpoints
    parser.add_argument('--save-dir', type=str, default='checkpoints/denoiser',
                       help='Checkpoint save directory')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')

    # Inference
    parser.add_argument('--inference', action='store_true',
                       help='Run inference only (no training)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint for inference')
    parser.add_argument('--input-image', type=str, default=None,
                       help='Input image for inference')

    return parser.parse_args()


def validate_paths(args):
    """Validate input paths"""
    images_dir = Path(args.images_dir)

    if not images_dir.exists():
        print(f"❌ Error: Images directory not found: {images_dir}")
        print("\nPlease ensure your FPUS23 images are available:")
        print("  fpus23_coco/images/train/")
        return False

    # Count images
    image_exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_count = sum(len(list(images_dir.glob(ext))) for ext in image_exts)

    if image_count == 0:
        print(f"❌ Error: No images found in {images_dir}")
        return False

    print(f"✅ Found {image_count} images in {images_dir}")
    return True


def create_dataloaders(args):
    """Create train and validation dataloaders"""
    print("\nCreating datasets...")

    # Create full dataset
    dataset = NoisyUltrasoundDataset(
        image_dir=args.images_dir,
        noise_factor=args.noise_factor,
        image_size=(args.image_size, args.image_size)
    )

    # Split into train/val
    val_size = int(args.val_split * len(dataset))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val:   {len(val_dataset)} images")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )

    return train_loader, val_loader


def visualize_denoising(model, val_loader, device, save_path):
    """
    Visualize denoising results on validation samples.

    Args:
        model: Trained denoiser model
        val_loader: Validation data loader
        device: Device to run on
        save_path: Where to save visualization
    """
    model.eval()

    # Get a batch from validation
    noisy, clean = next(iter(val_loader))
    noisy, clean = noisy.to(device), clean.to(device)

    # Denoise
    with torch.no_grad():
        denoised = model(noisy)

    # Convert to numpy (take first 4 samples)
    n_samples = min(4, len(noisy))
    noisy_np = noisy[:n_samples].cpu().numpy()
    clean_np = clean[:n_samples].cpu().numpy()
    denoised_np = denoised[:n_samples].cpu().numpy()

    # Create visualization
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3*n_samples))

    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_samples):
        # Noisy
        axes[i, 0].imshow(noisy_np[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title('Noisy Input')
        axes[i, 0].axis('off')

        # Denoised
        axes[i, 1].imshow(denoised_np[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('Denoised Output')
        axes[i, 1].axis('off')

        # Clean (ground truth)
        axes[i, 2].imshow(clean_np[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title('Clean Ground Truth')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Visualization saved: {save_path}")


def run_training(args):
    """Run training pipeline"""
    print("=" * 80)
    print("Training Denoising Autoencoder for FPUS23")
    print("=" * 80)

    # Validate paths
    if not validate_paths(args):
        return

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(args)

    # Create model
    print("\nInitializing model...")
    model = UltrasoundDenoiser(
        input_channels=1,
        base_channels=args.base_channels
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"  Resumed from epoch {start_epoch}")

    # Train
    print("\nStarting training...")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device:      {args.device}")
    print(f"  Save dir:    {save_dir}")

    train_denoiser(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        save_dir=save_dir
    )

    # Visualize results
    print("\nGenerating denoising visualization...")
    best_model_path = save_dir / 'denoiser_best.pt'
    if best_model_path.exists():
        model = UltrasoundDenoiser.load(str(best_model_path))
        model = model.to(args.device)
        viz_path = save_dir / 'denoising_results.png'
        visualize_denoising(model, val_loader, args.device, viz_path)

    print("\n" + "=" * 80)
    print("✅ Training complete!")
    print("=" * 80)
    print(f"\nBest model saved: {best_model_path}")
    print(f"\nExpected improvement: +2-3% mAP over median blur")
    print(f"\nNext steps:")
    print(f"  1. Integrate denoiser into YOLO preprocessing:")
    print(f"     denoiser = UltrasoundDenoiser.load('{best_model_path}')")
    print(f"     clean_image = denoiser.denoise(noisy_image)")
    print(f"  2. Train YOLO with denoised images")
    print(f"  3. Compare mAP with baseline (median blur)")


def run_inference(args):
    """Run inference on a single image"""
    print("=" * 80)
    print("Denoising Autoencoder - Inference Mode")
    print("=" * 80)

    # Check arguments
    if not args.checkpoint:
        print("❌ Error: --checkpoint required for inference")
        return

    if not args.input_image:
        print("❌ Error: --input-image required for inference")
        return

    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model = UltrasoundDenoiser.load(args.checkpoint)
    model = model.to(args.device)
    print("✅ Model loaded")

    # Load image
    print(f"\nLoading image: {args.input_image}")
    image = cv2.imread(args.input_image, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"❌ Error: Could not load image: {args.input_image}")
        return

    print(f"  Image shape: {image.shape}")

    # Denoise
    print("\nDenoising...")
    denoised = model.denoise(image)

    # Save result
    output_path = Path(args.input_image).parent / f"{Path(args.input_image).stem}_denoised.png"
    cv2.imwrite(str(output_path), denoised)
    print(f"✅ Denoised image saved: {output_path}")

    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original (Noisy)')
    axes[0].axis('off')

    axes[1].imshow(denoised, cmap='gray')
    axes[1].set_title('Denoised')
    axes[1].axis('off')

    plt.tight_layout()
    viz_path = Path(args.input_image).parent / f"{Path(args.input_image).stem}_comparison.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"✅ Comparison saved: {viz_path}")


def main():
    """Main entry point"""
    args = parse_args()

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  Warning: CUDA not available, using CPU")
        args.device = 'cpu'

    if args.inference:
        run_inference(args)
    else:
        run_training(args)


if __name__ == '__main__':
    main()
