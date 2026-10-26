"""
Exploratory Data Analysis (EDA) for GTSRB dataset - Member 1 module.
Generates:
  - outputs/eda_class_distribution.png
  - outputs/eda_samples.png
  - outputs/eda_image_sizes.png
  - outputs/eda_augmentation.png
Prints: total images, min/max per class, imbalance ratio.
"""
import argparse
import glob
import os
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from common import CLASS_NAMES, NUM_CLASSES, get_transforms

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/gtsrb", help="Path to gtsrb root folder containing Train/ and Test/")
    args = parser.parse_args()

    train_dir = os.path.join(args.data, "Train")
    if not os.path.exists(train_dir):
        # Fallback to checking data/ directly
        if os.path.exists(os.path.join("data", "Train")):
            train_dir = os.path.join("data", "Train")
        else:
            print(f"Error: Train directory not found at {train_dir}")
            sys.exit(1)

    os.makedirs("outputs", exist_ok=True)

    class_counts = {}
    sample_images = {}
    widths = []
    heights = []

    for c in range(NUM_CLASSES):
        folder = os.path.join(train_dir, str(c))
        if not os.path.exists(folder):
            continue
        files = glob.glob(os.path.join(folder, "*.*"))
        class_counts[c] = len(files)
        if files:
            sample_images[c] = files[0]
            for f in files:
                try:
                    with Image.open(f) as img:
                        w, h = img.size
                        widths.append(w)
                        heights.append(h)
                except Exception:
                    pass

    total_images = sum(class_counts.values())
    min_class = min(class_counts, key=class_counts.get)
    max_class = max(class_counts, key=class_counts.get)
    min_count = class_counts[min_class]
    max_count = class_counts[max_class]
    imbalance_ratio = max_count / min_count if min_count > 0 else 0

    print("=" * 60)
    print("GTSRB EXPLORATORY DATA ANALYSIS (EDA) STATS")
    print("=" * 60)
    print(f"Total training images: {total_images}")
    print(f"Total classes:         {len(class_counts)}")
    print(f"Min per class:         {min_count} (Class {min_class}: {CLASS_NAMES[min_class]})")
    print(f"Max per class:         {max_count} (Class {max_class}: {CLASS_NAMES[max_class]})")
    print(f"Imbalance ratio:       {imbalance_ratio:.2f}x")
    print(f"Image width range:     {min(widths)}px - {max(widths)}px (mean {np.mean(widths):.1f}px)")
    print(f"Image height range:    {min(heights)}px - {max(heights)}px (mean {np.mean(heights):.1f}px)")
    print("=" * 60)

    # 1. Class distribution plot
    fig, ax = plt.subplots(figsize=(14, 10))
    counts = [class_counts.get(i, 0) for i in range(NUM_CLASSES)]
    y_pos = np.arange(NUM_CLASSES)
    bars = ax.barh(y_pos, counts, color="#3b82f6", edgecolor="#1d4ed8", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{i}: {CLASS_NAMES[i][:28]}" for i in range(NUM_CLASSES)], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Images", fontsize=11, fontweight="bold")
    ax.set_title(f"GTSRB Class Distribution (Imbalance Ratio: {imbalance_ratio:.1f}x)", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 15, bar.get_y() + bar.get_height() / 2, f"{int(w)}",
                va="center", fontsize=7, color="#1e293b")
    fig.tight_layout()
    fig.savefig("outputs/eda_class_distribution.png", dpi=140)
    plt.close(fig)
    print("Saved outputs/eda_class_distribution.png")

    # 2. One sample per class grid (43 tiles)
    cols = 7
    rows = int(np.ceil(NUM_CLASSES / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    for ax in axes.flat:
        ax.axis("off")
    for c in range(NUM_CLASSES):
        ax = axes.flat[c]
        if c in sample_images:
            img = Image.open(sample_images[c])
            ax.imshow(img)
            ax.set_title(f"[{c}] {CLASS_NAMES[c][:18]}", fontsize=7, fontweight="bold")
        ax.axis("off")
    fig.suptitle("GTSRB Classes: 1 Example Per Class (43 Classes)", fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig("outputs/eda_samples.png", dpi=140)
    plt.close(fig)
    print("Saved outputs/eda_samples.png")

    # 3. Image sizes histogram
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].hist(widths, bins=35, color="#10b981", edgecolor="#047857", alpha=0.8)
    ax[0].set_title(f"Original Widths (min: {min(widths)}, max: {max(widths)})", fontweight="bold")
    ax[0].set_xlabel("Width (pixels)")
    ax[0].set_ylabel("Count")
    ax[0].axvline(64, color="red", linestyle="--", label="Target resize: 64px")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].hist(heights, bins=35, color="#f59e0b", edgecolor="#b45309", alpha=0.8)
    ax[1].set_title(f"Original Heights (min: {min(heights)}, max: {max(heights)})", fontweight="bold")
    ax[1].set_xlabel("Height (pixels)")
    ax[1].set_ylabel("Count")
    ax[1].axvline(64, color="red", linestyle="--", label="Target resize: 64px")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    fig.suptitle("Original GTSRB Image Dimensions Before Resizing to 64x64", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig("outputs/eda_image_sizes.png", dpi=140)
    plt.close(fig)
    print("Saved outputs/eda_image_sizes.png")

    # 4. Augmentation demo plot (1 image + 8 augmented versions)
    # Pick a directional sign to visually show rotation/shear without flip
    aug_class = 14 if 14 in sample_images else list(sample_images.keys())[0]  # Stop sign or first available
    base_pil = Image.open(sample_images[aug_class]).convert("RGB")
    aug_tf = get_transforms(train=True)

    fig, axes = plt.subplots(1, 9, figsize=(16, 2.8))
    axes[0].imshow(base_pil.resize((64, 64)))
    axes[0].set_title("Original", fontsize=9, fontweight="bold", color="darkblue")
    axes[0].axis("off")

    for i in range(1, 9):
        aug_tensor = aug_tf(base_pil)  # 3x64x64 tensor in [0,1]
        aug_np = aug_tensor.permute(1, 2, 0).numpy()
        axes[i].imshow(aug_np)
        axes[i].set_title(f"Aug #{i}", fontsize=8)
        axes[i].axis("off")

    fig.suptitle(f"Data Augmentation Pipeline: Affine (rotation, shift, shear) + Color Jitter (NO flips)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig("outputs/eda_augmentation.png", dpi=140)
    plt.close(fig)
    print("Saved outputs/eda_augmentation.png")

if __name__ == "__main__":
    main()
