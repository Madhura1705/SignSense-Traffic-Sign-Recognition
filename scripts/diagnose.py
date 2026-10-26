"""
Diagnostic script for Phase 4 failure analysis.
"""
import os
import sys
from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from common import (CLASS_NAMES, NUM_CLASSES, get_datasets, load_model,
                    get_transforms, ARCHS, arch_of)

def main():
    print("=" * 60)
    print("PHASE 4 DIAGNOSTIC SUITE")
    print("=" * 60)

    # Determine dataset source
    if os.path.exists("data/gtsrb/Train") and os.path.exists("data/gtsrb/Test"):
        source = "folder"
        data_root = "data/gtsrb"
    else:
        source = "torchvision"
        data_root = "data"

    print(f"Loading dataset with source='{source}', root='{data_root}'...")
    train_ds, val_ds, test_ds = get_datasets(root=data_root, source=source)

    # 2a: Print lengths
    print(f"[2a] Dataset lengths:")
    print(f"     Train: {len(train_ds)}")
    print(f"     Val:   {len(val_ds)}")
    print(f"     Test:  {len(test_ds)}")

    # 2b: Label histograms of 3000 random train and 3000 test samples
    print("\n[2b] Label distributions:")
    rng = np.random.default_rng(42)

    n_train = min(3000, len(train_ds))
    train_idx = rng.choice(len(train_ds), n_train, replace=False)
    train_labels = [train_ds[int(i)][1] for i in train_idx]
    train_hist = Counter(train_labels)

    n_test = min(3000, len(test_ds))
    test_idx = rng.choice(len(test_ds), n_test, replace=False)
    test_labels = [test_ds[int(i)][1] for i in test_idx]
    test_hist = Counter(test_labels)

    print(f"     Sampled {n_train} TRAIN labels:")
    print(f"     Unique classes present: {len(train_hist)} / {NUM_CLASSES}")
    print(f"     Min class count: {min(train_hist.values())}, Max class count: {max(train_hist.values())}")
    print(f"     Top 5 most frequent in train sample: {train_hist.most_common(5)}")

    print(f"\n     Sampled {n_test} TEST labels:")
    print(f"     Unique classes present: {len(test_hist)} / {NUM_CLASSES}")
    print(f"     Min class count: {min(test_hist.values())}, Max class count: {max(test_hist.values())}")
    print(f"     Top 5 most frequent in test sample: {test_hist.most_common(5)}")

    # 2c: One batch of 64 train images
    print("\n[2c] Batch statistics and visual check:")
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    imgs, targets = next(iter(loader))
    print(f"     Batch shape: {imgs.shape}")
    print(f"     Min value:   {imgs.min().item():.4f}")
    print(f"     Max value:   {imgs.max().item():.4f}")
    print(f"     Mean value:  {imgs.mean().item():.4f}")
    print(f"     Std value:   {imgs.std().item():.4f}")

    os.makedirs("outputs", exist_ok=True)
    fig, axes = plt.subplots(8, 8, figsize=(16, 16))
    for ax, img, lbl in zip(axes.flat, imgs, targets):
        ax.imshow(img.permute(1, 2, 0).numpy())
        c = lbl.item()
        ax.set_title(f"[{c}] {CLASS_NAMES[c][:10]}", fontsize=7)
        ax.axis("off")
    fig.suptitle("Debug Batch: 64 Train Images with Labels", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig("outputs/debug_batch.png", dpi=130)
    plt.close(fig)
    print("     Saved outputs/debug_batch.png")

    # 2d: Test existing models in models/*.pt
    print("\n[2d] Evaluating models in models/:")
    model_names = ["cnn", "resnet18", "mobilenet", "cnn_robust"]
    eval_n = min(2000, len(test_ds))
    eval_idx = rng.choice(len(test_ds), eval_n, replace=False)
    test_sub = torch.utils.data.Subset(test_ds, [int(i) for i in eval_idx])
    test_loader = DataLoader(test_sub, batch_size=64, shuffle=False)

    for name in model_names:
        pt_path = os.path.join("models", f"{name}.pt")
        if not os.path.exists(pt_path):
            print(f"     {name:12s}: Not found at {pt_path}")
            continue
        try:
            model = load_model(name, pt_path, device="cpu")
            preds, true_y = [], []
            with torch.no_grad():
                for bx, by in test_loader:
                    out = model(bx)
                    preds.append(out.argmax(1))
                    true_y.append(by)
            p_cat = torch.cat(preds).numpy()
            y_cat = torch.cat(true_y).numpy()
            acc = (p_cat == y_cat).mean()
            pred_hist = Counter(p_cat)
            print(f"\n     [{name}] on {eval_n} test images:")
            print(f"       Accuracy: {acc*100:.2f}%")
            print(f"       Unique predicted classes: {len(pred_hist)}")
            print(f"       Top 5 predictions: {pred_hist.most_common(5)}")
        except Exception as e:
            print(f"     [{name}] error: {e}")

    print("=" * 60)

if __name__ == "__main__":
    main()
