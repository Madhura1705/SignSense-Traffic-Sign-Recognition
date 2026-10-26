"""
Verify trained models locally on CPU:
Load each models/*.pt, predict 500 random test images from data/,
print accuracy and number of distinct predicted classes (> 35).
"""
import os
import sys
from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from common import CLASS_NAMES, NUM_CLASSES, get_datasets, load_model

def main():
    print("=" * 65)
    print("MODEL VERIFICATION ON CPU (500 TEST IMAGES)")
    print("=" * 65)

    if os.path.exists("data/gtsrb/Test"):
        source, data_root = "folder", "data/gtsrb"
    else:
        source, data_root = "torchvision", "data"

    _, _, test_ds = get_datasets(root=data_root, source=source)
    print(f"Total test set images: {len(test_ds)}")

    rng = np.random.default_rng(2026)
    n_sample = 500
    indices = rng.choice(len(test_ds), n_sample, replace=False)
    sub_ds = Subset(test_ds, [int(i) for i in indices])
    loader = DataLoader(sub_ds, batch_size=64, shuffle=False)

    models_to_test = ["cnn", "resnet18", "mobilenet", "cnn_robust"]

    for name in models_to_test:
        pt_path = os.path.join("models", f"{name}.pt")
        if not os.path.exists(pt_path):
            print(f"FAILED: {pt_path} not found.")
            continue

        model = load_model(name, pt_path, device="cpu")
        preds, targets = [], []
        with torch.no_grad():
            for bx, by in loader:
                out = model(bx)
                preds.append(out.argmax(1))
                targets.append(by)

        p = torch.cat(preds).numpy()
        y = torch.cat(targets).numpy()
        acc = float((p == y).mean())
        distinct_classes = len(set(p))

        status = "PASS" if acc >= 0.90 and distinct_classes >= 35 else "CHECK"
        print(f"[{status}] {name:12s} | Test Accuracy: {acc*100:6.2f}% | Distinct Predicted Classes: {distinct_classes:2d}/43")

    print("=" * 65)

if __name__ == "__main__":
    main()
