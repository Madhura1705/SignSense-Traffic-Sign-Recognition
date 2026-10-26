"""
Train + evaluate traffic sign classifiers on GTSRB.

Examples
  python train.py                                  # both models, auto-download GTSRB
  python train.py --models cnn --epochs-cnn 5      # quick run, custom CNN only
  python train.py --source folder --data data/gtsrb  # Kaggle folder layout
  python train.py --models mobilenet               # edge model only
  python train.py --models cnn_robust              # custom CNN with adversarial training (defence)
  python train.py --source fake --epochs-cnn 1 --epochs-resnet 1 --no-pretrained  # smoke test

Model names: cnn, resnet18, mobilenet, and <name>_robust for adversarially-trained versions.
Results are MERGED into outputs/metrics.json, so models can be trained one at a time
(e.g. by different team members) without overwriting each other.

Outputs
  models/<name>.pt                  trained weights
  outputs/<name>_curves.png         loss / accuracy curves
  outputs/<name>_confusion.png      43x43 confusion matrix
  outputs/<name>_report.txt         per-class precision / recall / F1
  outputs/<name>_misclassified.png  hardest mistakes
  outputs/<name>_gradcam.png        Grad-CAM explanations
  outputs/fgsm_robustness.png       accuracy vs adversarial noise
  outputs/comparison.csv / metrics.json
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from common import (CLASS_NAMES, NUM_CLASSES, IMG_SIZE, get_datasets, build_model, arch_of,
                    cam_target_layer, count_params, count_flops, GradCAM, overlay_cam, fgsm_attack)

OUT, MODELS = "outputs", "models"


def run_epoch(model, loader, device, criterion, optimizer=None, adv_eps=0.0):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train and adv_eps > 0:
                # Adversarial training (defence): half clean loss + half loss on FGSM images
                x_adv = fgsm_attack(model, x, y, adv_eps)
                logits = model(x)
                loss = 0.5 * criterion(logits, y) + 0.5 * criterion(model(x_adv), y)
            else:
                logits = model(x)
                loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)   # gradient clipping
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    ys, ps, probs = [], [], []
    for x, y in loader:
        out = torch.softmax(model(x.to(device)), 1).cpu()
        ys.append(y); ps.append(out.argmax(1)); probs.append(out)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(probs).numpy()


def train_model(name, epochs, lr, loaders, device, pretrained, adv_eps=0.0):
    train_loader, val_loader = loaders
    model = build_model(name, pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)  # L2 regularisation
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=1)
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best, path = 0.0, os.path.join(MODELS, f"{name}.pt")
    print(f"\n=== Training {name}  ({count_params(model)/1e6:.2f}M params) on {device}"
          + (f", adversarial training eps={adv_eps}" if adv_eps else "") + " ===")
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tl, ta = run_epoch(model, train_loader, device, criterion, optimizer, adv_eps)
        vl, va = run_epoch(model, val_loader, device, criterion)
        scheduler.step()
        for k, v in zip(hist, (tl, vl, ta, va)):
            hist[k].append(v)
        flag = ""
        if va >= best:                     # keep best checkpoint (early-stopping style)
            best = va
            torch.save(model.state_dict(), path)
            flag = "  *saved*"
        print(f"ep {ep:02d}/{epochs} | train loss {tl:.4f} acc {ta:.4f} | "
              f"val loss {vl:.4f} acc {va:.4f} | {time.time()-t0:.0f}s{flag}")
    model.load_state_dict(torch.load(path, map_location=device))
    return model, hist


def plot_curves(name, hist):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    e = range(1, len(hist["train_loss"]) + 1)
    ax[0].plot(e, hist["train_loss"], "o-", label="train"); ax[0].plot(e, hist["val_loss"], "o-", label="val")
    ax[0].set_title(f"{name} - loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(e, hist["train_acc"], "o-", label="train"); ax[1].plot(e, hist["val_acc"], "o-", label="val")
    ax[1].set_title(f"{name} - accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    for a in ax: a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/{name}_curves.png", dpi=130); plt.close(fig)


def plot_confusion(name, y, p):
    cm = confusion_matrix(y, p, labels=range(NUM_CLASSES), normalize="true")
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(range(NUM_CLASSES), fontsize=7); ax.set_yticklabels(
        [f"{i} {CLASS_NAMES[i][:28]}" for i in range(NUM_CLASSES)], fontsize=7)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title(f"{name} - normalised confusion matrix (test set)")
    fig.colorbar(im, fraction=0.03); fig.tight_layout()
    fig.savefig(f"{OUT}/{name}_confusion.png", dpi=130); plt.close(fig)


def show_grid(images, titles, path, suptitle, cols=6):
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.3, rows * 2.6))
    for ax in np.array(axes).ravel(): ax.axis("off")
    for ax, im, t in zip(np.array(axes).ravel(), images, titles):
        ax.imshow(im); ax.set_title(t, fontsize=7)
    fig.suptitle(suptitle); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def misclassified_grid(name, test_ds, y, p, probs, k=18):
    wrong = np.where(y != p)[0]
    wrong = wrong[np.argsort(-probs[wrong, p[wrong]])][:k]   # most confident mistakes first
    if len(wrong) == 0:
        return
    ims = [test_ds[i][0].permute(1, 2, 0).numpy() for i in wrong]
    tts = [f"T: {CLASS_NAMES[y[i]][:20]}\nP: {CLASS_NAMES[p[i]][:20]} ({probs[i, p[i]]:.2f})" for i in wrong]
    show_grid(ims, tts, f"{OUT}/{name}_misclassified.png", f"{name}: most confident mistakes")


def gradcam_grid(name, model, test_ds, device, k=12, seed=0):
    cam = GradCAM(model, cam_target_layer(model))
    rng = np.random.default_rng(seed)
    ims, tts = [], []
    for i in rng.choice(len(test_ds), k, replace=False):
        x, y = test_ds[int(i)]
        heat, c = cam(x.unsqueeze(0).to(device))
        ims += [x.permute(1, 2, 0).numpy(), overlay_cam(x, heat)]
        tts += [f"true: {CLASS_NAMES[y][:22]}", f"pred: {CLASS_NAMES[c][:22]}"]
    show_grid(ims, tts, f"{OUT}/{name}_gradcam.png", f"{name}: Grad-CAM (where the CNN looks)")


def fgsm_eval(model, loader, device, eps_list, max_batches=20):
    model.eval()
    accs = []
    for eps in eps_list:
        correct = n = 0
        for b, (x, y) in enumerate(loader):
            if b >= max_batches: break
            x, y = x.to(device), y.to(device)
            xa = fgsm_attack(model, x, y, eps)
            with torch.no_grad():
                correct += (model(xa).argmax(1) == y).sum().item()
            n += y.size(0)
        accs.append(correct / n)
    return accs


@torch.no_grad()
def latency_ms(model, device, reps=50):
    model.eval()
    x = torch.rand(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    for _ in range(10): model(x)
    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps): model(x)
    if device.type == "cuda": torch.cuda.synchronize()
    return (time.time() - t0) / reps * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="torchvision", choices=["torchvision", "folder", "fake"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--models", nargs="+", default=["cnn", "resnet18", "mobilenet", "cnn_robust"])
    ap.add_argument("--epochs-cnn", type=int, default=15)
    ap.add_argument("--epochs-resnet", type=int, default=8, help="also used for mobilenet")
    ap.add_argument("--adv-eps", type=float, default=0.02, help="FGSM eps for *_robust models")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True); os.makedirs(MODELS, exist_ok=True)
    torch.manual_seed(42); np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, test_ds = get_datasets(args.data, args.source)
    print(f"train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)}")
    kw = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)

    cfg = {"cnn": (args.epochs_cnn, 3e-3), "resnet18": (args.epochs_resnet, 1e-3),
           "mobilenet": (args.epochs_resnet, 2e-3)}
    eps_list = [0, 0.005, 0.01, 0.02, 0.04, 0.08]
    mpath = f"{OUT}/metrics.json"
    results = json.load(open(mpath)) if os.path.exists(mpath) else {}
    for name in args.models:
        epochs, lr = cfg[arch_of(name)]
        adv = args.adv_eps if name.endswith("_robust") else 0.0
        model, hist = train_model(name, epochs, lr, (train_loader, val_loader), device,
                                  pretrained=not args.no_pretrained, adv_eps=adv)
        plot_curves(name, hist)
        y, p, probs = predict_all(model, test_loader, device)
        acc, f1 = float((y == p).mean()), float(f1_score(y, p, average="macro"))
        with open(f"{OUT}/{name}_report.txt", "w") as f:
            f.write(classification_report(y, p, labels=range(NUM_CLASSES),
                                          target_names=CLASS_NAMES, digits=4, zero_division=0))
        plot_confusion(name, y, p)
        misclassified_grid(name, test_ds, y, p, probs)
        gradcam_grid(name, model, test_ds, device)
        fgsm_accs = fgsm_eval(model, test_loader, device, eps_list)
        flops = count_flops(model, device)
        results[name] = {
            "params_M": round(count_params(model) / 1e6, 3),
            "MFLOPs": round(flops / 1e6, 1) if flops else None,
            "adversarial_training_eps": adv,
            "test_accuracy": round(acc, 4), "macro_f1": round(f1, 4),
            "best_val_accuracy": round(max(hist["val_acc"]), 4),
            "latency_ms_per_image": round(latency_ms(model, device), 3),
            "fgsm_accuracy": dict(zip(map(str, eps_list), [round(a, 4) for a in fgsm_accs])),
            "history": hist,
        }
        print(f"[{name}] TEST accuracy {acc:.4f} | macro-F1 {f1:.4f}")
        json.dump(results, open(mpath, "w"), indent=2)      # save after every model

    # FGSM robustness plot
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, r in results.items():
        ax.plot([float(e) for e in r["fgsm_accuracy"]], list(r["fgsm_accuracy"].values()),
                "o--" if name.endswith("_robust") else "o-", label=name)
    ax.set_xlabel("FGSM epsilon (pixel scale 0-1)"); ax.set_ylabel("test accuracy")
    ax.set_title("Fooling the CNN: accuracy under adversarial noise"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/fgsm_robustness.png", dpi=130); plt.close(fig)

    with open(f"{OUT}/comparison.csv", "w") as f:
        f.write("model,params_M,MFLOPs,test_accuracy,macro_f1,latency_ms,fgsm_acc_eps0.02\n")
        for n, r in results.items():
            f.write(f"{n},{r['params_M']},{r.get('MFLOPs')},{r['test_accuracy']},{r['macro_f1']},"
                    f"{r['latency_ms_per_image']},{r['fgsm_accuracy']['0.02']}\n")
    print("\n" + open(f"{OUT}/comparison.csv").read())

    # save a few test images for the demo app
    os.makedirs("samples", exist_ok=True)
    from torchvision.utils import save_image
    for j, i in enumerate(np.random.default_rng(1).choice(len(test_ds), 8, replace=False)):
        x, yy = test_ds[int(i)]
        save_image(x, f"samples/{j}_class{yy}.png")


if __name__ == "__main__":
    main()
