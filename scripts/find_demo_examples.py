"""
Find demo examples where plain CNN is fooled but Robust CNN resists.
Writes docs/DEMO_EXAMPLES.md.
"""
import glob
import os
import sys
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from common import CLASS_NAMES, load_model, get_transforms, fgsm_attack

def main():
    tf = get_transforms(train=False)
    cnn = load_model("cnn", "models/cnn.pt", "cpu")
    cnn_robust = load_model("cnn_robust", "models/cnn_robust.pt", "cpu")

    candidates = sorted(glob.glob("samples/*.png"))
    cases = []

    for path in candidates:
        fname = os.path.basename(path)
        img = Image.open(path).convert("RGB")
        x = tf(img).unsqueeze(0)

        with torch.no_grad():
            c_clean = int(cnn(x).argmax(1))
            r_clean = int(cnn_robust(x).argmax(1))

        if c_clean != r_clean:
            continue

        for eps in [0.01, 0.02, 0.03, 0.04, 0.05]:
            xa = fgsm_attack(cnn, x, torch.tensor([c_clean]), eps)
            with torch.no_grad():
                c_adv = int(cnn(xa).argmax(1))
                r_adv = int(cnn_robust(xa).argmax(1))

            if c_adv != c_clean and r_adv == c_clean:
                cases.append((fname, CLASS_NAMES[c_clean], eps, CLASS_NAMES[c_adv], CLASS_NAMES[r_adv]))
                break

    print(f"Found {len(cases)} striking demo cases:")
    md = [
        "# SignSense: Demo Examples for Member 3 (Adversarial Robustness)\n",
        "These test sample images illustrate where the **Custom CNN** is successfully fooled by FGSM noise, but the **Robust CNN (Adversarial Training)** successfully resists and predicts the correct class.\n",
        "| Sample Image | True / Clean Class | Epsilon (Noise Strength) | Standard CNN Prediction | Robust CNN Prediction | Defense Status |\n",
        "|---|---|---|---|---|---|\n"
    ]

    for fname, true_cls, eps, c_pred, r_pred in cases:
        print(f"  {fname} (eps={eps}): Plain CNN -> {c_pred} | Robust CNN -> {r_pred}")
        md.append(f"| `samples/{fname}` | **{true_cls}** | `{eps}` | ❌ {c_pred} | ✅ **{r_pred}** | **Resisted** |\n")

    os.makedirs("docs", exist_ok=True)
    with open("docs/DEMO_EXAMPLES.md", "w", encoding="utf-8") as f:
        f.writelines(md)

    print("Saved docs/DEMO_EXAMPLES.md successfully!")

if __name__ == "__main__":
    main()
