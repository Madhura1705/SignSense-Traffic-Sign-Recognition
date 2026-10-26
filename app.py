"""
SignSense -- Traffic Sign Recognition web app (Gradio).

  python app.py            -> opens http://127.0.0.1:7860
  python app.py --share    -> also gives a public link (useful from Colab)

Loads every trained model found in models/*.pt (cnn, resnet18, mobilenet, cnn_robust ...).
Tabs: Recognise | Compare models | Fool the CNN | Model report | About
"""
import argparse, csv, glob, json, os, sys
import numpy as np
import torch
import gradio as gr
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from common import (CLASS_NAMES, get_transforms, load_model, cam_target_layer,
                    display_name, GradCAM, overlay_cam, fgsm_attack)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TF = get_transforms(train=False)
ORDER = ["resnet18", "mobilenet", "cnn", "cnn_robust", "resnet18_robust", "mobilenet_robust"]

# ---------------------------------------------------------------- load models
MODELS, CAMS = {}, {}
for path in sorted(glob.glob(os.path.join(ROOT, "models", "*.pt")),
                   key=lambda p: ORDER.index(os.path.basename(p)[:-3])
                   if os.path.basename(p)[:-3] in ORDER else 99):
    key = os.path.basename(path)[:-3]
    try:
        MODELS[key] = load_model(key, path, DEVICE)
        CAMS[key] = GradCAM(MODELS[key], cam_target_layer(MODELS[key]))
    except Exception as e:                       # unknown / corrupt file -> skip
        print(f"skipping {path}: {e}")
if not MODELS:
    raise SystemExit("No trained models found in models/. Run `python train.py` first.")
NAME = {k: display_name(k) for k in MODELS}
KEY_OF = {v: k for k, v in NAME.items()}
CHOICES = list(NAME.values())
STANDARD = [k for k in MODELS if not k.endswith("_robust")]
ROBUST = [k for k in MODELS if k.endswith("_robust")]


def read_json(p):
    p = os.path.join(ROOT, p)
    return json.load(open(p)) if os.path.exists(p) else {}


def read_csv(p):
    p = os.path.join(ROOT, p)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


# ---------------------------------------------------------------- helpers
def to_tensor(img):
    return TF(Image.fromarray(img).convert("RGB")).unsqueeze(0).to(DEVICE)


def probs_of(model, x):
    with torch.no_grad():
        return torch.softmax(model(x), 1)[0].cpu().numpy()


def top5(p):
    idx = p.argsort()[::-1][:5]
    return {f"{CLASS_NAMES[i]}  (class {i})": float(p[i]) for i in idx}


def big(arr_uint8, size=256):
    return Image.fromarray(arr_uint8).resize((size, size), Image.NEAREST)


def confidence_text(c):
    return "High confidence" if c > 0.9 else "Medium confidence" if c > 0.6 else "Low confidence - check manually"


# ---------------------------------------------------------------- tab callbacks
def predict(img, model_name):
    if img is None:
        return None, None, ""
    key = KEY_OF[model_name]
    x = to_tensor(img)
    p = probs_of(MODELS[key], x)
    cls = int(p.argmax())
    heat, _ = CAMS[key](x, cls)
    md = f"### {CLASS_NAMES[cls]}\n**{p[cls]*100:.1f}%** - {confidence_text(p[cls])}"
    return top5(p), big(overlay_cam(x[0].cpu(), heat)), md


def compare(img):
    if img is None:
        return [], ""
    x = to_tensor(img)
    gallery, preds = [], {}
    for key, model in MODELS.items():
        p = probs_of(model, x)
        cls = int(p.argmax())
        preds[key] = (cls, float(p[cls]))
        heat, _ = CAMS[key](x, cls)
        gallery.append((big(overlay_cam(x[0].cpu(), heat), 220),
                        f"{NAME[key]}: {CLASS_NAMES[cls]} ({p[cls]*100:.1f}%)"))
    votes = [c for c, _ in preds.values()]
    majority = max(set(votes), key=votes.count)
    rows = [f"### Ensemble vote: {CLASS_NAMES[majority]}  ({votes.count(majority)}/{len(votes)} models)", "",
            "| Model | Prediction | Confidence | Agrees with vote |", "|---|---|---|---|"]
    for key, (cls, conf) in preds.items():
        rows.append(f"| {NAME[key]} | {CLASS_NAMES[cls]} | {conf*100:.1f}% | {'✅' if cls == majority else '❌'} |")
    return gallery, "\n".join(rows)


def attack(img, model_name, eps):
    if img is None:
        return None, None, None, ""
    key = KEY_OF[model_name]
    model = MODELS[key]
    x = to_tensor(img)
    cls = int(probs_of(model, x).argmax())
    xa = fgsm_attack(model, x, torch.tensor([cls], device=DEVICE), eps)
    pa = probs_of(model, xa)
    acls = int(pa.argmax())
    noise = (xa - x)[0].permute(1, 2, 0).cpu().numpy()
    noise = ((noise - noise.min()) / (noise.max() - noise.min() + 1e-8) * 255).astype(np.uint8)
    adv = (xa[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    md = (f"**Before:** {CLASS_NAMES[cls]}  \n**After (eps={eps}):** {CLASS_NAMES[acls]} ({pa[acls]*100:.1f}%)  \n"
          + ("### ❌ The model was fooled" if acls != cls else "### ✅ The model resisted this noise"))
    # Same adversarial image shown to every other model (transfer attack) -- shows the defence effect
    others = [k for k in MODELS if k != key]
    if others:
        md += ("\n\n**Same adversarial image given to the other models:**\n\n"
               "| Model | Clean prediction | Adversarial prediction | Fooled? |\n|---|---|---|---|")
        for k in others:
            clean_k = int(probs_of(MODELS[k], x).argmax())
            adv_k = int(probs_of(MODELS[k], xa).argmax())
            md += (f"\n| {NAME[k]} | {CLASS_NAMES[clean_k]} | {CLASS_NAMES[adv_k]} | "
                   f"{'❌ yes' if adv_k != clean_k else '✅ no'} |")
    return big(adv), big(noise), top5(pa), md


def metrics_table():
    m = read_json("outputs/metrics.json")
    if not m:
        return "_Run train.py to generate metrics._"
    rows = ["| Model | Params (M) | MFLOPs | Test acc. | Macro F1 | Latency (ms) | Acc. under FGSM eps=0.02 |",
            "|---|---|---|---|---|---|---|"]
    for k, r in m.items():
        rows.append(f"| {display_name(k)} | {r['params_M']} | {r.get('MFLOPs', '-')} | {r['test_accuracy']*100:.2f}% | "
                    f"{r['macro_f1']:.4f} | {r['latency_ms_per_image']} | {r['fgsm_accuracy']['0.02']*100:.1f}% |")
    onnx = read_csv("outputs/onnx_benchmark.csv")
    if onnx:
        rows += ["", "#### Edge deployment: ONNX Runtime (CPU)", "",
                 "| Model | ONNX size (MB) | PyTorch CPU (ms) | ONNX Runtime CPU (ms) | Speed-up |", "|---|---|---|---|---|"]
        for r in onnx:
            rows.append(f"| {display_name(r['model'])} | {r['onnx_size_MB']} | {float(r['torch_cpu_ms']):.2f} | "
                        f"{float(r['onnxruntime_cpu_ms']):.2f} | x{r['speedup']} |")
    return "\n".join(rows)


def plots():
    return [(p, os.path.basename(p).replace(".png", "").replace("_", " "))
            for p in sorted(glob.glob(os.path.join(ROOT, "outputs", "*.png")))]


examples = sorted(glob.glob(os.path.join(ROOT, "samples", "*.png")) + glob.glob(os.path.join(ROOT, "samples", "*.jpg")))

CSS = """
#title {text-align:center; margin-bottom:0}
#subtitle {text-align:center; color:#6b7280; margin-top:4px}
.verdict h3 {margin:0}
"""
theme = gr.themes.Soft(primary_hue="red", secondary_hue="amber", neutral_hue="slate")
GR6 = int(gr.__version__.split(".")[0]) >= 6
_style = dict(theme=theme, css=CSS)          # Gradio 6 moved these to launch()


def image_input(label, height):
    try:
        return gr.Image(type="numpy", label=label, height=height, sources=["upload", "webcam", "clipboard"])
    except TypeError:                         # very old Gradio
        return gr.Image(type="numpy", label=label, height=height)


with gr.Blocks(title="SignSense - Traffic Sign Recognition", **({} if GR6 else _style)) as demo:
    gr.Markdown("# 🚦 SignSense", elem_id="title")
    gr.Markdown(f"Traffic sign recognition with deep CNNs - 43 classes (GTSRB) - {len(MODELS)} models loaded - "
                "Grad-CAM explanations - adversarial attack & defence - edge deployment", elem_id="subtitle")

    # ---- Tab 1
    with gr.Tab("🔍 Recognise"):
        with gr.Row():
            with gr.Column(scale=1):
                inp = image_input("Upload / webcam / paste a traffic sign", 280)
                gr.Markdown("<small>Works best on a tightly cropped sign, like the GTSRB images.</small>")
                mdl = gr.Radio(CHOICES, value=CHOICES[0], label="Model")
                btn = gr.Button("Recognise sign", variant="primary")
                if examples:
                    gr.Examples(examples, inputs=inp, label="Try a test image")
            with gr.Column(scale=1):
                verdict = gr.Markdown(elem_classes="verdict")
                lbl = gr.Label(num_top_classes=5, label="Top-5 predictions")
            with gr.Column(scale=1):
                cam = gr.Image(label="Grad-CAM: where the model looked", height=280)
                gr.Markdown("Red = regions that most influenced the prediction. "
                            "A good model focuses on the sign's symbol, not the background.")
        btn.click(predict, [inp, mdl], [lbl, cam, verdict])
        inp.change(predict, [inp, mdl], [lbl, cam, verdict])

    # ---- Tab 2
    with gr.Tab("⚖️ Compare models"):
        gr.Markdown("Run **every trained model** on the same image and compare predictions, confidence "
                    "and attention (Grad-CAM) side by side. The ensemble vote combines them.")
        with gr.Row():
            with gr.Column(scale=1):
                cinp = image_input("Traffic sign image", 260)
                cbtn = gr.Button("Compare all models", variant="primary")
                if examples:
                    gr.Examples(examples, inputs=cinp)
            with gr.Column(scale=2):
                cmd = gr.Markdown()
                cgal = gr.Gallery(label="Grad-CAM per model", columns=4, height="auto", object_fit="contain")
        cbtn.click(compare, cinp, [cgal, cmd])
        cinp.change(compare, cinp, [cgal, cmd])

    # ---- Tab 3
    with gr.Tab("⚠️ Fool the CNN"):
        gr.Markdown("**FGSM adversarial attack**: x' = x + eps · sign(∇ₓ Loss). Tiny, carefully chosen noise "
                    "that a human barely notices can completely change the prediction. "
                    + ("Compare the standard models with the **adversarially-trained (robust)** model, "
                       "which was trained on such images as a defence." if ROBUST else ""))
        with gr.Row():
            with gr.Column():
                ainp = image_input("Original image", 256)
                amdl = gr.Radio(CHOICES, value=NAME[STANDARD[0]] if STANDARD else CHOICES[0], label="Attack this model")
                eps = gr.Slider(0, 0.1, value=0.02, step=0.005, label="Noise strength (epsilon)")
                abtn = gr.Button("Attack", variant="primary")
                if examples:
                    gr.Examples(examples, inputs=ainp)
            with gr.Column():
                aimg = gr.Image(label="Adversarial image", height=256)
                anoise = gr.Image(label="Added noise (amplified)", height=256)
            with gr.Column():
                amd = gr.Markdown()
                albl = gr.Label(num_top_classes=5, label="Prediction on adversarial image")
        abtn.click(attack, [ainp, amdl, eps], [aimg, anoise, albl, amd])

    # ---- Tab 4
    with gr.Tab("📊 Model report"):
        gr.Markdown("### Model comparison (GTSRB test set, 12,630 images)")
        gr.Markdown(metrics_table())
        gr.Gallery(plots(), columns=2, height="auto", object_fit="contain",
                   label="EDA, training curves, confusion matrices, Grad-CAM, robustness, latency")

    # ---- Tab 5
    with gr.Tab("ℹ️ About"):
        about = os.path.join(ROOT, "ABOUT.md")
        gr.Markdown(open(about, encoding="utf-8").read() if os.path.exists(about) else "")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    demo.launch(share=a.share, server_port=a.port, **(_style if GR6 else {}))
