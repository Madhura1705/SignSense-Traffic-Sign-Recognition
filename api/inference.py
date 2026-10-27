import base64
import glob
import io
import os
import time
import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image

from src.common import (
    CLASS_NAMES,
    GradCAM,
    cam_target_layer,
    display_name,
    fgsm_attack,
    get_transforms,
    load_model,
    overlay_cam,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = "cpu"
TF = get_transforms(train=False)

MODELS = {}
CAMS = {}

# Load all valid models on startup
VALID_KEYS = ["cnn", "resnet18", "mobilenet", "cnn_robust"]
for path in sorted(glob.glob(os.path.join(ROOT, "models", "*.pt"))):
    key = os.path.splitext(os.path.basename(path))[0]
    if key not in VALID_KEYS:
        continue
    try:
        model = load_model(key, path, device=DEVICE)
        MODELS[key] = model
        CAMS[key] = GradCAM(model, cam_target_layer(model))
    except Exception as e:
        print(f"Skipping model {path}: {e}")

if not MODELS:
    print("Warning: No models loaded from models/*.pt")


def preprocess(pil_img: Image.Image) -> torch.Tensor:
    """Preprocess PIL image into (1, 3, 64, 64) tensor in [0, 1]."""
    img = pil_img.convert("RGB")
    return TF(img).unsqueeze(0).to(DEVICE)


def arr_to_png_b64(arr_uint8: np.ndarray, size: int = 256) -> str:
    """Convert numpy array (H, W, 3) or (H, W, 4) uint8 to base64 PNG data URL upscaled with NEAREST."""
    mode = "RGBA" if arr_uint8.ndim == 3 and arr_uint8.shape[2] == 4 else "RGB"
    img = Image.fromarray(arr_uint8, mode=mode)
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64_str}"


def to_input_b64(x: torch.Tensor) -> str:
    """Upscaled base64 PNG of the 64x64 model input."""
    arr = (x[0].permute(1, 2, 0).detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    return arr_to_png_b64(arr, size=256)


def to_overlay_b64(x: torch.Tensor, cam: np.ndarray, alpha: float = 0.45) -> str:
    """Overlay CAM heatmap on input image upscaled to 256x256 base64 PNG."""
    overlay = overlay_cam(x[0].detach().cpu(), cam, alpha=alpha)
    return arr_to_png_b64(overlay, size=256)


def to_heatmap_b64(cam: np.ndarray) -> str:
    """Jet-coloured heatmap as RGBA where alpha = heat value (0..255) upscaled to 256x256 base64 PNG."""
    jet_colors = cm.jet(cam)  # (H, W, 4) in [0, 1]
    rgba = (jet_colors * 255).astype(np.uint8)
    rgba[..., 3] = (np.clip(cam, 0, 1) * 255).astype(np.uint8)
    return arr_to_png_b64(rgba, size=256)


def to_noise_b64(xa: torch.Tensor, x: torch.Tensor) -> str:
    """Adversarial noise amplified to [0, 255] upscaled to 256x256 base64 PNG."""
    noise = (xa - x)[0].permute(1, 2, 0).detach().cpu().numpy()
    noise = ((noise - noise.min()) / (noise.max() - noise.min() + 1e-8) * 255).astype(np.uint8)
    return arr_to_png_b64(noise, size=256)


def to_adv_b64(xa: torch.Tensor) -> str:
    """Adversarial image upscaled to 256x256 base64 PNG."""
    adv = (xa[0].permute(1, 2, 0).detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    return arr_to_png_b64(adv, size=256)


def predict(key: str, x: torch.Tensor) -> dict:
    """Run model inference and return top prediction, top5, and latency."""
    if key not in MODELS:
        raise ValueError(f"Unknown model key '{key}'")
    model = MODELS[key]
    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    class_id = int(probs.argmax())
    confidence = float(probs[class_id])
    top_idx = probs.argsort()[::-1][:5]
    top5 = [
        {
            "class_id": int(i),
            "class_name": CLASS_NAMES[i],
            "prob": round(float(probs[i]), 4),
        }
        for i in top_idx
    ]
    return {
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id],
        "confidence": round(confidence, 4),
        "top5": top5,
        "latency_ms": latency_ms,
    }


def gradcam(key: str, x: torch.Tensor, class_id: int = None) -> np.ndarray:
    """Generate Grad-CAM heatmap (64x64 float 0..1)."""
    if key not in CAMS:
        raise ValueError(f"Unknown model key '{key}'")
    cam, _ = CAMS[key](x, class_id)
    return cam
