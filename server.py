import argparse
import csv
import io
import json
import os
import re
import threading
import time
import webbrowser
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import torch
from PIL import Image

import api.inference as inference
from src.common import CLASS_NAMES, arch_of, display_name, fgsm_attack

ROOT = os.path.dirname(os.path.abspath(__file__))
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB

app = FastAPI(title="SignSense API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Mount static assets
if os.path.exists(os.path.join(ROOT, "outputs")):
    app.mount("/outputs", StaticFiles(directory=os.path.join(ROOT, "outputs")), name="outputs")

if os.path.exists(os.path.join(ROOT, "samples")):
    app.mount("/samples", StaticFiles(directory=os.path.join(ROOT, "samples")), name="samples")

if os.path.exists(os.path.join(ROOT, "web")):
    app.mount("/static", StaticFiles(directory=os.path.join(ROOT, "web")), name="static")


def read_json_file(rel_path: str):
    p = os.path.join(ROOT, rel_path)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def read_csv_file(rel_path: str):
    p = os.path.join(ROOT, rel_path)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    return []


async def validate_image_upload(file: UploadFile) -> Image.Image:
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No image file provided.")
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum upload size is 10 MB.")
    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
        return Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file. Please upload a valid PNG, JPEG, or WebP image."
        )


def validate_model_key(model: str):
    if model not in inference.MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model key '{model}'. Valid model keys are: {list(inference.MODELS.keys())}"
        )


@app.get("/")
def serve_index():
    index_file = os.path.join(ROOT, "web", "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "SignSense Web UI placeholder"}


@app.get("/api/health")
def get_health():
    return {
        "status": "ok",
        "models": list(inference.MODELS.keys()),
        "device": inference.DEVICE,
    }


@app.get("/api/models")
def get_models():
    metrics = read_json_file(os.path.join("outputs", "metrics.json"))
    results = []
    for key in ["cnn", "resnet18", "mobilenet", "cnn_robust"]:
        if key in inference.MODELS:
            m_info = metrics.get(key, {})
            results.append({
                "key": key,
                "name": display_name(key),
                "arch": arch_of(key),
                "robust": key.endswith("_robust"),
                "params_M": m_info.get("params_M"),
                "MFLOPs": m_info.get("MFLOPs"),
                "test_accuracy": m_info.get("test_accuracy"),
                "macro_f1": m_info.get("macro_f1"),
                "latency_ms": m_info.get("latency_ms_per_image"),
                "fgsm_acc_002": m_info.get("fgsm_accuracy", {}).get("0.02"),
            })
    return results


@app.get("/api/samples")
def get_samples():
    samples = []
    samples_dir = os.path.join(ROOT, "samples")
    if os.path.exists(samples_dir):
        # 1. Standard GTSRB test samples
        for fname in sorted(os.listdir(samples_dir)):
            if fname.endswith(".png") and not fname.startswith("."):
                prefix = fname.split("_")[0]
                cid = int(prefix) if prefix.isdigit() else None
                cname = CLASS_NAMES[cid] if cid is not None and cid < len(CLASS_NAMES) else None
                samples.append({
                    "file": fname,
                    "url": f"/samples/{fname}",
                    "class_id": cid,
                    "class_name": cname,
                    "real": False,
                })
        # 2. Uncropped real-world test samples
        real_dir = os.path.join(samples_dir, "real")
        if os.path.exists(real_dir):
            for fname in sorted(os.listdir(real_dir)):
                if fname.endswith(".png") and not fname.startswith("."):
                    samples.append({
                        "file": f"real/{fname}",
                        "url": f"/samples/real/{fname}",
                        "class_id": None,
                        "class_name": None,
                        "real": True,
                    })
    return samples


@app.get("/api/team")
def get_team():
    team_data = read_json_file("team.json")
    if not team_data:
        return [
            {"name": "Member 1", "roll": "", "module": "Data & custom CNN", "summary": "EDA, preprocessing, CNN from scratch"},
            {"name": "Member 2", "roll": "", "module": "Transfer learning & edge deployment", "summary": "ResNet-18, MobileNetV3, ONNX benchmark"},
            {"name": "Member 3", "roll": "", "module": "Explainability & robustness", "summary": "Grad-CAM, FGSM attack, Robust CNN"},
            {"name": "Member 4", "roll": "", "module": "Application & integration", "summary": "FastAPI web dashboard, testing, deployment"},
        ]
    return team_data


@app.post("/api/predict")
async def predict_endpoint(image: UploadFile = File(...), model: str = Form(...)):
    validate_model_key(model)
    pil_img = await validate_image_upload(image)
    x = inference.preprocess(pil_img)
    res = inference.predict(model, x)
    cam = inference.gradcam(model, x, res["class_id"])
    return {
        "model": model,
        "class_id": res["class_id"],
        "class_name": res["class_name"],
        "confidence": res["confidence"],
        "top5": res["top5"],
        "latency_ms": res["latency_ms"],
        "input_b64": inference.to_input_b64(x),
        "overlay_b64": inference.to_overlay_b64(x, cam),
        "heatmap_b64": inference.to_heatmap_b64(cam),
    }


@app.post("/api/compare")
async def compare_endpoint(image: UploadFile = File(...)):
    pil_img = await validate_image_upload(image)
    x = inference.preprocess(pil_img)
    results = []
    votes = {}
    for key in ["cnn", "resnet18", "mobilenet", "cnn_robust"]:
        if key in inference.MODELS:
            pred = inference.predict(key, x)
            cam = inference.gradcam(key, x, pred["class_id"])
            pred["model"] = key
            pred["overlay_b64"] = inference.to_overlay_b64(x, cam)
            results.append(pred)
            cid = pred["class_id"]
            votes[cid] = votes.get(cid, 0) + 1

    max_cid = max(votes, key=votes.get) if votes else 0
    vote = {
        "class_id": max_cid,
        "class_name": CLASS_NAMES[max_cid] if max_cid < len(CLASS_NAMES) else "",
        "count": votes.get(max_cid, 0),
        "total": len(results),
    }
    return {"results": results, "vote": vote}


@app.post("/api/attack")
async def attack_endpoint(
    image: UploadFile = File(...),
    model: str = Form(...),
    eps: float = Form(...)
):
    validate_model_key(model)
    if eps < 0.0 or eps > 0.15:
        raise HTTPException(status_code=400, detail="Epsilon must be between 0.0 and 0.15.")
    pil_img = await validate_image_upload(image)
    x = inference.preprocess(pil_img)

    clean_pred = inference.predict(model, x)
    clean_id = clean_pred["class_id"]

    model_obj = inference.MODELS[model]
    xa = fgsm_attack(model_obj, x, torch.tensor([clean_id], device=inference.DEVICE), eps)

    adv_pred = inference.predict(model, xa)
    adv_id = adv_pred["class_id"]

    transfer = []
    for k in ["cnn", "resnet18", "mobilenet", "cnn_robust"]:
        if k in inference.MODELS and k != model:
            c_pred = inference.predict(k, x)
            a_pred = inference.predict(k, xa)
            transfer.append({
                "model": k,
                "name": display_name(k),
                "clean_class": c_pred["class_name"],
                "adv_class": a_pred["class_name"],
                "fooled": bool(a_pred["class_id"] != c_pred["class_id"]),
            })

    return {
        "model": model,
        "eps": eps,
        "clean": {
            "class_id": clean_id,
            "class_name": clean_pred["class_name"],
            "confidence": clean_pred["confidence"],
        },
        "adversarial": {
            "class_id": adv_id,
            "class_name": adv_pred["class_name"],
            "confidence": adv_pred["confidence"],
            "top5": adv_pred["top5"],
        },
        "fooled": bool(adv_id != clean_id),
        "input_b64": inference.to_input_b64(x),
        "adv_b64": inference.to_adv_b64(xa),
        "noise_b64": inference.to_noise_b64(xa, x),
        "transfer": transfer,
    }


@app.post("/api/attack/sweep")
async def sweep_endpoint(image: UploadFile = File(...)):
    pil_img = await validate_image_upload(image)
    x = inference.preprocess(pil_img)
    eps_values = [round(e * 0.01, 2) for e in range(11)]  # [0.0, 0.01, ..., 0.10]
    series = {}
    breaks_at = {}

    for k in ["cnn", "resnet18", "mobilenet", "cnn_robust"]:
        if k not in inference.MODELS:
            continue
        m = inference.MODELS[k]
        clean_pred = inference.predict(k, x)
        clean_id = clean_pred["class_id"]

        probs_series = []
        first_break = None
        for eps_val in eps_values:
            xa = fgsm_attack(m, x, torch.tensor([clean_id], device=inference.DEVICE), eps_val)
            with torch.no_grad():
                logits = m(xa)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_cls = int(probs.argmax())
            clean_prob = float(probs[clean_id])
            probs_series.append(round(clean_prob, 4))
            if pred_cls != clean_id and first_break is None and eps_val > 0:
                first_break = eps_val

        series[k] = probs_series
        breaks_at[k] = first_break

    return {"eps": eps_values, "series": series, "breaks_at": breaks_at}


@app.get("/api/report")
def get_report():
    metrics = read_json_file(os.path.join("outputs", "metrics.json"))
    onnx = read_csv_file(os.path.join("outputs", "onnx_benchmark.csv"))

    worst_classes = {}
    for key in ["cnn", "resnet18", "mobilenet", "cnn_robust"]:
        report_path = os.path.join(ROOT, "outputs", f"{key}_report.txt")
        if os.path.exists(report_path):
            classes = []
            with open(report_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip() or "precision" in line or "accuracy" in line or "avg" in line:
                        continue
                    m = re.match(r"^\s*(.*?)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)\s*$", line)
                    if m:
                        name, prec, rec, f1, supp = m.groups()
                        classes.append({
                            "class_name": name.strip(),
                            "precision": float(prec),
                            "recall": float(rec),
                            "f1": float(f1),
                            "support": int(supp),
                        })
            classes.sort(key=lambda item: item["f1"])
            worst_classes[key] = classes[:5]

    plots = {
        "dataset": [
            {"id": "distribution", "title": "Class Distribution", "url": "/outputs/eda_class_distribution.png"},
            {"id": "samples", "title": "Dataset Samples", "url": "/outputs/eda_samples.png"},
            {"id": "sizes", "title": "Image Sizes", "url": "/outputs/eda_image_sizes.png"},
            {"id": "augmentation", "title": "Data Augmentation", "url": "/outputs/eda_augmentation.png"},
        ],
        "per_model": {
            k: {
                "curves": f"/outputs/{k}_curves.png",
                "confusion": f"/outputs/{k}_confusion.png",
                "gradcam": f"/outputs/{k}_gradcam.png",
                "misclassified": f"/outputs/{k}_misclassified.png",
            }
            for k in ["cnn", "resnet18", "mobilenet", "cnn_robust"]
        },
        "global": [
            {"id": "fgsm_robustness", "title": "FGSM Robustness Curve", "url": "/outputs/fgsm_robustness.png"},
            {"id": "latency_vs_accuracy", "title": "Latency vs Accuracy Trade-off", "url": "/outputs/latency_vs_accuracy.png"},
        ],
    }

    return {
        "metrics": metrics,
        "onnx": onnx,
        "worst_classes": worst_classes,
        "plots": plots,
    }


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description="SignSense Web Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()

    if not args.no_browser:
        def open_browser():
            time.sleep(1.2)
            webbrowser.open(f"http://{args.host}:{args.port}")
        threading.Thread(target=open_browser, daemon=True).start()

    print(f"Starting SignSense server at http://{args.host}:{args.port}")
    uvicorn.run("server:app", host=args.host, port=args.port, reload=False)
