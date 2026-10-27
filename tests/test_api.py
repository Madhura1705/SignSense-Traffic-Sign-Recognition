import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "models" in data
    assert len(data["models"]) == 4
    for expected in ["cnn", "resnet18", "mobilenet", "cnn_robust"]:
        assert expected in data["models"]
    assert data["device"] == "cpu"


def test_models():
    res = client.get("/api/models")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 4
    keys = [m["key"] for m in data]
    for expected in ["cnn", "resnet18", "mobilenet", "cnn_robust"]:
        assert expected in keys
    for m in data:
        assert "name" in m
        assert "arch" in m
        assert "params_M" in m
        assert "test_accuracy" in m
        assert m["test_accuracy"] > 0.90


def test_samples():
    res = client.get("/api/samples")
    assert res.status_code == 200
    data = res.json()
    assert len(data) >= 12
    # Verify standard samples
    standard = [s for s in data if not s["real"]]
    assert len(standard) >= 12
    stop_sample = next((s for s in standard if s["class_id"] == 14), None)
    assert stop_sample is not None
    assert "Stop" in stop_sample["class_name"]

    # Verify real uncropped samples
    real = [s for s in data if s["real"]]
    assert len(real) >= 5
    for r in real:
        assert r["class_id"] is None
        assert r["class_name"] is None


def test_predict_stop_all_models():
    stop_path = os.path.join(ROOT, "samples", "14_stop.png")
    assert os.path.exists(stop_path), "samples/14_stop.png does not exist"

    for model_key in ["cnn", "resnet18", "mobilenet", "cnn_robust"]:
        with open(stop_path, "rb") as f:
            res = client.post(
                "/api/predict",
                files={"image": ("14_stop.png", f, "image/png")},
                data={"model": model_key},
            )
        assert res.status_code == 200, f"Predict failed for {model_key}: {res.text}"
        data = res.json()
        assert data["model"] == model_key
        assert data["class_id"] == 14, f"Expected 14 for {model_key}, got {data['class_id']}"
        assert "Stop" in data["class_name"]
        assert data["confidence"] > 0.90
        assert len(data["top5"]) == 5
        assert data["top5"][0]["class_id"] == 14
        assert data["input_b64"].startswith("data:image/png;base64,")
        assert data["overlay_b64"].startswith("data:image/png;base64,")
        assert data["heatmap_b64"].startswith("data:image/png;base64,")
        assert data["latency_ms"] >= 0


def test_compare_vote():
    stop_path = os.path.join(ROOT, "samples", "14_stop.png")
    with open(stop_path, "rb") as f:
        res = client.post(
            "/api/compare",
            files={"image": ("14_stop.png", f, "image/png")},
        )
    assert res.status_code == 200
    data = res.json()
    assert len(data["results"]) == 4
    for r in data["results"]:
        assert r["class_id"] == 14
        assert r["overlay_b64"].startswith("data:image/png;base64,")
    assert data["vote"]["class_id"] == 14
    assert "Stop" in data["vote"]["class_name"]
    assert data["vote"]["count"] == 4
    assert data["vote"]["total"] == 4


def test_attack_eps_zero():
    stop_path = os.path.join(ROOT, "samples", "14_stop.png")
    with open(stop_path, "rb") as f:
        res = client.post(
            "/api/attack",
            files={"image": ("14_stop.png", f, "image/png")},
            data={"model": "cnn", "eps": 0.0},
        )
    assert res.status_code == 200
    data = res.json()
    assert data["fooled"] is False
    assert data["clean"]["class_id"] == 14
    assert data["adversarial"]["class_id"] == 14
    assert data["adv_b64"].startswith("data:image/png;base64,")
    assert data["noise_b64"].startswith("data:image/png;base64,")
    assert len(data["transfer"]) == 3


def test_attack_speed_50_cnn():
    speed_path = os.path.join(ROOT, "samples", "2_speed_50.png")
    assert os.path.exists(speed_path)
    with open(speed_path, "rb") as f:
        res = client.post(
            "/api/attack",
            files={"image": ("2_speed_50.png", f, "image/png")},
            data={"model": "cnn", "eps": 0.05},
        )
    assert res.status_code == 200
    data = res.json()
    assert data["clean"]["class_id"] == 2
    assert "fooled" in data
    assert data["adv_b64"].startswith("data:image/png;base64,")
    assert data["noise_b64"].startswith("data:image/png;base64,")
    assert len(data["transfer"]) == 3
    # Check that transfer contains cnn_robust
    robust_transfer = next((t for t in data["transfer"] if t["model"] == "cnn_robust"), None)
    assert robust_transfer is not None


def test_attack_sweep():
    stop_path = os.path.join(ROOT, "samples", "14_stop.png")
    with open(stop_path, "rb") as f:
        res = client.post(
            "/api/attack/sweep",
            files={"image": ("14_stop.png", f, "image/png")},
        )
    assert res.status_code == 200
    data = res.json()
    assert len(data["eps"]) == 11
    assert data["eps"][0] == 0.0
    assert data["eps"][-1] == 0.10
    for k in ["cnn", "resnet18", "mobilenet", "cnn_robust"]:
        assert k in data["series"]
        assert len(data["series"][k]) == 11
        assert k in data["breaks_at"]


def test_report_plots_exist():
    res = client.get("/api/report")
    assert res.status_code == 200
    data = res.json()
    assert "metrics" in data
    assert "onnx" in data
    assert "worst_classes" in data
    assert "plots" in data

    # Verify dataset plots exist on disk
    for p in data["plots"]["dataset"]:
        rel = p["url"].lstrip("/")
        full_path = os.path.join(ROOT, rel)
        assert os.path.exists(full_path), f"Dataset plot missing: {full_path}"

    # Verify per_model plots exist on disk
    for model_key, model_plots in data["plots"]["per_model"].items():
        for plot_type, url in model_plots.items():
            rel = url.lstrip("/")
            full_path = os.path.join(ROOT, rel)
            assert os.path.exists(full_path), f"Per-model plot missing: {full_path}"

    # Verify global plots exist on disk
    for p in data["plots"]["global"]:
        rel = p["url"].lstrip("/")
        full_path = os.path.join(ROOT, rel)
        assert os.path.exists(full_path), f"Global plot missing: {full_path}"

    # Verify worst_classes has entries for all 4 models
    for k in ["cnn", "resnet18", "mobilenet", "cnn_robust"]:
        assert k in data["worst_classes"]
        assert len(data["worst_classes"][k]) == 5


def test_bad_model_key_and_invalid_upload():
    stop_path = os.path.join(ROOT, "samples", "14_stop.png")
    # Bad model key -> 400
    with open(stop_path, "rb") as f:
        res = client.post(
            "/api/predict",
            files={"image": ("14_stop.png", f, "image/png")},
            data={"model": "non_existent_model"},
        )
    assert res.status_code == 400
    assert "Unknown model key" in res.json()["detail"]

    # Non-image upload -> 400
    res = client.post(
        "/api/predict",
        files={"image": ("test.txt", b"This is not an image file", "text/plain")},
        data={"model": "cnn"},
    )
    assert res.status_code == 400
    assert "Invalid image file" in res.json()["detail"]
