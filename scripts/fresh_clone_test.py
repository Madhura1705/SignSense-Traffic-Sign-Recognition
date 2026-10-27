"""
Fresh-clone verification test
Clones current repo into a temporary folder, verifies all files are present,
tests running the server using existing python executable, and cleans up.
"""
import os
import sys
import shutil
import tempfile
import subprocess
import time
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
temp_dir = tempfile.mkdtemp(prefix="signsense_fresh_")
print(f"Cloning repo into {temp_dir} ...")

try:
    subprocess.check_call(["git", "clone", ROOT, temp_dir])
    print("Git clone successful.")

    # Check key files
    expected = [
        "server.py",
        "run.bat",
        "run.sh",
        "requirements.txt",
        "team.json",
        "web/index.html",
        "web/css/tokens.css",
        "web/js/main.js",
        "api/inference.py",
        "models/cnn.pt",
        "models/resnet18.pt",
        "models/mobilenet.pt",
        "models/cnn_robust.pt",
        "outputs/metrics.json",
        "outputs/comparison.csv",
        "outputs/onnx_benchmark.csv",
    ]
    for rel in expected:
        p = os.path.join(temp_dir, rel)
        assert os.path.exists(p), f"Missing expected file in fresh clone: {rel}"
    print("[PASS] All critical application files and weights verified in fresh clone.")

    # Start server on temporary port 7865
    port = 7865
    proc = subprocess.Popen(
        [sys.executable, os.path.join(temp_dir, "server.py"), "--port", str(port), "--no-browser"],
        cwd=temp_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    healthy = False
    for _ in range(30):
        try:
            r = requests.get(f"http://127.0.0.1:{port}/api/health", timeout=1)
            if r.status_code == 200 and r.json().get("status") == "ok":
                healthy = True
                break
        except Exception:
            pass
        time.sleep(0.5)

    proc.kill()
    assert healthy, "Server failed to start and report healthy in fresh clone!"
    print(f"[PASS] Server successfully started and passed /api/health probe on port {port}.")

finally:
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass
    print("Fresh clone cleanup complete.")
