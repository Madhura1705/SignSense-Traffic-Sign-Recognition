"""
Edge deployment step: export trained models to ONNX and benchmark
PyTorch vs ONNX Runtime on CPU (syllabus LO 5.3 / 5.4 / 4.4: FLOPs,
deployment on edge devices, latency-accuracy trade-off).

  python export_onnx.py                    # all trained models in models/
  python export_onnx.py --models mobilenet

Outputs: models/<name>.onnx, outputs/onnx_benchmark.csv, outputs/latency_vs_accuracy.png
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from common import IMG_SIZE, load_model, display_name


def bench(fn, reps=100):
    for _ in range(10):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1000


def main():
    import onnxruntime as ort
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    args = ap.parse_args()
    names = args.models or sorted(f[:-3] for f in os.listdir("models") if f.endswith(".pt"))
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    x = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)
    metrics = json.load(open("outputs/metrics.json")) if os.path.exists("outputs/metrics.json") else {}
    rows = []
    for name in names:
        model = load_model(name, f"models/{name}.pt", "cpu")
        path = f"models/{name}.onnx"
        torch.onnx.export(model, (x,), path, input_names=["image"], output_names=["logits"],
                          dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                          opset_version=17, dynamo=False)
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        with torch.no_grad():
            ref = model(x).numpy()
        out = sess.run(None, {"image": x.numpy()})[0]
        max_diff = float(np.abs(ref - out).max())
        with torch.no_grad():
            t_pt = bench(lambda: model(x))
        t_ox = bench(lambda: sess.run(None, {"image": x.numpy()}))
        size = os.path.getsize(path) / 1e6
        acc = metrics.get(name, {}).get("test_accuracy")
        rows.append((name, size, t_pt, t_ox, max_diff, acc))
        print(f"{name:12s} onnx {size:6.2f} MB | torch {t_pt:6.2f} ms | onnxruntime {t_ox:6.2f} ms "
              f"| speed-up x{t_pt/t_ox:.2f} | max |diff| {max_diff:.1e}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/onnx_benchmark.csv", "w") as f:
        f.write("model,onnx_size_MB,torch_cpu_ms,onnxruntime_cpu_ms,speedup,max_abs_diff,test_accuracy\n")
        for n, s, a, b, d, acc in rows:
            f.write(f"{n},{s:.2f},{a:.3f},{b:.3f},{a/b:.2f},{d:.2e},{acc}\n")

    pts = [r for r in rows if r[5] is not None]
    if pts:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for n, s, a, b, d, acc in pts:
            ax.scatter(b, acc * 100, s=40 + s * 8)
            ax.annotate(f"{display_name(n)}\n{s:.1f} MB", (b, acc * 100), fontsize=7,
                        xytext=(6, 4), textcoords="offset points")
        ax.set_xlabel("ONNX Runtime CPU latency (ms / image)")
        ax.set_ylabel("test accuracy (%)")
        ax.set_title("Latency vs accuracy (bubble size = model size)")
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig("outputs/latency_vs_accuracy.png", dpi=130)
    print("saved outputs/onnx_benchmark.csv")


if __name__ == "__main__":
    main()
