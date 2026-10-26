### About SignSense

**Problem.** Autonomous vehicles and driver-assistance systems must read traffic signs reliably, in poor light, at odd angles, on cheap in-car hardware, and even when someone deliberately tampers with the sign. SignSense classifies a cropped traffic-sign image into one of **43 classes** of the German Traffic Sign Recognition Benchmark (GTSRB), with about 39,000 training images and 12,630 test images.

**Four modules, one per team member**

| # | Module | What it does | Syllabus |
|---|---|---|---|
| 1 | **Data & custom CNN** | EDA, preprocessing and augmentation; VGG-style CNN built from scratch with BatchNorm, Dropout and He init | M2, M3, M4, M5 |
| 2 | **Transfer learning & edge deployment** | ResNet-18 vs MobileNetV3-Small; params, FLOPs and latency trade-off; ONNX export and ONNX Runtime benchmark | M4, M5 (LO 4.4, 5.3, 5.4) |
| 3 | **Explainability & robustness** | Grad-CAM; FGSM adversarial attack; adversarial training as a defence | M5 (visualising and fooling CNNs) |
| 4 | **Application & integration** | Gradio web app (5 tabs), model ensemble, Colab notebook, GitHub, deployment | Integration |

**Training recipe:** augmentation (rotation, shift, scale, shear, colour jitter, but no flips, since a flip turns "turn left" into "turn right"), AdamW with weight decay, OneCycle learning rate, label smoothing, gradient clipping, best-checkpoint saving (early-stopping style).

**Limitations:** works on cropped signs (no detection step); German signs only; adversarial training helps against FGSM but not against every attack.
