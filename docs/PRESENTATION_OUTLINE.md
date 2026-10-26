# SignSense: Presentation Slide Outline (16 Slides)

**Course:** PECE-06 Deep Learning, Sem VII, Project 13  
**Project:** SignSense — Robust Traffic Sign Recognition System for Autonomous Driving  
**Presenter Order:** Member 4 (Opening & Problem Formulation) $\to$ Member 1 (Data & Custom CNN) $\to$ Member 2 (Transfer Learning & Edge Deployment) $\to$ Member 3 (Explainability & Adversarial Defense) $\to$ Member 4 (Application, Limitations & Closing)

---

### Slide 1: Title & Project Overview
- **Presenter:** Member 4 (Application & System Integration)
- **Key Points:**
  - Introducing **SignSense**: An end-to-end traffic sign perception system benchmarked on 43 GTSRB classes (51,839 images).
  - Four integrated pillars: Custom CNN from scratch, Transfer Learning (ResNet-18 & MobileNetV3), Explainability (Grad-CAM), and Adversarial Hardening (FGSM defense).
  - Production-ready edge deployment via ONNX Runtime (<1 ms CPU latency) and an interactive 5-tab Gradio web application.
- **Visual Asset:** `docs/screenshots/recognise.png`

---

### Slide 2: Autonomous Driving Context & Core Challenges
- **Presenter:** Member 4 (Application & System Integration)
- **Key Points:**
  - Autonomous vehicle safety relies on instant, accurate perception of regulatory, warning, and prohibitory signs.
  - Key technical hurdles: extreme class imbalance (10.7x), physical occlusions, motion blur, and subtle inter-class visual similarities.
  - Safety hazard: deep neural networks are vulnerable to imperceptible adversarial noise and digital/physical tampering.
- **Visual Asset:** `docs/screenshots/about.png`

---

### Slide 3: GTSRB Dataset & Class Distribution Analysis
- **Presenter:** Member 1 (Data Engineering & Custom CNN)
- **Key Points:**
  - Evaluated on the benchmark GTSRB dataset: 39,209 training images and 12,630 held-out official test images across 43 classes.
  - Imbalance analysis reveals a 10.71x ratio between Class 2 (50 km/h: 2,250 samples) and Class 0 (20 km/h: 210 samples).
  - Justifies using Macro-F1 evaluation alongside top-1 accuracy, and incorporating label smoothing ($\alpha = 0.1$) into training.
- **Visual Asset:** `outputs/eda_class_distribution.png`

---

### Slide 4: Resolution Standardization & Directional Augmentation
- **Presenter:** Member 1 (Data Engineering & Custom CNN)
- **Key Points:**
  - Raw image sizes range from $25 \times 25$ to $243 \times 225$ px (mean ~50 px); inputs standardized to uniform $64 \times 64 \times 3$.
  - Affine augmentations applied: random rotation ($\pm 10^\circ$), translation ($\pm 6\%$), scaling, shear, and color jitter.
  - **Strict rule:** Flips (horizontal/vertical) are strictly forbidden because reflection reverses directional semantics (e.g., *Turn left* becomes *Turn right*).
- **Visual Asset:** `outputs/eda_augmentation.png`

---

### Slide 5: Custom CNN: Architectural Design from Scratch
- **Presenter:** Member 1 (Data Engineering & Custom CNN)
- **Key Points:**
  - 4-stage VGG-style deep architecture totaling 0.824M parameters ($32 \to 64 \to 128 \to 256$ feature channels).
  - Each stage utilizes paired $3 \times 3$ convolutions, Batch Normalization, ReLU activations, $2 \times 2$ MaxPool, and progressive Dropout ($0.1 \to 0.4$).
  - Weights initialized via Kaiming (He) normal initialization; image normalization embedded internally via `Normalized` wrapper module.
- **Visual Asset:** `outputs/eda_samples.png`

---

### Slide 6: Custom CNN Training Dynamics & High Performance
- **Presenter:** Member 1 (Data Engineering & Custom CNN)
- **Key Points:**
  - Optimized with AdamW ($\text{lr} = 10^{-3}$, $\text{weight\_decay} = 10^{-4}$) and OneCycle learning rate policy over 15 epochs.
  - Achieves **99.59% test accuracy** and **0.9960 Macro-F1** on the 12,630 held-out test images.
  - Confusion matrix shows near-perfect diagonal concentration; remaining rare errors stem from heavy motion blur and low contrast.
- **Visual Asset:** `outputs/cnn_curves.png` (or `outputs/cnn_confusion.png`)

---

### Slide 7: Transfer Learning with ResNet-18
- **Presenter:** Member 2 (Transfer Learning & Edge Deployment)
- **Key Points:**
  - Evaluated ResNet-18 (11.20M parameters) initialized with ImageNet-1k pretrained weights and fine-tuned for 8 epochs.
  - Identity skip connections ($\mathcal{F}(x) + x$) prevent vanishing gradients and provide rapid convergence (>95% accuracy in epoch 1).
  - Reaches **98.99% test accuracy**; slightly lower than Custom CNN because $32\times$ downsampling reduces $64 \times 64$ images to a $2 \times 2$ spatial grid, losing fine symbol pictograms.
- **Visual Asset:** `outputs/resnet18_confusion.png`

---

### Slide 8: MobileNetV3-Small: Neural Architecture for Edge Devices
- **Presenter:** Member 2 (Transfer Learning & Edge Deployment)
- **Key Points:**
  - Designed for resource-constrained in-car ECUs: 1.56M parameters and only **11.1 MFLOPs** ($28\times$ fewer FLOPs than Custom CNN).
  - Employs depthwise separable convolutions (decoupling spatial filtering from channel mixing) with Squeeze-and-Excitation attention.
  - Reaches **95.05% test accuracy** and 0.9302 Macro-F1 in 10 epochs, balancing lightweight compute with high accuracy.
- **Visual Asset:** `outputs/mobilenet_curves.png`

---

### Slide 9: Edge Deployment: ONNX Export & CPU Runtime Benchmark
- **Presenter:** Member 2 (Transfer Learning & Edge Deployment)
- **Key Points:**
  - All four models exported to ONNX format with dynamic batch dimensions and op-level graph optimizations.
  - ONNX Runtime CPU benchmark: MobileNetV3 executes in **0.74 ms/image** (**7.94x speedup** over PyTorch CPU), achieving $>1,350$ FPS.
  - Custom CNN and Robust CNN execute in ~2.8 ms (3.30 MB footprint), well within automotive real-time deadlines (<10 ms).
- **Visual Asset:** `outputs/latency_vs_accuracy.png`

---

### Slide 10: The Parameters vs. FLOPs vs. Latency Paradox
- **Presenter:** Member 2 (Transfer Learning & Edge Deployment)
- **Key Points:**
  - *Paradox 1 (Params vs. FLOPs):* Custom CNN has 13.6x fewer parameters than ResNet-18 (0.82M vs 11.2M), yet more FLOPs (310M vs 296M), because it preserves larger spatial maps.
  - *Paradox 2 (FLOPs vs. GPU Latency):* MobileNetV3 has 28x fewer FLOPs than Custom CNN, but on GPU at batch 1 it is slower (5.16 ms vs 1.00 ms) due to low arithmetic intensity and memory bandwidth bounds.
  - *Resolution:* On CPU with ONNX Runtime, MobileNetV3 is by far the fastest (0.74 ms), proving that hardware-aware benchmarking is essential.
- **Visual Asset:** `outputs/onnx_benchmark.csv` (table summary slide)

---

### Slide 11: Visual Interpretability via Grad-CAM
- **Presenter:** Member 3 (Explainability & Adversarial Defense)
- **Key Points:**
  - Computed Gradient-weighted Class Activation Mapping (Grad-CAM) on the final convolutional layer activations.
  - Verification: Models attend squarely to inner pictograms (e.g., digits "50", pedestrian silhouettes) rather than background foliage or sky.
  - Misclassification analysis: When the model errs, heatmaps diffuse across borders, providing an interpretable failure diagnostic.
- **Visual Asset:** `outputs/cnn_gradcam.png`

---

### Slide 12: Adversarial Vulnerability: The FGSM Attack
- **Presenter:** Member 3 (Explainability & Adversarial Defense)
- **Key Points:**
  - Implemented Fast Gradient Sign Method (FGSM): $x_{\text{adv}} = \text{clip}(x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}))$.
  - Vulnerability: At $\epsilon = 0.02$ (imperceptible noise, $\approx 5/255$), standard CNN accuracy drops from **99.59% to 82.89%**; MobileNet drops to **64.06%**.
  - In autonomous driving, uncalibrated high confidence on perturbed signs (e.g., misclassifying *Stop* as *Speed limit 80*) is catastrophic.
- **Visual Asset:** `docs/screenshots/attack.png`

---

### Slide 13: Adversarial Defense: Min-Max Adversarial Training
- **Presenter:** Member 3 (Explainability & Adversarial Defense)
- **Key Points:**
  - Developed `cnn_robust` via joint min-max empirical loss minimization: $\mathcal{L} = 0.5 \mathcal{L}_{\text{clean}} + 0.5 \mathcal{L}_{\text{adv}}$ at $\epsilon = 0.02$.
  - Defense results: Under FGSM attack ($\epsilon=0.02$), robustness jumps from **82.89% to 98.71%** (+15.8% gain).
  - **Zero accuracy penalty:** Clean test accuracy remains state-of-the-art at **99.69%** (Macro-F1: 0.9964), showing adversarial training acts as a regularizer.
- **Visual Asset:** `outputs/fgsm_robustness.png`

---

### Slide 14: Interactive Application: 5-Tab Gradio Dashboard
- **Presenter:** Member 4 (Application & System Integration)
- **Key Points:**
  - Integrated full inference pipeline into a responsive 5-tab Gradio web interface (`app.py`).
  - **Tab 1 (Recognise):** Real-time single sign prediction, top-5 probabilities, and Grad-CAM attention heatmap.
  - **Tab 2 (Compare):** Multi-model side-by-side diagnosis with an Ensemble Consensus Vote across all 4 architectures.
  - **Tab 3 (Fool the CNN):** Interactive $\epsilon$-slider demonstrating live adversarial noise injection and robust defense.
- **Visual Asset:** `docs/screenshots/compare.png`

---

### Slide 15: Real-World Testing, Domain Shift & Limitations
- **Presenter:** Member 4 (Application & System Integration)
- **Key Points:**
  - Tested on unconstrained Wikimedia Commons real-world photos; confidence drops to 10–35% due to background clutter and perspective skew.
  - Demonstrates that standalone classifiers must be paired with upstream object detectors (e.g., YOLOv8 / RT-DETR) in production.
  - Current system is specialized for German signs; international generalization requires target-domain transfer learning.
- **Visual Asset:** `docs/screenshots/report.png`

---

### Slide 16: Summary of Contributions & Future Scope
- **Presenter:** Member 4 (Application & System Integration)
- **Key Points:**
  - Successfully demonstrated all four modules with reproducible code, Colab walkthrough, and GitHub repository.
  - Highlights: 99.69% accuracy on custom robust CNN, 0.74 ms CPU inference on MobileNetV3, and interpretable Grad-CAM heatmaps.
  - Future roadmap: Upstream YOLOv8 detector integration, multi-step PGD attack defense, and INT8 quantization for embedded ECUs.
- **Visual Asset:** `docs/screenshots/about.png`
