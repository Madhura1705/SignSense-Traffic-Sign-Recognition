# SignSense: 15-Minute Faculty Demonstration Script

**Course:** PECE-06 Deep Learning, Sem VII, Project 13  
**Project:** SignSense — Traffic Sign Recognition with Explainability, Adversarial Defense, and Edge Deployment  
**Format:** Live interactive demonstration via custom FastAPI + Vanilla ES Modules web application (`http://127.0.0.1:7860`), accompanied by presentation slides.

---

## Demonstration Timeline Overview (15 Minutes)

| Time Window | Presenter | Module | Focus Area & Screen |
|---|---|---|---|
| **00:00 – 03:00** | **Member 4** | Integration & UI | System intro, project scope, Home live road ticker, Recognise page walkthrough |
| **03:00 – 06:00** | **Member 1** | Data & Custom CNN | GTSRB data, directional augmentation, 4-stage VGG design, Report $\to$ Training & Dataset tabs |
| **06:00 – 09:00** | **Member 2** | Transfer & Edge | Compare page, 4-model cards, consensus vote, FLOPs paradox, Report $\to$ Edge & ONNX tab |
| **09:00 – 12:00** | **Member 3** | Explainability & Robustness | Grad-CAM verification, Attack Lab page live attack presets, triplet view, robustness sweep |
| **12:00 – 15:00** | **Member 4** | Analysis & Conclusion | Real-world photo domain shift, Report $\to$ Errors tab (43-class search), wrap-up & faculty Q&A |

---

## Detailed Minute-by-Minute Script

### Act 1: Introduction & System Walkthrough (00:00 – 03:00)
**Presenter:** Member 4 (Application & System Integration)

- **UI Action:**
  1. Open browser to `http://127.0.0.1:7860/#/`.
  2. Point out the dark asphalt road navbar with sliding amber lane divider and green backend health monitor.
  3. Show the **Hero "Live Road" Ticker**:
     - Point out the real-time simulation advancing traffic signs every 3 seconds.
     - Toggle the **Grad-CAM Overlay** switch to demonstrate real-time saliency heatmaps on the ticker sign.
     - Hover over or click **Pause ticker** to demonstrate reduced-motion compliance.
  4. Point to the **Speed-Limit Roundels Metric Strip**: 99.7% best test accuracy, 98.7% robust accuracy under attack, 28x fewer FLOPs (MobileNetV3), and 43 distinct classes.
  5. Navigate to **Recognise** (`#/recognise`):
     - Click sample thumbnail `14_stop.png` from the GTSRB samples strip.
     - Ensure model selector is on `Custom CNN (Baseline)`.
     - Point out the instant classification: **Stop (Class 14)** with **99.98% confidence** and verified ground truth match.
     - Adjust the **Grad-CAM Opacity Slider** from 0% (raw image) to 100% (full Turbo heatmap), showing focus on the internal "STOP" lettering.
     - Show the **Top-5 Predicted Distribution** animated bars.
     - *(Optional):* Switch segmented control from "Upload image" to **"Live camera"** to show real-time 500 ms throttled webcam classification with center targeting box.
- **Spoken Script (3–5 sentences):**
  > "Good morning, respected professors. Today our team presents **SignSense**, an end-to-end traffic sign recognition and safety perception system engineered for autonomous vehicles. Our production stack replaces standard black-box wrappers with a lightweight, high-performance web architecture: a FastAPI backend serving in-memory PyTorch inference coupled with a responsive, vanilla ES module frontend with zero npm dependencies. As seen on our live road dashboard, SignSense classifies German traffic signs across 43 classes in sub-5 milliseconds, provides layered Grad-CAM saliency maps with interactive opacity blending, and supports both uploaded imagery and real-time dashcam streams."

---

### Act 2: Data Engineering & Custom CNN (03:00 – 06:00)
**Presenter:** Member 1 (Data Engineering & Custom CNN)

- **UI Action:**
  1. Navigate to **Report** (`#/report`) and click on the **Dataset** sub-tab (`#/report/dataset`).
  2. Click **Class Distribution** to view the histogram, then click **Augmentation** to display `outputs/eda_augmentation.png` in the zoomable Lightbox.
  3. Switch sub-tab to **Training** (`#/report/training`):
     - Select **Custom CNN (Baseline)**.
     - Point to the interactive Chart.js line charts for **Loss Convergence** and **Accuracy Progression**.
     - Point to the yellow callout note explaining GTSRB track-sequence leakage.
     - Click **View Static Plot in Lightbox** to inspect high-resolution multi-epoch curves.
- **Visual Outcome:**
  - Bar chart showing 10.71x class imbalance (from 210 to 2,250 samples per category).
  - Augmentation grid illustrating random affine rotations, scale, perspective, and color jitter (highlighting why horizontal/vertical flips are strictly forbidden).
  - Smooth training curves showing validation loss plateauing around epoch 10 and test accuracy achieving 99.59%.
- **Spoken Script (3–5 sentences):**
  > "In Module 1, I engineered our data ingestion pipeline and developed our baseline CNN from scratch. GTSRB exhibits severe class imbalance—ranging from 210 images in Class 0 to 2,250 in Class 2—which we counteracted using label smoothing and evaluated using Macro-F1. In autonomous driving, horizontal and vertical flips must be forbidden because reflecting a directional sign reverses its meaning, turning 'Keep Right' into 'Keep Left'. I architected a 4-stage VGG-style CNN with 0.824M parameters featuring paired $3 \times 3$ convolutions, Batch Normalization, spatial Dropout, and Kaiming initialization. Trained with OneCycleLR and AdamW, our custom network achieves **99.59% test accuracy and 0.9960 Macro-F1** on the independent test set."

---

### Act 3: Transfer Learning, Consensus & Edge Deployment (06:00 – 09:00)
**Presenter:** Member 2 (Transfer Learning & Edge Deployment)

- **UI Action:**
  1. Switch to **Compare Models** (`#/compare`).
  2. Click sample thumbnail `17_no_entry.png` (or `2_speed_50.png`).
  3. Point out the **Ensemble Consensus Banner** at the top: Unanimous consensus across all 4 architectures.
  4. Review the **4-Model Responsive Grid**:
     - Custom CNN (0.82M params, 310 MFLOPs)
     - ResNet-18 (11.20M params, 296 MFLOPs)
     - MobileNetV3-Small (1.56M params, **11.1 MFLOPs**)
     - Robust CNN (0.82M params, 310 MFLOPs)
  5. Inspect the **Confidence per Model** Chart.js horizontal bar chart.
  6. Review the **Speed vs Size Trade-off Table** highlighting MobileNet's 28x FLOP reduction.
  7. Navigate to **Report $\to$ Edge & ONNX** (`#/report/edge`):
     - Review the ONNX runtime benchmark table: MobileNet achieves **0.74 ms CPU inference** (7.94x speed-up vs PyTorch).
     - Click **Lightbox** on the Latency vs Accuracy trade-off bubble plot.
- **Visual Outcome:**
  - Unanimous consensus banner with green highlight.
  - Sub-3 ms individual inference latencies on CPU.
  - Clear architectural explanation of depthwise separable convolution trade-offs.
- **Spoken Script (3–5 sentences):**
  > "In Module 2, I explored transfer learning, residual networks, and edge acceleration for resource-constrained automotive ECUs. We adapted ResNet-18 (98.99% accuracy) and MobileNetV3-Small, which achieves 95.05% test accuracy with only 11.1 MFLOPs—a **28x reduction in computational complexity**. In our Compare page, the system runs all four models in parallel and computes an ensemble consensus vote to eliminate single-architecture blind spots. To target deployment on production vehicles, I exported the models to ONNX and benchmarked them with ONNX Runtime. Under ONNX Runtime CPU, MobileNet executes in **0.74 milliseconds**—equivalent to over 1,350 frames per second at an 8x speedup over PyTorch, making it ideal for low-power edge microcontrollers."

---

### Act 4: Explainability & Live Adversarial Defense (09:00 – 12:00)
**Presenter:** Member 3 (Explainability & Adversarial Defense)

- **UI Action:**
  1. Switch to **Attack Lab** (`#/attack`).
  2. Point out the warning prohibition header and the mathematical FGSM formula: $x' = x + \epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y))$.
  3. Click **Preset 1: Speed 50 fooled (ε=0.02)**.
  4. Point to the **Triplet Image Display**:
     - Left: Original clean image (*Speed limit 50km/h*).
     - Middle: Amplified adversarial perturbation noise (click to enlarge in Lightbox).
     - Right: Perturbed adversarial image (perceptually indistinguishable to human eye).
  5. Point to the **Verdict Banner**: 🚨 Fooled! Baseline CNN mistakes it for *Speed limit 80km/h* with 87% confidence.
  6. Show the **Transfer Attack Table**:
     - Baseline CNN: Fooled (80 km/h)
     - ResNet-18: Resistant (50 km/h)
     - MobileNetV3: Fooled (30 km/h)
     - **Robust CNN (cnn_robust):** Resisted! Confidently identifies *Speed limit 50km/h*.
  7. Click **Run Robustness Sweep (ε = 0 to 0.10)** button:
     - Watch the dynamic Chart.js curve plot across 6 perturbation steps.
     - Highlight the automated takeaway note: Robust CNN maintains 98.7% accuracy at $\epsilon = 0.02$ vs 82.9% for Baseline CNN.
  8. Navigate to **Report $\to$ Explainability** (`#/report/explainability`):
     - Review the 5-step mathematical derivation of Grad-CAM equations.
- **Visual Outcome:**
  - Live adversarial noise generation and transferability visualization.
  - Clear demonstration of adversarial vulnerability and hardening success.
- **Spoken Script (3–5 sentences):**
  > "In Module 3, I addressed model transparency and safety against adversarial exploitation. First, our Grad-CAM saliency heatmaps verify that our networks localize on inner sign numerals rather than roadside foliage. Next, to evaluate vulnerability, I implemented the Fast Gradient Sign Method (FGSM). As demonstrated in our Attack Lab, imperceptible noise at $\epsilon = 0.02$ tricks the standard CNN into misreading a 50 km/h speed limit as an 80 km/h sign—a catastrophic hazard for an autonomous vehicle. To defend against this, I implemented joint min-max adversarial training. Our resulting `cnn_robust` model easily resists the attack, boosting FGSM accuracy from **82.89% to 98.71%**, while preserving 99.69% clean test accuracy."

---

### Act 5: Real-World Testing, Error Analysis & Wrap-Up (12:00 – 15:00)
**Presenter:** Member 4 (Application & System Integration)

- **UI Action:**
  1. Return to **Recognise** (`#/recognise`).
  2. Under the samples strip, click the real-world sample `real_stop.png` (an uncropped natural photograph taken from Wikimedia Commons with background street trees).
  3. Point out the prediction outcome: Notice that while the top class is still *Stop*, confidence drops to ~30–40% due to background context and uncropped aspect ratio.
  4. Navigate to **Report $\to$ Errors** (`#/report/errors`):
     - Click **Confusion Matrix** to view the $43 \times 43$ normalized error patterns.
     - Click **Most Confident Mistakes** to view high-confidence edge case failures.
     - Click **Per-Class Scores (43 Classes)**:
       - Type "Speed limit" into the search box to filter classes dynamically.
       - Point out the 5 worst-performing classes highlighted in amber (`.worst-class-row`).
  5. Navigate to **Report $\to$ Overview** (`#/report/overview`):
     - Click column headers to demonstrate interactive table sorting.
     - Review the clean vs adversarial accuracy grouped bar chart.
  6. Conclude the demonstration and open the floor for faculty questions.
- **Visual Outcome:**
  - Real-world domain shift observation.
  - Detailed error analysis with searchable per-class scores.
  - Polished responsive dashboard summary.
- **Spoken Script (3–5 sentences):**
  > "To conclude, Module 4 integrated our research models into an enterprise-grade, accessible web system. In our real-world testing with uncropped photos, we observed how background foliage and perspective distortion cause confidence degradation, underscoring that in production, our classifier must be paired with an upstream detector like YOLOv8. Our modular Report section exposes full transparency: from confusion matrices and searchable per-class F1 scores to edge ONNX benchmarks. In summary, SignSense demonstrates that high accuracy, explainability, adversarial resilience, and edge efficiency can be successfully unified in a real-time safety perception system. Thank you, and we welcome your questions."
