# SignSense: 15-Minute Faculty Demonstration Script

**Course:** PECE-06 Deep Learning, Sem VII, Project 13  
**Project:** SignSense — Traffic Sign Recognition with Explainability, Adversarial Defense, and Edge Deployment  
**Format:** Live interactive demonstration via Gradio interface (`http://127.0.0.1:7860`), accompanied by presentation slides.

---

## Demonstration Timeline Overview (15 Minutes)

| Time Window | Presenter | Module | Focus Area & Screen |
|---|---|---|---|
| **00:00 – 03:00** | **Member 4** | Integration & UI | System intro, project scope, architecture overview, Tab 1 walk-through |
| **03:00 – 06:00** | **Member 1** | Data & Custom CNN | GTSRB data, directional augmentation, 4-stage VGG design, training curves |
| **06:00 – 09:00** | **Member 2** | Transfer & Edge | Tab 2 (Compare Models), Ensemble vote, FLOPs paradox, ONNX CPU benchmark |
| **09:00 – 12:00** | **Member 3** | Explainability & Robustness | Grad-CAM verification, Tab 3 (Fool the CNN) live attack vs. defense |
| **12:00 – 15:00** | **Member 4** | Analysis & Conclusion | Tab 4 (Reports), real-world domain shift, limitations, wrap-up & faculty Q&A |

---

## Detailed Minute-by-Minute Script

### Act 1: Introduction & System Walkthrough (00:00 – 03:00)
**Presenter:** Member 4 (Application & System Integration)

- **UI Action:**
  1. Open browser to `http://127.0.0.1:7860`. Show the header: "SignSense: Traffic Sign Recognition".
  2. In **Tab 1: Recognise**, click on example thumbnail `samples/14_stop.png`.
  3. Ensure the model dropdown is set to `Custom CNN (scratch)`. Click **Classify**.
- **Visual Outcome:**
  - Prediction outputs: **Stop (Class 14)** with **99.98% confidence**.
  - Top-5 horizontal probability bar chart updates instantly.
  - Grad-CAM heatmap overlays the octagonal stop sign, highlighting the central letters "STOP".
- **Spoken Script (3–5 sentences):**
  > "Good morning, respected professors. Today our team presents **SignSense**, an end-to-end traffic sign recognition and safety perception system built for autonomous vehicles using PyTorch. In real-world driving, a perception model must be accurate, computationally light for in-car ECUs, robust against tampering, and interpretable to human safety engineers. As seen on our live dashboard, our system instantly classifies cropped road signs into 43 GTSRB classes, provides top-5 predictive distributions, and displays real-time Grad-CAM saliency maps. Our project is divided into four distinct modules: data engineering with custom CNNs, transfer learning with edge deployment, explainability and adversarial robustness, and full-stack integration."

---

### Act 2: Data Engineering & Custom CNN (03:00 – 06:00)
**Presenter:** Member 1 (Data Engineering & Custom CNN)

- **UI Action:**
  1. Navigate to **Tab 4: Model report**. Under *Select Report Category*, select **EDA & Data Distribution**.
  2. Display `outputs/eda_class_distribution.png` and `outputs/eda_augmentation.png`.
  3. Switch category to **Custom CNN (scratch)** and display `outputs/cnn_curves.png`.
- **Visual Outcome:**
  - Bar chart showing 10.71x class imbalance (from 210 to 2,250 samples).
  - Augmentation grid illustrating rotations, affine shears, and color jitter.
  - Smooth training convergence curves showing validation accuracy reaching ~99.6% over 15 epochs.
- **Spoken Script (3–5 sentences):**
  > "In Module 1, I addressed the core data pipeline and built our baseline CNN from scratch. GTSRB features 39,209 training and 12,630 test images with a severe 10.71x class imbalance, which we balanced using label smoothing and evaluated via Macro-F1. A critical domain-specific constraint in traffic sign recognition is that horizontal and vertical flips are strictly prohibited, as reflecting a directional sign transforms 'Turn Left' into 'Turn Right'. I engineered a 4-stage VGG-style CNN with 0.824M parameters featuring paired $3 \times 3$ convolutions, Batch Normalization, Dropout, and Kaiming initialization. Optimized via AdamW and OneCycleLR over 15 epochs, our custom model achieved **99.59% accuracy and 0.9960 Macro-F1** on the held-out test set."

---

### Act 3: Transfer Learning, Consensus & Edge Deployment (06:00 – 09:00)
**Presenter:** Member 2 (Transfer Learning & Edge Deployment)

- **UI Action:**
  1. Switch to **Tab 2: Compare models**.
  2. Click on sample image `samples/17_no_entry.png` (or `samples/2_speed_50.png`). Click **Compare All 4 Models**.
  3. Point to the four individual prediction columns, the individual inference latencies, and the **Ensemble Consensus Vote** box at the bottom.
  4. Bring up the ONNX benchmark slide or table on screen.
- **Visual Outcome:**
  - All four models predict *No entry* (or *Speed limit 50km/h*) with high confidence (>98%).
  - Latencies display ~1–3 ms on GPU / CPU.
  - Ensemble Consensus Vote combines softmax outputs into a single unanimous decision.
- **Spoken Script (3–5 sentences):**
  > "For Module 2, I investigated transfer learning, computational complexity, and edge deployment on embedded automotive hardware. We fine-tuned ResNet-18, reaching 98.99% accuracy, and adapted MobileNetV3-Small, which achieves 95.05% test accuracy with only 11.1 MFLOPs—a 28x computational reduction compared to standard CNNs. In Tab 2, our system runs all four architectures simultaneously and computes an ensemble consensus probability, mitigating single-model blind spots. To prepare for production deployment, I exported all models to ONNX and benchmarked them with ONNX Runtime CPU. On CPU, MobileNetV3 executes in just **0.74 ms per image**—yielding over 1,350 frames per second at an 8x speedup over PyTorch, making it ideal for cost-effective in-vehicle microcontrollers."

---

### Act 4: Explainability & Live Adversarial Defense (09:00 – 12:00)
**Presenter:** Member 3 (Explainability & Adversarial Defense)

- **UI Action:**
  1. Switch to **Tab 3: Fool the CNN**.
  2. Select sample image `samples/2_speed_50.png` (True class: *Speed limit 50km/h*).
  3. Drag the **Epsilon ($\epsilon$)** slider to **`0.02`**. Click **Generate Attack & Compare**.
  4. Point to the perturbed image and the enlarged residual noise map.
  5. Point to the side-by-side model outputs: Standard CNN vs. Robust CNN.
  6. *(Optional second example):* Select `samples/1_speed_30.png` at $\epsilon = 0.04$, showing Standard CNN predicting *Speed limit 120km/h* while Robust CNN retains *Speed limit 30km/h*.
- **Visual Outcome:**
  - Residual noise map highlights high-frequency adversarial gradient patterns.
  - **Standard CNN:** Fails! Predicts *Speed limit (80km/h)* with 87% confidence.
  - **Robust CNN:** Resists! Correctly identifies **Speed limit (50km/h)** with 96% confidence.
- **Spoken Script (3–5 sentences):**
  > "In Module 3, I focused on model interpretability and adversarial defense against perceptual deception. First, our Grad-CAM heatmaps verify that our models genuinely focus on the inner pictograms rather than background road artifacts. Next, to test safety, I implemented the Fast Gradient Sign Method (FGSM). As demonstrated here live, adding an imperceptible noise of $\epsilon = 0.02$ causes the standard CNN to mistake a 50 km/h sign for an 80 km/h sign—a fatal error for an autonomous vehicle. To defend against this, I implemented joint min-max adversarial training. Our resulting `cnn_robust` model easily resists the attack, correctly identifying the 50 km/h sign and boosting FGSM test robustness from **82.89% to 98.71%**, with zero degradation in nominal clean accuracy."

---

### Act 5: Real-World Testing, System Analysis & Wrap-Up (12:00 – 15:00)
**Presenter:** Member 4 (Application & System Integration)

- **UI Action:**
  1. Return to **Tab 1: Recognise**.
  2. Upload one uncropped real-world photo from `samples/real/real_stop.png` (a real stop sign with tree and street background).
  3. Run classification and show the lower confidence score (~25–35%).
  4. Switch to **Tab 4: Model report** and highlight the overall comparison plot `outputs/latency_vs_accuracy.png`.
  5. Conclude the presentation and invite questions.
- **Visual Outcome:**
  - Real-world photo demonstrates domain shift (background clutter dilutes confidence).
  - Latency vs. accuracy scatter plot shows trade-off spectrum: MobileNet for speed (0.74 ms) vs. Custom Robust CNN for peak accuracy (99.69%).
- **Spoken Script (3–5 sentences):**
  > "To evaluate real-world readiness, we also tested unconstrained real-world street photographs from Wikimedia Commons. As expected, without prior cropping, confidence drops due to background domain shift, confirming that in a complete autonomous vehicle stack, SignSense must be paired with an upstream object detector like YOLOv8. In summary, our project delivers an end-to-end verified system: a custom CNN achieving 99.59% accuracy, an adversarially hardened model reaching 98.71% robustness under attack, and an edge-optimized MobileNet running at 0.74 ms on CPU. All code, checkpoints, and documentation are publicly available on our GitHub repository. We now warmly invite your questions."

---

## Contingency Backup Plan ("What If the Live Demo Fails?")

If the local web server, browser, or hardware experiences an unexpected glitch during the presentation, immediately execute the following fallback protocol:

1. **Backup Level 1: Static Pre-Rendered Screenshots:**
   - All tabs and UI states are pre-captured at high resolution in `docs/screenshots/`:
     - Tab 1: `docs/screenshots/recognise.png`
     - Tab 2: `docs/screenshots/compare.png`
     - Tab 3: `docs/screenshots/attack.png`
     - Tab 4: `docs/screenshots/report.png`
     - Tab 5: `docs/screenshots/about.png`
   - Open these image files directly in the Windows photo viewer to demonstrate functionality without pausing the talk.

2. **Backup Level 2: Google Colab Story Notebook:**
   - Navigate to the cloud notebook:  
     `https://colab.research.google.com/github/Madhura1705/SignSense-Traffic-Sign-Recognition/blob/main/notebooks/SignSense_Train_Colab.ipynb`
   - The notebook contains pre-executed cells displaying all four model architectures, test metrics, Grad-CAM heatmaps, and adversarial attack cells.

3. **Backup Level 3: Terminal Quick Verification:**
   - Run the automated test script in PowerShell:
     ```powershell
     .venv\Scripts\python scripts/verify_models.py
     ```
   - Proves live evaluation of all four `.pt` checkpoints on 500 test images in under 10 seconds.
