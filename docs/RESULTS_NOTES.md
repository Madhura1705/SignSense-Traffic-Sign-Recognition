# SignSense: Results & Analysis Notes

This document provides detailed observations for every generated plot in `outputs/` and synthesizes the core deep learning findings for the project.

---

## 1. Summary of Quantitative Results

### Test Set Performance (12,630 images)
| Model | Parameters | MFLOPs (64x64) | Test Accuracy | Macro F1 | FGSM Accuracy ($\epsilon = 0.02$) | GPU Latency (batch=1) |
|---|---|---|---|---|---|---|
| **Custom CNN (scratch)** | **0.824 M** | 310.1 | **99.59%** | **0.9960** | 82.89% | 0.999 ms |
| **ResNet-18 (transfer)** | 11.199 M | 296.1 | 98.99% | 0.9831 | 94.61% | 2.289 ms |
| **MobileNetV3-Small (edge)** | 1.562 M | **11.1** | 95.05% | 0.9302 | 64.06% | 5.156 ms |
| **Robust CNN (adv. train)** | **0.824 M** | 310.1 | **99.69%** | **0.9964** | **98.71%** | 1.037 ms |

### ONNX Edge Deployment Benchmark (CPU)
| Model | ONNX Size | PyTorch CPU Latency | ONNX Runtime CPU Latency | Speed-Up | Max Output Difference | Test Accuracy |
|---|---|---|---|---|---|---|
| **Custom CNN** | 3.30 MB | 7.00 ms | 2.93 ms | 2.39x | $1.19 \times 10^{-6}$ | 99.59% |
| **Robust CNN** | 3.30 MB | 7.17 ms | 2.79 ms | 2.57x | $1.31 \times 10^{-6}$ | 99.69% |
| **MobileNetV3-Small** | 6.26 MB | 5.87 ms | **0.74 ms** | **7.94x** | $3.10 \times 10^{-6}$ | 95.05% |
| **ResNet-18** | 44.79 MB | 13.62 ms | 5.21 ms | 2.62x | $9.54 \times 10^{-7}$ | 98.99% |

---

## 2. Key Technical Findings & Viva Points

1. **Custom CNN beats ResNet-18 at 64x64:**
   - The Custom CNN achieves **99.59%** accuracy vs. ResNet-18's **98.99%**, despite having 13.6x fewer parameters.
   - *Why?* ResNet-18 was engineered for $224 \times 224$ ImageNet inputs and applies an aggressive $32\times$ downsampling factor across its 5 residual stages ($64 \to 32 \to 16 \to 8 \to 4 \to 2$). On a $64 \times 64$ traffic sign, the feature map shrinks to a tiny $2 \times 2$ grid before Global Average Pooling, discarding crucial high-frequency spatial symbol geometry (e.g. inner speed digits). The Custom CNN downsamples only $8\times$ (to $8 \times 8$, adaptive pooled to $4 \times 4$), retaining rich structural spatial features.

2. **Parameters vs. FLOPs vs. Latency Paradox (MobileNet vs. CNN):**
   - MobileNetV3-Small uses **$28\times$ fewer FLOPs** than the Custom CNN (11.1 vs. 310.1 MFLOPs), yet on GPU at batch size 1 it is slower (5.16 ms vs. 1.00 ms).
   - *Why?* Depthwise-separable convolutions divide a standard convolution into depthwise and pointwise operations, minimizing arithmetic operations but drastically lowering arithmetic intensity (FLOPs per byte of memory loaded). On high-bandwidth GPUs at batch size 1, depthwise layers become memory-bound rather than compute-bound.
   - Conversely, on CPU with ONNX Runtime, MobileNetV3 achieves an astonishing **0.74 ms/image** ($1350+$ FPS) with a 7.94x runtime speed-up, proving it is ideal for embedded edge CPUs.

3. **Adversarial Training Defense ($82.9\% \to 98.7\%$):**
   - Under FGSM adversarial noise with strength $\epsilon = 0.02$, the standard Custom CNN drops to 82.89% test accuracy.
   - Training with an adversarial loss mixture ($0.5 \mathcal{L}_{\text{clean}} + 0.5 \mathcal{L}_{\text{FGSM}}$) enables `cnn_robust` to reach **98.71% accuracy under attack**, with **zero loss in clean accuracy** (99.69% clean test accuracy).
   - Adversarial training acts as a powerful regularizer, smoothing decision boundaries around the data manifold.

4. **Validation Accuracy (~100%) vs. Test Accuracy (95–99.7%):**
   - Validation accuracy approaches ~100% early in training because GTSRB consists of short video track recordings (30 consecutive frames of the same sign approaching). A random train/val split inevitably places adjacent video frames from the same track into both subsets.
   - Therefore, the held-out **12,630-image test set** (captured under separate physical driving tracks, lighting, and angles) is the only true, unbiased benchmark.

5. **Parameters vs. FLOPs Viva Point:**
   - The Custom CNN has more FLOPs than ResNet-18 (310.1 vs. 296.1 MFLOPs) even though ResNet-18 has 13.6x more parameters (11.2M vs. 0.82M). This occurs because the custom CNN maintains larger spatial dimensions ($64 \to 32 \to 16 \to 8$) across multiple $3 \times 3$ convolutional blocks, whereas ResNet rapidly reduces spatial dimensions.

---

## 3. Analysis of Every Output Plot

### Exploratory Data Analysis (EDA)
- **`outputs/eda_class_distribution.png`**: Highlights the substantial class imbalance in GTSRB (imbalance ratio of 10.71x), ranging from 210 images for Class 0 (Speed limit 20km/h) up to 2,250 images for Class 2 (Speed limit 50km/h), justifying the use of label smoothing and Macro-F1 evaluation.
- **`outputs/eda_samples.png`**: Displays one authentic visual tile for each of the 43 GTSRB classes, illustrating the high visual diversity and subtle inter-class distinctions between circular speed signs.
- **`outputs/eda_image_sizes.png`**: Shows histograms of raw image widths (25–243px, mean 50.8px) and heights (25–225px, mean 50.3px), demonstrating why standardizing to a uniform $64 \times 64$ input is necessary for batch training.
- **`outputs/eda_augmentation.png`**: Demonstrates the training augmentation pipeline (random affine transforms: rotation, shift, scale, shear, and color jitter) while visually confirming that horizontal/vertical flips are strictly excluded to preserve semantic meaning.

### Custom CNN (from scratch)
- **`outputs/cnn_curves.png`**: Shows clean convergence over 15 epochs; training loss steadily decreases with OneCycleLR while validation accuracy climbs rapidly without signs of severe overfitting.
- **`outputs/cnn_confusion.png`**: Shows near-perfect diagonal concentration across all 43 classes on the 12,630 test set, with minor confusion only among high-speed circular limits (e.g. 70 vs 80 km/h).
- **`outputs/cnn_gradcam.png`**: Confirms that the last convolutional layer concentrates activation intensity squarely over the inner sign symbol rather than background foliage or sky.
- **`outputs/cnn_misclassified.png`**: Displays the model's hardest errors, which primarily consist of severely motion-blurred, extreme low-light, or heavily occluded real-world sign crops.

### ResNet-18 (Transfer Learning)
- **`outputs/resnet18_curves.png`**: Pretrained ImageNet feature representations allow the model to reach >95% accuracy in the very first epoch, converging smoothly over 8 epochs.
- **`outputs/resnet18_confusion.png`**: Shows sharp diagonal accuracy (98.99% overall), with rare misclassifications isolated to low-contrast warning signs.
- **`outputs/resnet18_gradcam.png`**: Activations form broad attention fields around the perimeter and center of the signs, characteristic of deep residual networks with skip connections.
- **`outputs/resnet18_misclassified.png`**: The worst errors correlate with signs taken at oblique angles where the 32x downsampling erodes subtle pictogram boundaries.

### MobileNetV3-Small (Edge Model)
- **`outputs/mobilenet_curves.png`**: Demonstrates stable training over 10 epochs reaching 95.05% test accuracy with light footprint and low compute demands.
- **`outputs/mobilenet_confusion.png`**: Well-defined diagonal with slightly higher dispersion in classes with fewer training samples (e.g. Class 0, Class 24), reflecting the compact representation capacity.
- **`outputs/mobilenet_gradcam.png`**: Heatmaps from depthwise separable convolutions are tightly focused on high-contrast internal edges and contours.
- **`outputs/mobilenet_misclassified.png`**: Confusions center on signs with intricate internal iconography (such as Pedestrians vs. Road Work).

### Robust CNN (Adversarial Defense)
- **`outputs/cnn_robust_curves.png`**: Demonstrates that joint optimization on clean and FGSM-perturbed batches converges stably without destabilizing clean validation accuracy.
- **`outputs/cnn_robust_confusion.png`**: Clean confusion matrix matches the standard CNN (99.69% accuracy), verifying that adversarial defense does not degrade nominal recognition performance.
- **`outputs/cnn_robust_gradcam.png`**: Grad-CAM attention is notably more compact and centered on structural features, reflecting smoother gradient fields around the sign boundaries.
- **`outputs/cnn_robust_misclassified.png`**: Exhibits fewer false confidences compared to the standard CNN, producing better calibrated uncertainty.

### Cross-Model Robustness & Edge Deployment
- **`outputs/fgsm_robustness.png`**: Illustrates test accuracy as a function of adversarial perturbation epsilon ($\epsilon \in [0, 0.08]$); while standard CNN and MobileNet plunge rapidly under noise, `cnn_robust` maintains a high accuracy curve (>98% at $\epsilon=0.02$).
- **`outputs/latency_vs_accuracy.png`**: Scatter plot contrasting CPU inference latency against test accuracy, where bubble sizes represent model binary footprint (MB). Clearly illustrates MobileNetV3 as the latency winner (0.74 ms) and Custom CNN as the accuracy/size sweet spot (99.59% at 3.3 MB).
