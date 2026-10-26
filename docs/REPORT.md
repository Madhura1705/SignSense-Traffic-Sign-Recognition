# SignSense: Robust Traffic Sign Recognition System for Autonomous Driving
## Deep Learning Mini-Project Report (PECE-06 Deep Learning, Sem VII, Project 13)

---

### Executive Summary & Metadata
- **Project Title:** SignSense — Traffic Sign Recognition with Explainability, Adversarial Defense, and Edge Deployment
- **Dataset:** German Traffic Sign Recognition Benchmark (GTSRB, 43 classes, 51,839 images total)
- **Academic Program:** Department of Computer Engineering / Information Technology, Semester VII
- **Course Code:** PECE-06 Deep Learning
- **Repository:** [GitHub: Madhura1705/SignSense-Traffic-Sign-Recognition](https://github.com/Madhura1705/SignSense-Traffic-Sign-Recognition)
- **Notebook:** [Google Colab End-to-End Walkthrough](https://colab.research.google.com/github/Madhura1705/SignSense-Traffic-Sign-Recognition/blob/main/notebooks/SignSense_Train_Colab.ipynb)

---

## 1. Abstract

Autonomous vehicles, Advanced Driver Assistance Systems (ADAS), and edge-enabled robotics demand ultra-reliable, real-time perception to correctly classify road signs under adverse illumination, geometric distortions, and physical or digital tampering. This project presents **SignSense**, a production-grade traffic sign recognition pipeline built upon the 43-class German Traffic Sign Recognition Benchmark (GTSRB). We investigate, compare, and deploy four neural architectures: (1) a custom 4-stage VGG-style Convolutional Neural Network (0.82M parameters) trained from scratch with Batch Normalization, Dropout, and Kaiming initialization; (2) a fine-tuned ResNet-18 residual network (11.20M parameters); (3) an inverted residual MobileNetV3-Small network (1.56M parameters, 11.1 MFLOPs) optimized for resource-constrained edge computing; and (4) an adversarially hardened CNN (`cnn_robust`) trained via joint Fast Gradient Sign Method (FGSM) min-max optimization. 

Evaluated on the official held-out test set of 12,630 authentic images, our Custom CNN achieves **99.59%** test accuracy (Macro-F1: 0.9960), outperforming ResNet-18 (**98.99%**), because standard ResNet downsamples $64 \times 64$ inputs $32\times$ into a degenerate $2 \times 2$ spatial grid prior to pooling. Under FGSM adversarial attacks ($\epsilon = 0.02$), standard CNN accuracy collapses to 82.89%, while our adversarially trained CNN maintains **98.71%** accuracy alongside a **99.69%** clean test accuracy. For embedded edge deployment, models are exported to Open Neural Network Exchange (ONNX) format with fused internal image normalization and evaluated via ONNX Runtime CPU. MobileNetV3 achieves **0.74 ms** inference latency ($>1,350$ FPS) with an ONNX footprint of only 6.26 MB, representing a 7.94x acceleration over native PyTorch CPU. The entire system is packaged into an interactive 5-tab Gradio web application providing single-image diagnosis, multi-model consensus voting, live adversarial perturbation exploration, Grad-CAM visual explainability, and automated model benchmarking.

---

## 2. Introduction & Problem Formulation

### 2.1 Context and Motivation
Traffic sign recognition (TSR) is an essential perceptual sub-module for Level 2+ Advanced Driver Assistance Systems and Level 4/5 autonomous vehicles. Reliable traffic sign classification is critical for dynamic velocity management (e.g., speed limit compliance), junction navigation (e.g., stop, yield, and roundabout signs), and hazard mitigation (e.g., road works, pedestrian crossings). Unlike generic object recognition datasets such as ImageNet or CIFAR-10, traffic signs feature distinct properties:
1. **Directional Semantics:** Orientation conveys vital driving instructions (e.g., *Turn left ahead* vs. *Turn right ahead*; *Keep right* vs. *Keep left*). Symmetric data augmentations like horizontal and vertical flips invert these semantics, causing fatal errors.
2. **Subtle Inter-Class Discriminability:** Classes within sign families share identical circular or triangular silhouettes and boundary colours, differing only by fine interior pictograms or digits (e.g., 30 km/h vs. 50 km/h vs. 80 km/h).
3. **Severe Environmental Degradation:** Real-world acquisition suffers from motion blur, severe backlight, physical occlusion by foliage, weather phenomena (rain, snow, fog), and sensor noise.
4. **Adversarial Susceptibility:** Deep neural networks are known to be brittle to imperceptible input perturbations (adversarial noise or physical stickers) that can induce arbitrary misclassifications with high confidence.
5. **Real-Time Edge Constraints:** In-vehicle electronic control units (ECUs) possess restricted compute budgets, requiring sub-10 ms latency and compact memory footprints.

### 2.2 Project Scope & Objectives
The project is structured into four integrated modules distributed across team members:
- **Module 1 (M1): Data Engineering & Custom CNN Architecture.** Exploratory data analysis, directional-preserving augmentations, and a 4-stage VGG-style CNN trained from scratch with Batch Normalization, Dropout, He initialization, and OneCycleLR.
- **Module 2 (M2): Transfer Learning, Architecture Search & Edge Optimization.** Adaptation of ResNet-18 and MobileNetV3-Small, comparative analysis of the parameter-FLOPs-latency trade-off, ONNX graph export, and ONNX Runtime CPU benchmarking.
- **Module 3 (M3): Explainability & Adversarial Robustness.** Gradient-weighted Class Activation Mapping (Grad-CAM), Fast Gradient Sign Method (FGSM) adversarial attack formulation, and min-max adversarial training defense.
- **Module 4 (M4): System Integration, Interactive UI & Cloud Deployment.** Packaging into an interactive Gradio web app with ensemble voting, Colab story notebook, and automated documentation.

---

## 3. Literature Survey

The development of SignSense is grounded in seven foundational works across computer vision, deep learning optimization, model interpretability, and adversarial robustness:

1. **GTSRB Benchmark (Stallkamp et al., 2012):**  
   *Stallkamp, J., Schlipsing, M., Salmen, J., & Igel, C. (2012). Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition. Neural Networks, 32, 323-332.*  
   Introduced the German Traffic Sign Recognition Benchmark comprising 51,839 real-world images across 43 classes. Demonstrated that multi-scale convolutional neural networks (99.46% accuracy) could surpass human performance (98.84%), establishing the canonical dataset and video-track separation methodology for autonomous vehicle perception.

2. **Residual Learning Framework (He et al., 2016):**  
   *He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In CVPR (pp. 770-778).*  
   Formulated identity shortcut connections $\mathcal{F}(x) + x$ that mitigate vanishing/exploding gradients, enabling networks to exceed 100 layers. In this study, ResNet-18 serves as our representative residual transfer learning architecture.

3. **MobileNetV3 Edge Architectures (Howard et al., 2019):**  
   *Howard, A., et al. (2019). Searching for MobileNetV3. In ICCV (pp. 1314-1324).*  
   Combined automated neural architecture search (NAS), depthwise separable convolutions, squeeze-and-excitation attention blocks, and hard-swish non-linearities to minimize Multiply-Accumulate operations (MACs). We adopt MobileNetV3-Small for ultra-low latency automotive edge execution.

4. **Visual Interpretability via Grad-CAM (Selvaraju et al., 2017):**  
   *Selvaraju, R. R., Cogswell, M., Das, A., Vedaldi, A., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. In ICCV (pp. 618-626).*  
   Derived gradient-weighted class activation heatmaps by flowing task-specific classification gradients into the final convolutional feature maps, enabling visual verification that our models fixate on sign symbols rather than background context.

5. **Explaining and Harnessing Adversarial Examples (Goodfellow et al., 2015):**  
   *Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and harnessing adversarial examples. In ICLR.*  
   Demonstrated that neural network vulnerability stems from high-dimensional linearity rather than extreme non-linearity, formulating the single-step Fast Gradient Sign Method (FGSM) under an $L_\infty$ perturbation bound to generate adversarial samples.

6. **Batch Normalization (Ioffe & Szegedy, 2015):**  
   *Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating deep network training by reducing internal covariate shift. In ICML (pp. 448-456).*  
   Introduced mini-batch mean and variance normalization with learnable affine scaling ($\gamma, \beta$), stabilizing layer input distributions, enabling higher learning rates, and acting as an effective regularizer.

7. **Adversarially Robust Optimization (Madry et al., 2018):**  
   *Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). Towards deep learning models resistant to adversarial attacks. In ICLR.*  
   Cast adversarial defense as a principled empirical risk min-max game: minimizing empirical loss under worst-case inner bounded perturbations, forming the theoretical justification for our `cnn_robust` joint training strategy.

---

## 4. Dataset Description & Exploratory Data Analysis (M1)

### 4.1 Dataset Properties
The German Traffic Sign Recognition Benchmark (GTSRB) contains 51,839 cropped color images categorized into 43 traffic sign classes. 
- **Training Set:** 39,209 images organized across 43 class subdirectories.
- **Held-Out Test Set:** 12,630 images collected from distinct video tracks under varied lighting, angles, and camera hardware.
- **Internal Split:** Training images were split into 31,367 training samples (80%) and 7,842 validation samples (20%).

### 4.2 Exploratory Findings
1. **Severe Class Imbalance:** The dataset exhibits an imbalance ratio of **10.71x**. The most frequent class, *Speed limit (50km/h)* (Class 2), contains 2,250 training images, whereas *Speed limit (20km/h)* (Class 0) contains only 210 images. This disparity requires macro-averaged metrics (Macro-F1) rather than micro-accuracy alone, alongside label smoothing regularization.
2. **Resolution Heterogeneity:** Raw crops vary from $25 \times 25$ pixels up to $243 \times 225$ pixels (mean width: 50.8 px, mean height: 50.3 px). Standardizing all inputs to a uniform $64 \times 64$ resolution balances spatial detail with batch computational efficiency.
3. **Video Track Frame Correlation:** GTSRB training data was gathered as 30-frame track segments of approaching signs. A naive random split leaks adjacent frames into both train and validation sets, inflating validation accuracy toward ~100%. The held-out 12,630-image test set remains the sole unbiased metric of true generalization.

### 4.3 Preprocessing & Data Augmentation Pipeline
To enhance robustness without distorting class identity:
- **Geometry Transforms:** Random rotation ($\pm 10^\circ$), random translation ($\pm 6\%$), random scaling ($[0.9, 1.1]$), and random shearing ($\pm 5^\circ$).
- **Photometric Transforms:** Color jitter adjusting brightness ($\pm 0.2$), contrast ($\pm 0.2$), and saturation ($\pm 0.2$).
- **Strict Prohibition of Flips:** Horizontal and vertical reflections are strictly excluded. For example, reflecting *Turn left* (Class 34) creates *Turn right* (Class 33), corrupting ground truth labels.
- **Tensor Format:** Images are mapped to tensors in range $[0.0, 1.0]$. Standardization (mean $[0.485, 0.456, 0.406]$, std $[0.229, 0.224, 0.225]$) is embedded directly within the network architecture via a `Normalized` wrapper module.

---

## 5. Methodology & Mathematical Formulations

```
+---------------------------------------------------------------------------------------------------------+
|                                           SIGNSENSE ARCHITECTURE                                        |
+---------------------------------------------------------------------------------------------------------+
                                                     |
                                   Input Image: x in [0, 1]^(3 x 64 x 64)
                                                     |
                                       +-------------v-------------+
                                       |     Normalized Module     |
                                       |  (mean / std subtraction) |
                                       +-------------+-------------+
                                                     |
                 +-------------------+---------------+-------------------+--------------------+
                 |                   |                                   |                    |
                 v                   v                                   v                    v
         [Module 1 (M1)]      [Module 2 (M2)]                     [Module 2 (M2)]      [Module 3 (M3)]
           Custom CNN            ResNet-18                         MobileNetV3-S          Robust CNN
         (from scratch)      (Transfer Learning)                     (Edge NAS)          (Adv Trained)
                 |                   |                                   |                    |
        ConvBlock1 (32)      Conv 7x7, Stride 2                  Conv 3x3 (16)        Identical custom
        ConvBlock2 (64)      Residual Stage 1 (64)               Inverted Bottleneck  CNN architecture;
        ConvBlock3 (128)     Residual Stage 2 (128)              Blocks with SE &     Joint loss:
        ConvBlock4 (256)     Residual Stage 3 (256)              Hard-Swish (576)     0.5 L_clean +
        AdaptivePool (4x4)   Residual Stage 4 (512)              Conv 1x1 + GAP       0.5 L_FGSM (eps=0.02)
        Linear (512 -> 43)   GAP (512 -> 43)                     Linear (1024 -> 43)          |
                 |                   |                                   |                    |
                 +-------------------+---------------+-------------------+--------------------+
                                                     |
                                                     v
                                      43-Class Logits: z in R^(43)
                                                     |
                                            Softmax Probabilities
                                                     |
                         +---------------------------+---------------------------+
                         |                                                       |
                         v                                                       v
                 [Module 3 (M3)]                                         [Module 4 (M4)]
              Explainability & Attacks                                 Deployment & Serving
                         |                                                       |
            - Grad-CAM Heatmap Analysis                             - ONNX Graph Export (CPU fp32)
            - FGSM Adversarial Crafting                             - ONNX Runtime Benchmark
            - Robustness Boundary Analysis                          - 5-Tab Gradio Dashboard
```

### 5.1 Module 1: Custom CNN Architecture (M1)
The custom CNN is engineered specifically for $64 \times 64$ spatial inputs. It comprises 4 sequential convolutional stages with doubling filter counts:
$$\text{Input } (3 \times 64 \times 64) \to \text{Stage 1 } (32) \to \text{Stage 2 } (64) \to \text{Stage 3 } (128) \to \text{Stage 4 } (256)$$

Each stage consists of two contiguous $3 \times 3$ 2D convolutions, each followed by Batch Normalization and ReLU activation, capped by a $2 \times 2$ MaxPool with stride 2 and a spatial Dropout layer:

#### Mathematical Formulations:
1. **Convolution Spatial Output Dimension:**  
   Given input spatial dimension $W$, kernel size $K$, padding $P$, and stride $S$, the output dimension $O$ is:
   $$O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$
   For $W=64$, $K=3$, $P=1$, $S=1$: $O = \lfloor \frac{64 - 3 + 2}{1} \rfloor + 1 = 64$ (spatial dimensions preserved until MaxPool).

2. **Batch Normalization (Ioffe & Szegedy, 2015):**  
   For a mini-batch $\mathcal{B} = \{x_1, \dots, x_m\}$, the layer computes mini-batch mean $\mu_\mathcal{B}$ and variance $\sigma_\mathcal{B}^2$, standardizing each activation before applying learnable scale $\gamma$ and shift $\beta$:
   $$\mu_\mathcal{B} = \frac{1}{m} \sum_{i=1}^m x_i, \quad \sigma_\mathcal{B}^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_\mathcal{B})^2$$
   $$\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon_{\text{BN}}}}, \quad y_i = \gamma \hat{x}_i + \beta$$
   During inference, fixed running exponential moving averages are substituted for $\mu_\mathcal{B}$ and $\sigma_\mathcal{B}^2$.

3. **Cross-Entropy Loss with Label Smoothing:**  
   To prevent overconfidence on imbalanced classes, one-hot ground-truth labels $y_k \in \{0, 1\}$ are smoothed with uniform distribution $\frac{1}{K}$ ($K=43$) weighted by parameter $\alpha = 0.1$:
   $$y'_k = (1 - \alpha) y_k + \frac{\alpha}{K}$$
   $$\mathcal{L}_{\text{CE}}(p, y') = - \sum_{k=1}^K y'_k \log p_k, \quad p_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$

4. **Kaiming (He) Initialization:**  
   Weights $W$ for layers preceding ReLU activations are sampled from $\mathcal{N}(0, \sigma^2)$ where $\sigma = \sqrt{\frac{2}{\text{fan\_in}}}$, preventing signal attenuation during forward propagation.

#### Detailed Custom CNN Layer Specification:
| Stage | Layer Component | Kernel / Stride / Pad | Output Shape | Parameters |
|---|---|---|---|---|
| **Input** | Tensor Image | — | $(3, 64, 64)$ | 0 |
| **Stage 1** | Conv2d + BatchNorm + ReLU | $3 \times 3, S=1, P=1$ | $(32, 64, 64)$ | 896 + 64 |
| | Conv2d + BatchNorm + ReLU | $3 \times 3, S=1, P=1$ | $(32, 64, 64)$ | 9,216 + 64 |
| | MaxPool2d + Dropout(0.1) | $2 \times 2, S=2$ | $(32, 32, 32)$ | 0 |
| **Stage 2** | Conv2d + BatchNorm + ReLU | $3 \times 3, S=1, P=1$ | $(64, 32, 32)$ | 18,432 + 128 |
| | Conv2d + BatchNorm + ReLU | $3 \times 3, S=1, P=1$ | $(64, 32, 32)$ | 36,864 + 128 |
| | MaxPool2d + Dropout(0.2) | $2 \times 2, S=2$ | $(64, 16, 16)$ | 0 |
| **Stage 3** | Conv2d + BatchNorm + ReLU | $3 \times 3, S=1, P=1$ | $(128, 16, 16)$ | 73,728 + 256 |
| | Conv2d + BatchNorm + ReLU | $3 \times 3, S=1, P=1$ | $(128, 16, 16)$ | 147,456 + 256 |
| | MaxPool2d + Dropout(0.3) | $2 \times 2, S=2$ | $(128, 8, 8)$ | 0 |
| **Stage 4** | Conv2d + BatchNorm + ReLU | $3 \times 3, S=1, P=1$ | $(256, 8, 8)$ | 294,912 + 512 |
| | Conv2d + BatchNorm + ReLU | $3 \times 3, S=1, P=1$ | $(256, 8, 8)$ | 210,000+ |
| **Head** | AdaptiveAvgPool2d | Target $(4, 4)$ | $(256, 4, 4)$ | 0 |
| | Flatten + Dropout(0.4) | — | $(4096)$ | 0 |
| | Linear Classifier | $4096 \to 512$ | $(512)$ | 2,097,664 |
| | Linear Output Head | $512 \to 43$ | $(43)$ | 22,059 |
| **Total** | — | — | — | **824,363 (~0.824M)** |

---

### 5.2 Module 2: Transfer Learning & Edge Optimization (M2)

#### Architectures Evaluated:
1. **ResNet-18 (He et al., 2016):** Pretrained ImageNet weights are adapted by replacing the final 1000-way linear classifier with a 43-way fully connected head. ResNet employs identity skip connections:
   $$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$
   This residual mapping ensures uninterrupted gradient backpropagation through 18 layers.
2. **MobileNetV3-Small (Howard et al., 2019):** Engineered for low-power mobile devices. It utilizes depthwise-separable convolutions, which decompose a standard convolution into:
   - A depthwise convolution applying a single $3 \times 3$ filter per input channel: $\text{Params} = K \times K \times C_{\text{in}}$.
   - A pointwise $1 \times 1$ convolution mixing channel features: $\text{Params} = 1 \times 1 \times C_{\text{in}} \times C_{\text{out}}$.
   The theoretical reduction in computational operations compared to standard convolution is:
   $$\text{Reduction} = \frac{K \cdot K \cdot C_{\text{in}} + C_{\text{in}} \cdot C_{\text{out}}}{K \cdot K \cdot C_{\text{in}} \cdot C_{\text{out}}} = \frac{1}{C_{\text{out}}} + \frac{1}{K^2}$$
   For $3 \times 3$ kernels, depthwise separable convolutions require approximately **8 to 9 times fewer FLOPs**.

#### ONNX Export & Edge Inference:
To enable real-time inference on edge CPUs without a full PyTorch runtime, models were exported to the Open Neural Network Exchange (ONNX) format using dynamic batch dimensions and opset version 14:
```python
torch.onnx.export(
    model, dummy_input, onnx_path,
    input_names=["input"], output_names=["logits"],
    dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    opset_version=14
)
```
The ONNX Runtime executes computational graph optimizations including constant folding, redundant node elimination, and hardware-specific SIMD vectorization (AVX-512).

---

### 5.3 Module 3: Explainability & Adversarial Defense (M3)

#### Grad-CAM (Selvaraju et al., 2017):
To verify that models rely on true sign semantics rather than contextual artifacts (e.g., surrounding foliage or sky), we compute Gradient-weighted Class Activation Mappings:
1. Compute the gradient of the unnormalized score for target class $c$ ($y^c$) with respect to feature activation map $A^k$ of the final convolutional layer:
   $$\frac{\partial y^c}{\partial A_{i,j}^k}$$
2. Spatially pool gradients across height $u$ and width $v$ to yield channel importance weights $\alpha_k^c$:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^u \sum_{j=1}^v \frac{\partial y^c}{\partial A_{i,j}^k}$$
3. Take a weighted linear combination of forward feature activation maps and pass through a Rectified Linear Unit (ReLU) to isolate positive contributions:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$

#### Fast Gradient Sign Method (FGSM Attack; Goodfellow et al., 2015):
FGSM computes a single-step adversarial perturbation restricted to an $L_\infty$ ball of radius $\epsilon$:
$$x_{\text{adv}} = \text{clip}_{[0, 1]} \left( x + \epsilon \cdot \text{sign}\left( \nabla_x \mathcal{L}(\theta, x, y) \right) \right)$$
Perturbation magnitude $\epsilon = 0.02$ adds subtle noise ($\pm 5.1$ intensity levels on an 8-bit scale), remaining barely perceptible to humans while severely disrupting deep classifiers.

#### Adversarial Training Defense (Madry et al., 2018):
To harden models against evasion attacks, we train `cnn_robust` by framing learning as an empirical min-max optimization problem:
$$\min_\theta \mathbb{E}_{(x, y) \sim \mathcal{D}} \left[ \frac{1}{2} \mathcal{L}_{\text{CE}}(f_\theta(x), y) + \frac{1}{2} \mathcal{L}_{\text{CE}}\left(f_\theta(x_{\text{adv}}), y\right) \right]$$
where $x_{\text{adv}} = x + \epsilon \cdot \text{sign}\left(\nabla_x \mathcal{L}(\theta, x, y)\right)$ with $\epsilon = 0.02$. Joint optimization ensures that decision boundaries are smoothed along directions of steepest loss ascent without degrading clean data accuracy.

---

### 5.4 Module 4: Web Application & System Integration (M4)

The application layer encapsulates the inference pipeline into an interactive Gradio web interface structured across five operational tabs:
1. **Recognise Tab:** Single-image classification with model selector, top-5 probability horizontal bar chart, and real-time Grad-CAM heatmap visualization.
2. **Compare Models Tab:** Parallel side-by-side inference across all four models, including an Ensemble Consensus Vote computed via mean softmax probability aggregation:
   $$P_{\text{ensemble}}(c \mid x) = \frac{1}{M} \sum_{m=1}^M P_m(c \mid x)$$
3. **Fool the CNN Tab:** Interactive adversarial perturbation laboratory. Users manipulate an $\epsilon$ slider ($0.00 \le \epsilon \le 0.10$), inspect the enlarged residual perturbation $\text{sign}(\nabla_x \mathcal{L})$, and directly contrast clean CNN predictions against the defended `cnn_robust`.
4. **Model Report Tab:** Static analytical dashboard rendering training curves, confusion matrices, misclassification galleries, and latency-vs-accuracy trade-off scatter plots.
5. **About Tab:** Comprehensive project background, architecture documentation, team responsibilities, and system constraints.

---

## 6. Implementation & Training Details

### 6.1 Hardware and Software Environment
- **Development & Training Hardware:** Google Colab Cloud GPU (NVIDIA Tesla T4, 16 GB GDDR6 VRAM, Intel Xeon 2.20 GHz host, 13 GB RAM).
- **Edge Deployment Benchmark Platform:** Local Host CPU (Intel Iris Xe / Core i5, Windows 11 64-bit, 16 GB DDR4 RAM).
- **Core Frameworks:** PyTorch 2.1+, Torchvision 0.16+, Gradio 6.30+, ONNX 1.15+, ONNX Runtime 1.23+, Scikit-Learn 1.3+, Matplotlib 3.8+.

### 6.2 Hyperparameters & Training Recipes
All models were optimized using AdamW with weight decay and a OneCycle learning rate schedule:

| Hyperparameter / Configuration | Custom CNN (M1) | ResNet-18 (M2) | MobileNetV3-Small (M2) | Robust CNN (M3) |
|---|---|---|---|---|
| **Base Architecture** | 4-Stage VGG Custom | ResNet-18 (ImageNet) | MobileNetV3-Small (ImageNet) | 4-Stage VGG Custom |
| **Pretrained Weights** | None (Scratch) | Pretrained ImageNet-1k | Pretrained ImageNet-1k | None (Scratch) |
| **Epochs Trained** | 15 | 8 | 10 | 15 |
| **Batch Size** | 64 | 64 | 64 | 64 |
| **Optimizer** | AdamW | AdamW | AdamW | AdamW |
| **Peak Learning Rate ($\eta_{\max}$)** | $1.0 \times 10^{-3}$ | $5.0 \times 10^{-4}$ | $1.0 \times 10^{-3}$ | $1.0 \times 10^{-3}$ |
| **LR Scheduler** | OneCycleLR | OneCycleLR | OneCycleLR | OneCycleLR |
| **Weight Decay ($\lambda$)** | $1.0 \times 10^{-4}$ | $1.0 \times 10^{-4}$ | $1.0 \times 10^{-4}$ | $1.0 \times 10^{-4}$ |
| **Label Smoothing ($\alpha$)** | 0.1 | 0.1 | 0.1 | 0.1 |
| **Gradient Clipping ($\|\mathbf{g}\|_2$)** | 1.0 | 1.0 | 1.0 | 1.0 |
| **Adversarial Loss Weight** | 0.0 | 0.0 | 0.0 | 0.5 clean / 0.5 adv |
| **Adversarial Training $\epsilon$** | — | — | — | 0.02 |

---

## 7. Experimental Results & Discussion

### 7.1 Quantitative Benchmark on Held-Out Test Set (12,630 Images)
All models were evaluated on the official, held-out GTSRB test set:

| Model Architecture | Parameters (M) | FLOPs (MFLOPs) | Test Accuracy (%) | Macro F1-Score | FGSM Accuracy ($\epsilon = 0.02$) | GPU Latency (batch=1) |
|---|---|---|---|---|---|---|
| **Custom CNN (scratch)** | **0.824** | 310.1 | **99.59%** | **0.9960** | 82.89% | **0.999 ms** |
| **ResNet-18 (transfer)** | 11.199 | 296.1 | 98.99% | 0.9831 | 94.61% | 2.289 ms |
| **MobileNetV3-Small (edge)** | 1.562 | **11.1** | 95.05% | 0.9302 | 64.06% | 5.156 ms |
| **Robust CNN (adv. train)** | **0.824** | 310.1 | **99.69%** | **0.9964** | **98.71%** | 1.037 ms |

```
                       TEST ACCURACY VS. FGSM ROBUSTNESS
  100% +-----------------------------------------------------------------+
       |                                                    * Robust CNN |
   95% |                   # ResNet-18                                   |
       |                                                                 |
   90% |                                                                 |
       |                                                                 |
   85% |   o Custom CNN                                                  |
       |                                                                 |
   80% |                                                                 |
       |                                                                 |
   65% |   x MobileNetV3-Small                                           |
       +-----------------------------------------------------------------+
          Clean Test Accuracy: [95.05% - 99.69%]
          FGSM (eps=0.02) Robustness:
          - MobileNetV3-Small: 64.06%
          - Custom CNN:        82.89%
          - ResNet-18:         94.61%
          - Robust CNN:        98.71%  <-- (Highest overall defense)
```

### 7.2 ONNX CPU Edge Benchmark
Models were exported to ONNX format with graph optimization and evaluated over 100 iterations on CPU (batch size = 1):

| Model Name | ONNX Size (MB) | PyTorch CPU Latency | ONNX Runtime Latency | Speed-Up Factor | Max Absolute Diff | Test Accuracy |
|---|---|---|---|---|---|---|
| **Custom CNN** | 3.30 MB | 7.00 ms | 2.93 ms | 2.39x | $1.19 \times 10^{-6}$ | 99.59% |
| **Robust CNN** | 3.30 MB | 7.17 ms | 2.79 ms | 2.57x | $1.31 \times 10^{-6}$ | 99.69% |
| **MobileNetV3-Small** | 6.26 MB | 5.87 ms | **0.74 ms** | **7.94x** | $3.10 \times 10^{-6}$ | 95.05% |
| **ResNet-18** | 44.79 MB | 13.62 ms | 5.21 ms | 2.62x | $9.54 \times 10^{-7}$ | 98.99% |

### 7.3 Core Technical Findings & Analytical Discussions

#### 1. Why Custom CNN Outperforms ResNet-18 at $64 \times 64$:
A key result is that our 0.82M-parameter Custom CNN achieves higher accuracy (**99.59%**) than the 11.20M-parameter ResNet-18 (**98.99%**). This is explained by spatial downsampling ratios. ResNet-18 was designed for $224 \times 224$ ImageNet inputs and applies five successive stride-2 downsampling operations ($32\times$ reduction):
$$64 \to 32 \to 16 \to 8 \to 4 \to 2$$
At the final residual block, the spatial feature map shrinks to a minimal $2 \times 2$ grid before Global Average Pooling. Traffic signs require fine interior pictogram resolution (such as differentiating the curved numerals in *Speed limit 30* vs. *Speed limit 80*). Aggressive downsampling discards this high-frequency geometric detail. Conversely, the Custom CNN downsamples only $8\times$ ($64 \to 8 \times 8$), preserving essential structural information.

#### 2. The FLOPs vs. Latency Paradox:
MobileNetV3-Small exhibits **$28\times$ fewer FLOPs** than the Custom CNN (11.1 MFLOPs vs. 310.1 MFLOPs), yet on GPU at batch size 1, it runs slower (5.16 ms vs. 1.00 ms). This disparity reflects arithmetic intensity (FLOPs per byte of memory bandwidth):
- Depthwise separable convolutions perform minimal arithmetic per weight load.
- On modern parallel GPUs, memory access latency dominates compute time when batch size equals 1.
- However, on CPU execution via ONNX Runtime, MobileNetV3 achieves **0.74 ms** ($1,350+$ FPS) with an ONNX footprint of only 6.26 MB, demonstrating clear advantages for low-power edge CPUs.

#### 3. Efficacy of Adversarial Training:
Evaluating standard models under FGSM noise ($\epsilon = 0.02$) reveals vulnerability:
- Custom CNN accuracy falls from **99.59% to 82.89%**.
- MobileNetV3 collapses from **95.05% to 64.06%**.
- In contrast, `cnn_robust` sustains **98.71% accuracy under attack**, while improving clean accuracy to **99.69%**.
- Adversarial training acts as a geometry-aware regularizer, smoothing decision boundaries around the data manifold without compromising nominal classification.

#### 4. Hardest Classes and Misclassification Patterns:
Analysis of confusion matrices reveals that remaining errors concentrate in visually similar sign families:
- *Speed limit 70 km/h* (Class 4) vs. *Speed limit 80 km/h* (Class 5).
- *Pedestrians* (Class 27) vs. *General caution* (Class 18) under severe motion blur or low illumination.
- *End of speed limit 80 km/h* (Class 6) vs. *End of all speed and passing limits* (Class 32) when diagonal cancellation slashes obscure internal symbols.

---

## 8. Interactive Application & Deployment (M4)

The SignSense system is accessible through an interactive Gradio web application providing five focused views:

| View Tab | Primary Functionality | Interactive Inputs | Display Outputs |
|---|---|---|---|
| **1. Recognise** | Single-image inspection & Grad-CAM | Image upload or sample selection, model choice | Top-5 confidence bar chart, predicted class, Grad-CAM attention overlay |
| **2. Compare Models** | Multi-model consensus & ensemble | Image input | 4 parallel prediction panels, Ensemble Consensus Vote, per-model latency |
| **3. Fool the CNN** | Adversarial attack exploration | Clean image, $\epsilon$ noise slider ($0.00 - 0.10$) | Clean vs. perturbed image, magnified noise map, Standard vs. Robust CNN side-by-side |
| **4. Model Report** | Offline metric reporting | Plot category selector | High-resolution training curves, confusion matrices, misclassification galleries |
| **5. About** | Architecture & documentation | — | Project overview, team module breakdown, mathematical formulations |

---

## 9. Limitations & Future Scope

### 9.1 Current Limitations
1. **Cropped Image Assumption:** The pipeline assumes signs have already been detected and tightly cropped. Passing unconstrained, full-frame dashcam images causes domain shift, lowering classification confidence.
2. **Geographical Scope:** Training data is restricted to the German Traffic Sign Recognition Benchmark. While international conventions share similarities, foreign signage (e.g., US MUTCD octagonal or diamond signs) requires fine-tuning.
3. **Single-Step Adversarial Defense:** While `cnn_robust` provides defense against FGSM ($\epsilon = 0.02$), iterative attacks such as Projected Gradient Descent (PGD) or Carlini-Wagner ($L_2$) may degrade performance without multi-step robust optimization.

### 9.2 Future Directions
1. **End-to-End Perception Pipeline:** Integrate an upstream real-time object detector (such as YOLOv8 or RT-DETR) to localize sign bounding boxes directly from raw $1920 \times 1080$ dashcam video before classification.
2. **Iterative PGD Adversarial Hardening:** Extend adversarial training to multi-step PGD with random restarts and randomized smoothing to establish certified robustness guarantees.
3. **INT8 Quantization & TensorRT Deployment:** Quantize ONNX models to INT8 using Post-Training Quantization (PTQ) with calibration datasets, targeting sub-0.3 ms latencies on automotive hardware such as NVIDIA Jetson Orin.

---

## 10. Team Contribution Breakdown

| Team Member | Module Title | Specific Contributions & Technical Deliverables |
|---|---|---|
| **Member 1** | **Data Engineering & Custom CNN** | • Downloaded and sorted 51,839 GTSRB images using Kaggle API fallback.<br>• Authored exploratory data analysis scripts generating 4 analytical plots.<br>• Designed and trained 4-stage VGG-style CNN with Batch Normalization, Dropout, and He initialization.<br>• Implemented OneCycleLR schedule and directional-preserving augmentations. |
| **Member 2** | **Transfer Learning & Edge Deployment** | • Fine-tuned ResNet-18 residual architecture with ImageNet weights.<br>• Adapted MobileNetV3-Small edge architecture utilizing depthwise separable convolutions.<br>• Benchmarked parameter counts, FLOPs, and GPU/CPU latencies.<br>• Exported all models to ONNX graph format and executed ONNX Runtime CPU benchmarks. |
| **Member 3** | **Explainability & Adversarial Defense** | • Implemented Grad-CAM activation heatmap generation for convolutional feature layers.<br>• Formulated Fast Gradient Sign Method (FGSM) adversarial attack engine.<br>• Implemented joint min-max adversarial training, creating the hardened `cnn_robust` model.<br>• Generated empirical robustness curve evaluations across varying noise budgets $\epsilon \in [0, 0.08]$. |
| **Member 4** | **System Integration & Web UI** | • Developed 5-tab Gradio web interface (`app.py`) with ensemble consensus voting.<br>• Formatted and executed end-to-end executable Colab story notebook (`SignSense_Colab.ipynb`).<br>• Established GitHub repository structure, release packaging, and documentation.<br>• Authored presentation outline, demonstration script, and viva examination guide. |

---

## 11. References

1. **Stallkamp, J., Schlipsing, M., Salmen, J., & Igel, C. (2012).** Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition. *Neural Networks*, 32, 323–332.
2. **He, K., Zhang, X., Ren, S., & Sun, J. (2016).** Deep residual learning for image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 770–778.
3. **Howard, A., Sandler, M., Chu, G., Chen, L. C., Chen, B., Tan, M., Wang, W., Zhu, Y., Pang, R., Vasudevan, V., Le, Q. V., & Adam, H. (2019).** Searching for MobileNetV3. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 1314–1324.
4. **Selvaraju, R. R., Cogswell, M., Das, A., Vedaldi, A., Parikh, D., & Batra, D. (2017).** Grad-CAM: Visual explanations from deep networks via gradient-based localization. In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, pp. 618–626.
5. **Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015).** Explaining and harnessing adversarial examples. In *International Conference on Learning Representations (ICLR)*.
6. **Ioffe, S., & Szegedy, C. (2015).** Batch Normalization: Accelerating deep network training by reducing internal covariate shift. In *International Conference on Machine Learning (ICML)*, pp. 448–456.
7. **Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018).** Towards deep learning models resistant to adversarial attacks. In *International Conference on Learning Representations (ICLR)*.
