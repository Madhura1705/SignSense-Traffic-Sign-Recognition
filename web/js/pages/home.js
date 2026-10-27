import { createPanel } from '../components/index.js';

export function render(container) {
  container.innerHTML = `
    <div class="page-header">
      <h1 class="page-title">SignSense</h1>
      <p class="page-description">Traffic sign recognition system for autonomous driving with explainability and adversarial robustness.</p>
    </div>
  `;

  const panel = createPanel({
    title: 'Reading traffic signs the way a car has to',
    content: `
      <p style="color: var(--muted); margin-bottom: 16px;">
        Explore our 43-class GTSRB perception stack featuring scratch CNNs, transfer learning (ResNet-18 & MobileNetV3), Grad-CAM saliency, and FGSM adversarial hardening.
      </p>
      <div class="flex-row">
        <a href="#/recognise" class="btn btn-primary">Try it with your image</a>
        <a href="#/report" class="btn btn-secondary">See the results</a>
      </div>
    `,
  });

  container.appendChild(panel);
}

export function destroy() {
  // Teardown logic
}
