import { createPanel } from '../components/index.js';

export function render(container) {
  container.innerHTML = `
    <div class="page-header">
      <h2 class="page-title">Compare Models</h2>
      <p class="page-description">Run all four neural network architectures in parallel and evaluate model consensus via Ensemble Voting.</p>
    </div>
  `;

  const panel = createPanel({
    title: 'Multi-Model Consensus',
    content: `
      <p style="color: var(--muted); margin-bottom: 12px;">
        Compare Custom CNN, ResNet-18, MobileNetV3-Small, and Robust CNN side by side (Phase D implementation).
      </p>
    `,
  });

  container.appendChild(panel);
}

export function destroy() {
  // Teardown
}
