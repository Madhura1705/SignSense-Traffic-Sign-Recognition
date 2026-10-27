import { createPanel } from '../components/index.js';

export function render(container) {
  container.innerHTML = `
    <div class="page-header">
      <h2 class="page-title">Attack Lab</h2>
      <p class="page-description">Fast Gradient Sign Method (FGSM) adversarial testing and robust min-max training defense.</p>
    </div>
  `;

  const panel = createPanel({
    title: 'Adversarial Robustness',
    content: `
      <p style="color: var(--muted); margin-bottom: 12px;">
        Interactive epsilon slider, amplified noise visualization, and cross-model transfer attacks (Phase D implementation).
      </p>
    `,
  });

  container.appendChild(panel);
}

export function destroy() {
  // Teardown
}
