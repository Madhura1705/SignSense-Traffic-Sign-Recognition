import { createPanel } from '../components/index.js';

export function render(container) {
  container.innerHTML = `
    <div class="page-header">
      <h2 class="page-title">Recognise</h2>
      <p class="page-description">Inspect individual traffic sign predictions with top-5 confidence distributions and real-time Grad-CAM heatmaps.</p>
    </div>
  `;

  const panel = createPanel({
    title: 'Traffic Sign Diagnosis',
    content: `
      <p style="color: var(--muted); margin-bottom: 12px;">
        Single-image diagnosis and live camera perception (Phase C implementation).
      </p>
    `,
  });

  container.appendChild(panel);
}

export function destroy() {
  // Teardown camera / timers
}
