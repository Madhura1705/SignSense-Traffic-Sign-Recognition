import { createConfidenceBar } from './confidenceBar.js';

export function createTop5List(top5Array = []) {
  const container = document.createElement('div');
  container.className = 'top5-list';

  top5Array.forEach((item, idx) => {
    const row = document.createElement('div');
    row.className = 'top5-item';

    const meta = document.createElement('div');
    meta.className = 'top5-meta';

    const name = document.createElement('span');
    name.className = 'top5-name';
    name.textContent = `${idx + 1}. ${item.class_name}`;

    const prob = document.createElement('span');
    prob.className = 'top5-prob';
    prob.textContent = `${(item.prob * 100).toFixed(1)}%`;

    meta.appendChild(name);
    meta.appendChild(prob);

    const bar = createConfidenceBar(item.prob);

    row.appendChild(meta);
    row.appendChild(bar);
    container.appendChild(row);
  });

  return container;
}
