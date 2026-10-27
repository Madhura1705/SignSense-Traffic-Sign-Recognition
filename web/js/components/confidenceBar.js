export function createConfidenceBar(prob) {
  const wrap = document.createElement('div');
  wrap.className = 'confidence-bar-wrap';
  wrap.setAttribute('role', 'progressbar');
  wrap.setAttribute('aria-valuenow', Math.round(prob * 100));
  wrap.setAttribute('aria-valuemin', '0');
  wrap.setAttribute('aria-valuemax', '100');

  const fill = document.createElement('div');
  fill.className = 'confidence-bar-fill';
  const pct = Math.min(Math.max(prob * 100, 0), 100);
  fill.style.width = `${pct.toFixed(1)}%`;

  if (prob >= 0.90) fill.classList.add('high');
  else if (prob >= 0.60) fill.classList.add('medium');
  else fill.classList.add('low');

  wrap.appendChild(fill);
  return wrap;
}
