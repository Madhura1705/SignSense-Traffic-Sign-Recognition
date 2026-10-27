export function createSignBadge({ type = 'circle', label = '' }) {
  const badge = document.createElement('span');
  badge.className = `sign-badge ${type}`;
  badge.textContent = label;
  badge.setAttribute('aria-hidden', 'true');
  return badge;
}
