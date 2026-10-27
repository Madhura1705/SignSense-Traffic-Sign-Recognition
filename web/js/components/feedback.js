export function createSpinner() {
  const spinner = document.createElement('div');
  spinner.className = 'spinner';
  spinner.setAttribute('role', 'status');
  spinner.setAttribute('aria-label', 'Loading');
  return spinner;
}

export function createSkeleton({ width = '100%', height = '20px', radius = '6px' } = {}) {
  const el = document.createElement('div');
  el.style.width = width;
  el.style.height = height;
  el.style.borderRadius = radius;
  el.style.backgroundColor = 'var(--border)';
  el.style.opacity = '0.6';
  el.setAttribute('aria-hidden', 'true');
  return el;
}

export function createErrorBanner(message) {
  const banner = document.createElement('div');
  banner.className = 'error-banner';
  banner.setAttribute('role', 'alert');
  banner.textContent = message;
  return banner;
}

export function createEmptyState({
  title = 'No data available',
  subtitle = 'Upload an image or select a sample above.',
  icon = '🔍',
}) {
  const el = document.createElement('div');
  el.className = 'empty-state';

  const iconEl = document.createElement('div');
  iconEl.style.fontSize = '36px';
  iconEl.textContent = icon;

  const titleEl = document.createElement('div');
  titleEl.style.fontWeight = '600';
  titleEl.style.fontSize = '16px';
  titleEl.style.color = 'var(--ink)';
  titleEl.textContent = title;

  const subtitleEl = document.createElement('div');
  subtitleEl.style.fontSize = '14px';
  subtitleEl.textContent = subtitle;

  el.appendChild(iconEl);
  el.appendChild(titleEl);
  el.appendChild(subtitleEl);
  return el;
}
