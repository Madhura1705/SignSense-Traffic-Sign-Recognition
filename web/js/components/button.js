export function createButton({
  label,
  variant = 'primary', // 'primary' | 'secondary' | 'danger'
  onClick,
  disabled = false,
  id,
  type = 'button',
  ariaLabel,
}) {
  const btn = document.createElement('button');
  btn.type = type;
  btn.className = `btn btn-${variant}`;
  btn.textContent = label;
  if (id) btn.id = id;
  if (ariaLabel) btn.setAttribute('aria-label', ariaLabel);
  btn.disabled = disabled;
  if (onClick) {
    btn.addEventListener('click', onClick);
  }
  return btn;
}
