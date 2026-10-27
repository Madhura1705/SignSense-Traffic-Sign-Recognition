export function createPanel({ title, content, className = '' }) {
  const panel = document.createElement('div');
  panel.className = `panel ${className}`.trim();

  if (title) {
    const header = document.createElement('div');
    header.className = 'panel-header';
    const h3 = document.createElement('h3');
    h3.className = 'panel-title';
    h3.textContent = title;
    header.appendChild(h3);
    panel.appendChild(header);
  }

  if (typeof content === 'string') {
    const body = document.createElement('div');
    body.innerHTML = content;
    panel.appendChild(body);
  } else if (content instanceof HTMLElement) {
    panel.appendChild(content);
  }

  return panel;
}
