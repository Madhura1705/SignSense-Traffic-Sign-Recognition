let activeLightbox = null;
let previousFocusedElement = null;

export function openLightbox({ src, alt = 'Enlarged image', caption = '' }) {
  if (activeLightbox) closeLightbox();

  previousFocusedElement = document.activeElement;

  const backdrop = document.createElement('div');
  backdrop.className = 'lightbox-backdrop';
  backdrop.setAttribute('role', 'dialog');
  backdrop.setAttribute('aria-modal', 'true');
  backdrop.setAttribute('aria-label', caption || alt);

  const content = document.createElement('div');
  content.className = 'lightbox-content';

  const closeBtn = document.createElement('button');
  closeBtn.className = 'lightbox-close';
  closeBtn.innerHTML = '&times;';
  closeBtn.setAttribute('aria-label', 'Close image viewer');

  const img = document.createElement('img');
  img.className = 'lightbox-img';
  img.src = src;
  img.alt = alt;

  content.appendChild(closeBtn);
  content.appendChild(img);

  if (caption) {
    const cap = document.createElement('div');
    cap.style.color = '#FFFFFF';
    cap.style.marginTop = '12px';
    cap.style.fontSize = '14px';
    cap.textContent = caption;
    content.appendChild(cap);
  }

  backdrop.appendChild(content);
  document.body.appendChild(backdrop);
  activeLightbox = backdrop;

  closeBtn.focus();

  function onKeyDown(e) {
    if (e.key === 'Escape') {
      closeLightbox();
    } else if (e.key === 'Tab') {
      // Focus trap
      e.preventDefault();
      closeBtn.focus();
    }
  }

  function onClick(e) {
    if (e.target === backdrop || e.target === closeBtn) {
      closeLightbox();
    }
  }

  backdrop.addEventListener('click', onClick);
  document.addEventListener('keydown', onKeyDown);
  backdrop._cleanup = () => {
    document.removeEventListener('keydown', onKeyDown);
    backdrop.removeEventListener('click', onClick);
  };
}

export function closeLightbox() {
  if (activeLightbox) {
    if (activeLightbox._cleanup) activeLightbox._cleanup();
    activeLightbox.remove();
    activeLightbox = null;
    if (previousFocusedElement && typeof previousFocusedElement.focus === 'function') {
      previousFocusedElement.focus();
    }
  }
}
