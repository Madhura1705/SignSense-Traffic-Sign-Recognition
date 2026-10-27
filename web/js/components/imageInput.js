export function createImageInput({
  onImageSelected,
  samples = [],
  initialPreviewUrl = null,
}) {
  const container = document.createElement('div');
  container.className = 'image-input-box';

  const hiddenInput = document.createElement('input');
  hiddenInput.type = 'file';
  hiddenInput.accept = 'image/png,image/jpeg,image/webp';
  hiddenInput.style.display = 'none';

  const dropzone = document.createElement('div');
  dropzone.className = 'dropzone';
  dropzone.setAttribute('tabindex', '0');
  dropzone.setAttribute('role', 'button');
  dropzone.setAttribute('aria-label', 'Upload traffic sign image');

  const preview = document.createElement('img');
  preview.className = 'dropzone-preview';
  preview.alt = 'Uploaded sign preview';
  preview.style.display = initialPreviewUrl ? 'block' : 'none';
  if (initialPreviewUrl) preview.src = initialPreviewUrl;

  const prompt = document.createElement('div');
  prompt.className = 'dropzone-prompt';
  prompt.style.display = initialPreviewUrl ? 'none' : 'flex';
  prompt.innerHTML = `
    <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
      <polyline points="17 8 12 3 7 8"></polyline>
      <line x1="12" y1="3" x2="12" y2="15"></line>
    </svg>
    <div style="font-weight: 600; font-size: 15px; color: var(--ink);">Drop a traffic sign image here</div>
    <div style="font-size: 13px;">or click to browse from device (paste supported)</div>
  `;

  dropzone.appendChild(preview);
  dropzone.appendChild(prompt);
  dropzone.appendChild(hiddenInput);

  function handleFile(file, sampleMeta = null) {
    if (!file || !file.type.startsWith('image/')) return;
    const url = URL.createObjectURL(file);
    preview.src = url;
    preview.style.display = 'block';
    prompt.style.display = 'none';
    if (onImageSelected) onImageSelected(file, sampleMeta);
  }

  // Click & File Selection
  dropzone.addEventListener('click', () => hiddenInput.click());
  hiddenInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  });

  // Drag and Drop
  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  // Paste from Clipboard
  window.addEventListener('paste', (e) => {
    if (e.clipboardData && e.clipboardData.items) {
      for (let i = 0; i < e.clipboardData.items.length; i++) {
        const item = e.clipboardData.items[i];
        if (item.type.indexOf('image') !== -1) {
          const blob = item.getAsFile();
          handleFile(blob);
          break;
        }
      }
    }
  });

  // Keyboard accessibility
  dropzone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      hiddenInput.click();
    }
  });

  container.appendChild(dropzone);

  // Sample picker strip
  if (samples && samples.length > 0) {
    const stripBox = document.createElement('div');
    stripBox.className = 'sample-strip-container';

    const stripTitle = document.createElement('div');
    stripTitle.className = 'sample-strip-title';
    stripTitle.textContent = 'Or try these sample test signs:';
    stripBox.appendChild(stripTitle);

    const strip = document.createElement('div');
    strip.className = 'sample-strip';

    samples.forEach((sample) => {
      const img = document.createElement('img');
      img.className = 'sample-thumb';
      img.src = sample.url;
      img.alt = sample.class_name || sample.file;
      img.title = sample.class_name ? `${sample.class_name} (Class ${sample.class_id})` : sample.file;
      img.setAttribute('tabindex', '0');
      img.setAttribute('role', 'button');

      const loadSample = async () => {
        try {
          const res = await fetch(sample.url);
          const blob = await res.blob();
          const file = new File([blob], sample.file, { type: 'image/png' });
          handleFile(file, sample);
        } catch (err) {
          console.error('Failed to load sample:', err);
        }
      };

      img.addEventListener('click', loadSample);
      img.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          loadSample();
        }
      });

      strip.appendChild(img);
    });

    stripBox.appendChild(strip);
    container.appendChild(stripBox);
  }

  container.setPreview = (url) => {
    preview.src = url;
    preview.style.display = 'block';
    prompt.style.display = 'none';
  };

  return container;
}
