export function createSegmentedControl({
  name,
  options = [], // [{ value, label }]
  value,
  onChange,
  ariaLabel = 'Select option',
}) {
  const container = document.createElement('div');
  container.className = 'segmented-control';
  container.setAttribute('role', 'radiogroup');
  container.setAttribute('aria-label', ariaLabel);

  let currentValue = value !== undefined ? value : options[0]?.value;
  const buttons = [];

  function updateActive(newVal, emit = true) {
    currentValue = newVal;
    buttons.forEach((btn) => {
      const isChecked = btn.dataset.value === String(currentValue);
      btn.setAttribute('aria-checked', isChecked ? 'true' : 'false');
      btn.setAttribute('tabindex', isChecked ? '0' : '-1');
      btn.classList.toggle('active', isChecked);
    });
    if (emit && onChange) onChange(currentValue);
  }

  options.forEach((opt, idx) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'segment-item';
    btn.setAttribute('role', 'radio');
    btn.dataset.value = String(opt.value);
    btn.textContent = opt.label;

    const isChecked = opt.value === currentValue;
    btn.setAttribute('aria-checked', isChecked ? 'true' : 'false');
    btn.setAttribute('tabindex', isChecked ? '0' : '-1');
    if (isChecked) btn.classList.add('active');

    btn.addEventListener('click', () => {
      updateActive(opt.value);
    });

    // Arrow-key keyboard navigation (Accessibility)
    btn.addEventListener('keydown', (e) => {
      let targetIdx = null;
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        targetIdx = (idx + 1) % options.length;
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        targetIdx = (idx - 1 + options.length) % options.length;
      }
      if (targetIdx !== null) {
        e.preventDefault();
        buttons[targetIdx].focus();
        updateActive(options[targetIdx].value);
      }
    });

    buttons.push(btn);
    container.appendChild(btn);
  });

  container.getValue = () => currentValue;
  container.setValue = (newVal) => updateActive(newVal, false);

  return container;
}
