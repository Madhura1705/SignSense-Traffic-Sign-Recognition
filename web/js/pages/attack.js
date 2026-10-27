import { api } from '../api.js';
import {
  createPanel,
  createButton,
  createSegmentedControl,
  createImageInput,
  createSpinner,
  createErrorBanner,
  createEmptyState,
  openLightbox,
} from '../components/index.js';

let currentFile = null;
let currentSampleMeta = null;
let currentModelKey = 'cnn';
let currentEps = 0.02;
let debounceTimer = null;
let abortController = null;
let sweepChart = null;
let isSweeping = false;

export async function render(container) {
  container.innerHTML = `
    <div class="page-header">
      <h1 class="page-title" style="color: var(--stop-red);">Attack Lab: Adversarial Robustness</h1>
      <p class="page-description">
        Fast Gradient Sign Method (FGSM) crafts imperceptible adversarial noise by computing the sign of the loss gradient with respect to the input pixels.
      </p>
      <div class="formula-box">
        x' = x + &epsilon; &middot; sign(&nabla;<sub>x</sub> J(&theta;, x, y))
      </div>
    </div>

    <!-- Presets from docs/DEMO_EXAMPLES.md -->
    <div class="panel" style="margin-bottom: 24px;">
      <div style="font-weight: 600; font-size: 14px; margin-bottom: 6px;">Demonstration presets (Attack vs Robust Defense):</div>
      <div class="presets-strip" id="presets-strip">
        <button class="preset-btn" data-file="2_speed_50.png" data-eps="0.02" data-model="cnn">
          Speed 50 at &epsilon; 0.02
        </button>
        <button class="preset-btn" data-file="28_children_crossing.png" data-eps="0.03" data-model="cnn">
          Children crossing at &epsilon; 0.03
        </button>
        <button class="preset-btn" data-file="40_roundabout.png" data-eps="0.04" data-model="cnn">
          Roundabout at &epsilon; 0.04
        </button>
        <button class="preset-btn" data-file="1_speed_30.png" data-eps="0.04" data-model="cnn">
          Speed 30 at &epsilon; 0.04
        </button>
      </div>
    </div>

    <div class="recognise-layout" style="align-items: flex-start;">
      <!-- Left Controls: ImageInput & Epsilon Slider -->
      <div style="display: flex; flex-direction: column; gap: 20px;">
        <div class="panel">
          <div style="display: flex; flex-direction: column; gap: 14px;">
            <div>
              <div style="font-weight: 600; font-size: 14px; margin-bottom: 6px;">Target model to attack:</div>
              <div id="attack-model-selector-container"></div>
            </div>

            <div>
              <div style="display: flex; justify-content: space-between; font-size: 14px; font-weight: 600; margin-bottom: 6px;">
                <span>Perturbation noise strength (&epsilon;):</span>
                <span id="eps-display-val" style="color: var(--stop-red);">0.020</span>
              </div>
              <input type="range" id="eps-slider" class="slider-input" min="0" max="0.10" step="0.005" value="0.02" style="width: 100%; accent-color: var(--stop-red);">
              <div style="display: flex; justify-content: space-between; font-size: 11px; color: var(--muted); margin-top: 4px;">
                <span>0.00 (Clean)</span>
                <span>0.05 (Subtle)</span>
                <span>0.10 (Severe)</span>
              </div>
            </div>
          </div>

          <div style="margin-top: 20px;" id="attack-image-input-container"></div>
        </div>

        <!-- Why this matters note -->
        <div class="panel" style="border-left: 4px solid var(--lane);">
          <h4 style="font-size: 15px; margin-bottom: 6px;">Why this matters for autonomous driving</h4>
          <p style="font-size: 13px; color: var(--muted); line-height: 1.45;">
            In physical tests, researchers proved that subtle black-and-white stickers placed on a Stop sign caused vision models to predict "Speed limit 80".
            SignSense implemented joint min-max adversarial training (<strong>cnn_robust</strong>) to defend against these evasion attacks.
          </p>
        </div>
      </div>

      <!-- Right Column: Triplet Images, Verdict & Transfer Table -->
      <div id="attack-results-panel">
        <!-- Rendered dynamically -->
      </div>
    </div>

    <!-- Bottom Section: Robustness Sweep Chart -->
    <div class="panel" style="margin-top: 24px;" id="sweep-section">
      <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; margin-bottom: 14px;">
        <div>
          <h3 class="panel-title">Multi-model robustness sweep</h3>
          <p style="font-size: 13.5px; color: var(--muted);">
            Evaluates clean class retention across noise levels &epsilon; &in; [0, 0.10] for all four architectures.
          </p>
        </div>
        <button id="run-sweep-btn" class="btn btn-primary" style="background-color: var(--stop-red);">
          Run robustness sweep
        </button>
      </div>

      <div id="sweep-chart-container" style="display: none;">
        <div style="height: 280px; position: relative;">
          <canvas id="sweep-chart-canvas"></canvas>
        </div>
        <div id="sweep-takeaway" style="margin-top: 14px; padding: 12px 16px; background-color: var(--paper); border-radius: var(--radius-sm); font-size: 13.5px; font-weight: 500;"></div>
      </div>
    </div>
  `;

  const modelSelectorContainer = document.getElementById('attack-model-selector-container');
  const imageInputContainer = document.getElementById('attack-image-input-container');
  const resultsPanel = document.getElementById('attack-results-panel');
  const epsSlider = document.getElementById('eps-slider');
  const epsDisplay = document.getElementById('eps-display-val');
  const runSweepBtn = document.getElementById('run-sweep-btn');

  // Slider debounced handler
  epsSlider.addEventListener('input', (e) => {
    currentEps = parseFloat(e.target.value);
    epsDisplay.textContent = currentEps.toFixed(3);
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      if (currentFile) runAttack();
    }, 250);
  });

  // Setup Model Selector
  try {
    const models = await api.getModels();
    const options = models.map((m) => ({ value: m.key, label: m.name }));
    const selector = createSegmentedControl({
      name: 'attack-model-picker',
      options,
      value: currentModelKey,
      onChange: (val) => {
        currentModelKey = val;
        if (currentFile) runAttack();
      },
      ariaLabel: 'Target model for FGSM attack',
    });
    modelSelectorContainer.appendChild(selector);
  } catch (err) {
    console.error('Failed to load models:', err);
  }

  // Load samples
  let samples = [];
  try {
    const allSamples = await api.getSamples();
    samples = allSamples.filter((s) => !s.real);
  } catch (err) {
    console.error('Failed to load samples:', err);
  }

  // Setup ImageInput
  const imageInput = createImageInput({
    onImageSelected: (file, sampleMeta) => {
      currentFile = file;
      currentSampleMeta = sampleMeta;
      runAttack();
    },
    samples,
  });
  imageInputContainer.appendChild(imageInput);

  // Bind Preset Buttons
  const presetBtns = container.querySelectorAll('.preset-btn');
  presetBtns.forEach((btn) => {
    btn.addEventListener('click', async () => {
      const fileName = btn.dataset.file;
      const epsVal = parseFloat(btn.dataset.eps);
      const modKey = btn.dataset.model;

      currentEps = epsVal;
      currentModelKey = modKey;
      epsSlider.value = epsVal.toString();
      epsDisplay.textContent = epsVal.toFixed(3);

      try {
        const url = `/samples/${fileName}`;
        const res = await fetch(url);
        const blob = await res.blob();
        const file = new File([blob], fileName, { type: 'image/png' });
        currentFile = file;
        currentSampleMeta = { file: fileName, url };
        imageInput.setPreview(url);
        runAttack();
      } catch (err) {
        console.error('Failed to load preset:', err);
      }
    });
  });

  // Bind Sweep Button
  runSweepBtn.addEventListener('click', runSweep);

  renderEmptyState(resultsPanel);
}

function renderEmptyState(panel) {
  panel.innerHTML = '';
  panel.appendChild(
    createEmptyState({
      title: 'No sign selected for attack test',
      subtitle: 'Pick a demonstration preset above or upload an image to launch an FGSM attack.',
      icon: '⚠️',
    })
  );
}

/**
 * Execute Attack API call
 */
async function runAttack() {
  const panel = document.getElementById('attack-results-panel');
  if (!panel || !currentFile) return;

  if (abortController) abortController.abort();
  abortController = new AbortController();

  panel.innerHTML = '';
  const loading = document.createElement('div');
  loading.className = 'panel';
  loading.style.display = 'flex';
  loading.style.flexDirection = 'column';
  loading.style.alignItems = 'center';
  loading.style.justifyContent = 'center';
  loading.style.padding = '48px 0';
  loading.style.gap = '12px';
  loading.appendChild(createSpinner());
  const loadText = document.createElement('div');
  loadText.style.color = 'var(--muted)';
  loadText.textContent = `Generating FGSM perturbation (ε = ${currentEps})...`;
  loading.appendChild(loadText);
  panel.appendChild(loading);

  try {
    const data = await api.attack(currentFile, currentModelKey, currentEps, abortController.signal);
    renderAttackResults(panel, data);
  } catch (err) {
    if (err.name === 'AbortError') return;
    panel.innerHTML = '';
    panel.appendChild(createErrorBanner(err.message || 'Attack generation failed.'));
  }
}

/**
 * Render Triplet Images, Verdict, and Transfer Attack Table
 */
function renderAttackResults(panel, data) {
  panel.innerHTML = '';

  const { clean, adversarial, fooled, input_b64, adv_b64, noise_b64, transfer, model, eps } = data;

  // 1. Big Verdict Banner
  const verdict = document.createElement('div');
  verdict.className = `verdict-banner ${fooled ? 'fooled' : 'resisted'}`;
  if (fooled) {
    verdict.textContent = `❌ Fooled: ${clean.class_name} became ${adversarial.class_name} (${(adversarial.confidence * 100).toFixed(1)}%)`;
  } else {
    verdict.textContent = `✅ Resisted: Model correctly retained ${clean.class_name} (${(adversarial.confidence * 100).toFixed(1)}%)`;
  }
  panel.appendChild(verdict);

  // 2. Three Images Side-by-Side: Original | Noise (amplified) | Adversarial
  const triplet = document.createElement('div');
  triplet.className = 'attack-triplet';

  // Card 1: Original
  const origCard = document.createElement('div');
  origCard.className = 'attack-img-card';
  origCard.innerHTML = `
    <h4>Original image</h4>
    <img class="attack-img" src="${input_b64 || adv_b64}" id="orig-img" alt="Clean image">
    <div style="font-weight: 600; font-size: 14px; margin-top: 4px;">${clean.class_name}</div>
    <div style="font-size: 12.5px; color: var(--muted);">${(clean.confidence * 100).toFixed(1)}% clean conf.</div>
  `;
  triplet.appendChild(origCard);

  // Card 2: Noise (amplified)
  const noiseCard = document.createElement('div');
  noiseCard.className = 'attack-img-card';
  noiseCard.innerHTML = `
    <h4>Added noise (amplified)</h4>
    <img class="attack-img" src="${noise_b64}" id="noise-img" alt="Amplified adversarial noise">
    <div style="font-weight: 600; font-size: 13.5px; color: var(--stop-red); margin-top: 4px;">&plusmn;${(eps * 255).toFixed(1)}/255 intensity</div>
    <div style="font-size: 12px; color: var(--muted);">Click to zoom noise in Lightbox</div>
  `;
  const noiseImg = noiseCard.querySelector('#noise-img');
  noiseImg.addEventListener('click', () => {
    openLightbox({
      src: noise_b64,
      caption: `FGSM Adversarial Noise Map (Amplified, ε = ${eps})`,
    });
  });
  triplet.appendChild(noiseCard);

  // Card 3: Adversarial
  const advCard = document.createElement('div');
  advCard.className = 'attack-img-card';
  advCard.innerHTML = `
    <h4>Adversarial image</h4>
    <img class="attack-img" src="${adv_b64}" id="adv-img" alt="Adversarial perturbed image">
    <div style="font-weight: 600; font-size: 14px; margin-top: 4px; color: ${fooled ? 'var(--stop-red)' : 'var(--go-green)'};">
      ${adversarial.class_name}
    </div>
    <div style="font-size: 12.5px; color: var(--muted);">${(adversarial.confidence * 100).toFixed(1)}% adv conf.</div>
  `;
  triplet.appendChild(advCard);

  panel.appendChild(triplet);

  // 3. Transfer Attack Table (Same adversarial image given to every other model)
  if (transfer && transfer.length > 0) {
    const transferPanel = document.createElement('div');
    transferPanel.className = 'panel';

    const tTitle = document.createElement('h3');
    tTitle.className = 'panel-title';
    tTitle.style.marginBottom = '6px';
    tTitle.textContent = 'Transfer attack across other architectures';

    const tDesc = document.createElement('p');
    tDesc.style.fontSize = '13.5px';
    tDesc.style.color = 'var(--muted)';
    tDesc.style.marginBottom = '12px';
    tDesc.textContent = 'Evaluates whether the adversarial noise crafted for this model transfers to fool other network architectures:';

    const table = document.createElement('table');
    table.className = 'transfer-table';
    table.innerHTML = `
      <thead>
        <tr>
          <th>Model</th>
          <th>Clean Prediction</th>
          <th>Adversarial Prediction</th>
          <th>Fooled?</th>
        </tr>
      </thead>
      <tbody>
        ${transfer
          .map((t) => {
            const isRobust = t.model === 'cnn_robust';
            return `
              <tr class="${isRobust ? 'highlight-robust' : ''}">
                <td><strong>${t.name}</strong>${isRobust ? ' (Adversarially trained defense)' : ''}</td>
                <td>${t.clean_class}</td>
                <td>${t.adv_class}</td>
                <td>${t.fooled ? '<span style="color: var(--stop-red); font-weight: 600;">❌ Yes</span>' : '<span style="color: var(--go-green); font-weight: 600;">✅ Resisted</span>'}</td>
              </tr>
            `;
          })
          .join('')}
      </tbody>
    `;

    transferPanel.appendChild(tTitle);
    transferPanel.appendChild(tDesc);
    transferPanel.appendChild(table);
    panel.appendChild(transferPanel);
  }
}

/**
 * Execute Multi-Model Robustness Sweep & Render Chart.js
 */
async function runSweep() {
  if (!currentFile || isSweeping) return;
  const sweepContainer = document.getElementById('sweep-chart-container');
  const runBtn = document.getElementById('run-sweep-btn');
  const canvas = document.getElementById('sweep-chart-canvas');
  const takeaway = document.getElementById('sweep-takeaway');

  if (!sweepContainer || !runBtn || !canvas) return;

  isSweeping = true;
  runBtn.disabled = true;
  runBtn.textContent = 'Running sweep across 44 evaluations...';

  try {
    const data = await api.sweep(currentFile);
    sweepContainer.style.display = 'block';

    if (sweepChart) {
      sweepChart.destroy();
      sweepChart = null;
    }

    const { eps, series, breaks_at } = data;

    // Palette & Styles for Lines
    const lineConfig = {
      cnn_robust: { label: 'Robust CNN (Defense)', color: '#1E8E3E', width: 3.5, dash: [] },
      cnn: { label: 'Custom CNN (Scratch)', color: '#C8102E', width: 2, dash: [4, 4] },
      resnet18: { label: 'ResNet-18 (Transfer)', color: '#0A4DA2', width: 2, dash: [] },
      mobilenet: { label: 'MobileNetV3 (Edge)', color: '#D97706', width: 2, dash: [2, 2] },
    };

    const datasets = Object.keys(series).map((k) => {
      const cfg = lineConfig[k] || { label: k, color: '#64748B', width: 2, dash: [] };
      return {
        label: cfg.label,
        data: series[k].map((p) => Math.round(p * 1000) / 10),
        borderColor: cfg.color,
        backgroundColor: cfg.color,
        borderWidth: cfg.width,
        borderDash: cfg.dash,
        pointRadius: 3,
        tension: 0.15,
      };
    });

    if (window.Chart) {
      sweepChart = new window.Chart(canvas, {
        type: 'line',
        data: {
          labels: eps.map((e) => e.toFixed(2)),
          datasets,
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { position: 'top' },
            tooltip: {
              callbacks: {
                title: (items) => `Noise strength ε = ${items[0].label}`,
                label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y}% clean retention`,
              },
            },
          },
          scales: {
            x: {
              title: { display: true, text: 'Perturbation Strength (Epsilon ε)' },
              grid: { color: '#E2E8F0' },
            },
            y: {
              min: 0,
              max: 100,
              title: { display: true, text: 'Probability of Clean Class (%)' },
              grid: { color: '#E2E8F0' },
              ticks: { callback: (v) => `${v}%` },
            },
          },
        },
      });
    }

    // Generate takeaway sentence from data
    if (takeaway) {
      const cnnBreak = breaks_at.cnn !== null ? `ε = ${breaks_at.cnn}` : 'holds';
      const robBreak = breaks_at.cnn_robust !== null ? `ε = ${breaks_at.cnn_robust}` : 'holds past ε = 0.10';
      takeaway.innerHTML = `
        <strong>Takeaway:</strong> Under white-box FGSM attack, the baseline Custom CNN breaks at <strong>${cnnBreak}</strong>,
        while the adversarially trained Robust CNN <strong>${robBreak}</strong>, demonstrating empirical resilience without clean-accuracy degradation.
      `;
    }
  } catch (err) {
    console.error('Sweep failed:', err);
    alert('Robustness sweep failed. Please check server logs.');
  } finally {
    isSweeping = false;
    runBtn.disabled = false;
    runBtn.textContent = 'Run robustness sweep';
  }
}

export function destroy() {
  clearTimeout(debounceTimer);
  if (sweepChart) {
    sweepChart.destroy();
    sweepChart = null;
  }
  if (abortController) {
    abortController.abort();
    abortController = null;
  }
}
