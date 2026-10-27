import { api } from '../api.js';
import {
  createPanel,
  createButton,
  createConfidenceBar,
  createSignBadge,
  openLightbox,
} from '../components/index.js';

let tickerInterval = null;
let isPaused = false;
let currentSamples = [];
let sampleIndex = 0;
let abortController = null;

export async function render(container) {
  container.innerHTML = `
    <!-- Hero Live Road -->
    <section class="hero-road" aria-label="SignSense Hero Live Road">
      <div>
        <h1 class="hero-road-title">Reading traffic signs the way a car has to</h1>
        <p class="hero-road-desc">
          SignSense is an autonomous vehicle perception stack trained on 43 German Traffic Sign Benchmark classes.
          Featuring scratch CNNs, transfer learning, real-time Grad-CAM explainability, and adversarial hardening.
        </p>
        <div class="flex-row" style="flex-wrap: wrap; gap: 14px;">
          <a href="#/recognise" class="btn btn-primary" style="background-color: var(--sign-blue); font-size: 15px;">Try it with your image</a>
          <a href="#/report" class="btn btn-secondary" style="font-size: 15px;">See the results</a>
        </div>
      </div>

      <!-- Live Road Sign Ticker -->
      <div class="road-ticker-panel" id="road-ticker" tabindex="0" role="region" aria-label="Live sign recognition demonstration">
        <div class="road-sign-viewer">
          <img id="ticker-sign-img" class="road-sign-img" src="/samples/14_stop.png" alt="Live demonstration sign">
          <img id="ticker-heatmap-img" class="road-sign-heatmap" src="" alt="Grad-CAM attention heatmap">
        </div>
        <div class="road-prediction-name" id="ticker-pred-name">Stop</div>
        <div class="road-ticker-meta" id="ticker-bar-container"></div>
        <div class="road-ticker-controls">
          <label style="display: flex; align-items: center; gap: 6px; cursor: pointer;">
            <input type="checkbox" id="ticker-cam-toggle">
            <span>Show where it looked</span>
          </label>
          <div style="display: flex; align-items: center; gap: 8px;">
            <span id="ticker-latency" style="font-size: 12px; color: #94A3B8;">~80 ms</span>
            <button id="ticker-pause-btn" class="btn btn-secondary" style="padding: 4px 8px; font-size: 12px; height: auto;">
              Pause
            </button>
          </div>
        </div>
      </div>
    </section>

    <!-- Speed-Limit Roundels Results Strip -->
    <section class="results-strip" id="results-strip" aria-label="Key benchmark metrics">
      <div class="roundel-item">
        <div class="roundel" id="roundel-acc">99.7%</div>
        <div class="roundel-caption">Best test accuracy on 12,630 GTSRB images</div>
      </div>
      <div class="roundel-item">
        <div class="roundel" id="roundel-robust">98.7%</div>
        <div class="roundel-caption">Robust CNN under FGSM attack (vs 82.9% plain)</div>
      </div>
      <div class="roundel-item">
        <div class="roundel" id="roundel-flops">28x</div>
        <div class="roundel-caption">Fewer FLOPs on MobileNetV3 for edge deployment</div>
      </div>
      <div class="roundel-item">
        <div class="roundel" id="roundel-classes">43</div>
        <div class="roundel-caption">Distinct traffic sign categories recognised</div>
      </div>
    </section>

    <!-- How It Works (4-Step Pipeline) -->
    <section class="pipeline-section" aria-label="How SignSense Works">
      <h2 class="section-title">How it works</h2>
      <div class="pipeline-grid">
        <div class="pipeline-step">
          <div class="step-number">01</div>
          <div class="step-title">Upload sign</div>
          <div class="step-desc">A road sign crop is captured via dashboard camera or uploaded by the user.</div>
        </div>
        <div class="pipeline-step">
          <div class="step-number">02</div>
          <div class="step-title">Resize to 64x64</div>
          <div class="step-desc">Standardized to 64x64 pixels and normalized inside the model tensor wrapper.</div>
        </div>
        <div class="pipeline-step">
          <div class="step-number">03</div>
          <div class="step-title">CNN predicts 1 of 43</div>
          <div class="step-desc">Convolutional stages extract high-frequency pictograms and compute class probabilities.</div>
        </div>
        <div class="pipeline-step">
          <div class="step-number">04</div>
          <div class="step-title">Grad-CAM shows why</div>
          <div class="step-desc">Gradient backpropagation produces heatmaps proving the model focused on the sign's symbol.</div>
        </div>
      </div>
    </section>

    <!-- The Four Modules (Team Contributions) -->
    <section class="modules-section" aria-label="Team Modules">
      <h2 class="section-title">The four modules</h2>
      <div class="modules-grid" id="modules-grid">
        <!-- Rendered dynamically from /api/team -->
      </div>
    </section>

    <!-- About the Dataset & Limitations -->
    <section class="info-columns" aria-label="Dataset and Limitations">
      <div class="panel">
        <h3 class="panel-title" style="margin-bottom: 8px;">About the dataset</h3>
        <p style="font-size: 14px; color: var(--muted); margin-bottom: 12px; line-height: 1.5;">
          The German Traffic Sign Recognition Benchmark (GTSRB) contains 51,839 real-world images across 43 classes.
          It exhibits a severe 10.71x class imbalance (2,250 samples for 50 km/h vs. 210 for 20 km/h), requiring label smoothing and Macro-F1 evaluation.
        </p>
        <div class="dataset-thumb-box" id="eda-samples-thumb">
          <img class="dataset-thumb-img" src="/outputs/eda_samples.png" alt="GTSRB 43 class grid overview">
          <div style="font-size: 12px; color: var(--sign-blue); font-weight: 600; margin-top: 6px;">
            Click to enlarge 43-class visual sample grid
          </div>
        </div>
      </div>

      <div class="panel">
        <h3 class="panel-title" style="margin-bottom: 8px;">Limitations & future scope</h3>
        <div style="display: flex; flex-direction: column; gap: 14px; font-size: 14px; color: var(--muted); line-height: 1.5;">
          <div>
            <strong style="color: var(--ink);">Current limitations:</strong>
            Works on pre-cropped traffic signs without bounding box regression; optimized for European signage conventions; defends against single-step FGSM.
          </div>
          <div>
            <strong style="color: var(--ink);">Future roadmap:</strong>
            Upstream YOLOv8 detector for raw 1080p dashcam video feeds; multi-step PGD adversarial training; and INT8 quantization for automotive microcontrollers.
          </div>
        </div>
      </div>
    </section>
  `;

  // Bind Lightbox to dataset thumbnail
  const thumb = document.getElementById('eda-samples-thumb');
  if (thumb) {
    thumb.addEventListener('click', () => {
      openLightbox({
        src: '/outputs/eda_samples.png',
        alt: 'GTSRB 43 traffic sign classes grid',
        caption: 'German Traffic Sign Recognition Benchmark (GTSRB) — 43 Classes',
      });
    });
  }

  // Load Dynamic Data: Models & Metrics
  loadModelRoundels();

  // Load Dynamic Data: Team Modules
  loadTeamModules();

  // Setup Live Road Ticker
  setupLiveRoadTicker();
}

/**
 * Load Dynamic Speed-Limit Roundels from /api/models
 */
async function loadModelRoundels() {
  try {
    const models = await api.getModels();
    if (!models || models.length === 0) return;

    let bestAcc = 0;
    let robustAcc = 0;
    let plainAcc = 0;
    let cnnFlops = 310.1;
    let mobFlops = 11.1;

    models.forEach((m) => {
      if (m.test_accuracy && m.test_accuracy > bestAcc) bestAcc = m.test_accuracy;
      if (m.key === 'cnn_robust' && m.fgsm_acc_002) robustAcc = m.fgsm_acc_002;
      if (m.key === 'cnn' && m.fgsm_acc_002) plainAcc = m.fgsm_acc_002;
      if (m.key === 'cnn' && m.MFLOPs) cnnFlops = m.MFLOPs;
      if (m.key === 'mobilenet' && m.MFLOPs) mobFlops = m.MFLOPs;
    });

    const roundelAcc = document.getElementById('roundel-acc');
    const roundelRobust = document.getElementById('roundel-robust');
    const roundelFlops = document.getElementById('roundel-flops');

    if (roundelAcc && bestAcc > 0) {
      roundelAcc.textContent = `${(bestAcc * 100).toFixed(1)}%`;
    }
    if (roundelRobust && robustAcc > 0) {
      roundelRobust.textContent = `${(robustAcc * 100).toFixed(1)}%`;
    }
    if (roundelFlops && mobFlops > 0) {
      const ratio = Math.round(cnnFlops / mobFlops);
      roundelFlops.textContent = `${ratio}x`;
    }
  } catch (err) {
    console.warn('Using default roundel figures:', err);
  }
}

/**
 * Load Team Modules from /api/team
 */
async function loadTeamModules() {
  const grid = document.getElementById('modules-grid');
  if (!grid) return;

  try {
    const team = await api.getTeam();
    grid.innerHTML = '';

    const moduleMeta = [
      { badge: 'circle', link: '#/report', btnText: 'View training report' },
      { badge: 'circle', link: '#/compare', btnText: 'Compare models' },
      { badge: 'triangle', link: '#/attack', btnText: 'Open attack lab' },
      { badge: 'octagon', link: '#/recognise', btnText: 'Try recognise' },
    ];

    team.forEach((m, idx) => {
      const meta = moduleMeta[idx] || moduleMeta[0];
      const card = document.createElement('div');
      card.className = 'module-card';

      const header = document.createElement('div');
      header.className = 'module-card-header';
      const badge = createSignBadge({ type: meta.badge, label: `M${idx + 1}` });

      const titles = document.createElement('div');
      const h3 = document.createElement('h3');
      h3.className = 'module-title';
      h3.textContent = m.module;
      const memName = document.createElement('div');
      memName.className = 'module-member-name';
      memName.textContent = m.name;
      titles.appendChild(h3);
      titles.appendChild(memName);

      header.appendChild(badge);
      header.appendChild(titles);

      const summary = document.createElement('div');
      summary.className = 'module-summary';
      summary.textContent = m.summary;

      const syllabus = document.createElement('div');
      syllabus.className = 'module-syllabus';
      syllabus.textContent = `Syllabus: ${m.syllabus || 'PECE-06'}`;

      const linkBtn = document.createElement('a');
      linkBtn.href = meta.link;
      linkBtn.className = 'btn btn-secondary';
      linkBtn.style.alignSelf = 'flex-start';
      linkBtn.textContent = meta.btnText;

      card.appendChild(header);
      card.appendChild(summary);
      card.appendChild(syllabus);
      card.appendChild(linkBtn);
      grid.appendChild(card);
    });
  } catch (err) {
    console.error('Failed to load team modules:', err);
  }
}

/**
 * Setup Hero Live Road Ticker
 */
async function setupLiveRoadTicker() {
  const signImg = document.getElementById('ticker-sign-img');
  const heatImg = document.getElementById('ticker-heatmap-img');
  const predName = document.getElementById('ticker-pred-name');
  const barContainer = document.getElementById('ticker-bar-container');
  const camToggle = document.getElementById('ticker-cam-toggle');
  const pauseBtn = document.getElementById('ticker-pause-btn');
  const tickerPanel = document.getElementById('road-ticker');
  const latencyLabel = document.getElementById('ticker-latency');

  if (!signImg || !predName) return;

  // Grad-CAM heatmap toggle
  if (camToggle && heatImg) {
    camToggle.addEventListener('change', () => {
      heatImg.classList.toggle('visible', camToggle.checked);
    });
  }

  // Hover and button pause
  if (tickerPanel) {
    tickerPanel.addEventListener('mouseenter', () => { isPaused = true; });
    tickerPanel.addEventListener('mouseleave', () => {
      if (pauseBtn && pauseBtn.textContent === 'Pause') isPaused = false;
    });
  }

  if (pauseBtn) {
    pauseBtn.addEventListener('click', () => {
      if (isPaused) {
        isPaused = false;
        pauseBtn.textContent = 'Pause';
      } else {
        isPaused = true;
        pauseBtn.textContent = 'Play';
      }
    });
  }

  // Load sample signs
  try {
    const allSamples = await api.getSamples();
    currentSamples = allSamples.filter((s) => !s.real);
  } catch (err) {
    console.error('Failed to load samples for ticker:', err);
  }

  if (!currentSamples || currentSamples.length === 0) return;

  // Process next sign
  async function advanceSign() {
    if (isPaused) return;

    sampleIndex = (sampleIndex + 1) % currentSamples.length;
    const sample = currentSamples[sampleIndex];

    try {
      if (abortController) abortController.abort();
      abortController = new AbortController();

      const res = await fetch(sample.url, { signal: abortController.signal });
      const blob = await res.blob();
      const file = new File([blob], sample.file, { type: 'image/png' });

      const pred = await api.predict(file, 'resnet18', abortController.signal);

      signImg.src = pred.input_b64;
      if (heatImg) heatImg.src = pred.heatmap_b64;
      predName.textContent = pred.class_name;
      if (latencyLabel) latencyLabel.textContent = `${pred.latency_ms} ms`;

      if (barContainer) {
        barContainer.innerHTML = '';
        const bar = createConfidenceBar(pred.confidence);
        barContainer.appendChild(bar);
      }
    } catch (e) {
      if (e.name !== 'AbortError') {
        console.warn('Ticker prediction error:', e);
      }
    }
  }

  // Initial prediction
  advanceSign();

  // Respect prefers-reduced-motion
  const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (!prefersReduced) {
    tickerInterval = setInterval(advanceSign, 3000);
  }
}

export function destroy() {
  if (tickerInterval) {
    clearInterval(tickerInterval);
    tickerInterval = null;
  }
  if (abortController) {
    abortController.abort();
    abortController = null;
  }
}
