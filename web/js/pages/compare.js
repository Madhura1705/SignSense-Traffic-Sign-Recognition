import { api } from '../api.js';
import {
  createPanel,
  createImageInput,
  createConfidenceBar,
  createSpinner,
  createErrorBanner,
  createEmptyState,
  openLightbox,
} from '../components/index.js';

let currentFile = null;
let currentSampleMeta = null;
let compareChart = null;
let abortController = null;
let modelsCache = [];

export async function render(container) {
  container.innerHTML = `
    <div class="page-header">
      <h1 class="page-title">Compare Models</h1>
      <p class="page-description">Run all four neural network architectures in parallel and evaluate model consensus via Ensemble Voting.</p>
    </div>

    <div class="compare-layout">
      <!-- Left Column: Image Input & Samples -->
      <div class="panel">
        <h3 class="panel-title" style="margin-bottom: 12px;">Input Sign</h3>
        <div id="compare-image-input-container"></div>
      </div>

      <!-- Right Column: Ensemble & 4-Model Results -->
      <div id="compare-results-panel">
        <!-- Rendered dynamically -->
      </div>
    </div>
  `;

  const inputContainer = document.getElementById('compare-image-input-container');
  const resultsPanel = document.getElementById('compare-results-panel');

  // Fetch model metadata cache
  try {
    modelsCache = await api.getModels();
  } catch (e) {
    console.error('Failed to load models metadata:', e);
  }

  // Load samples
  let standardSamples = [];
  let realSamples = [];
  try {
    const allSamples = await api.getSamples();
    standardSamples = allSamples.filter((s) => !s.real);
    realSamples = allSamples.filter((s) => s.real);
  } catch (e) {
    console.error('Failed to load samples:', e);
  }

  // Setup ImageInput component
  const imageInput = createImageInput({
    onImageSelected: (file, sampleMeta) => {
      currentFile = file;
      currentSampleMeta = sampleMeta;
      runCompare();
    },
    samples: standardSamples,
  });
  inputContainer.appendChild(imageInput);

  // Real-world photos sub-heading
  if (realSamples.length > 0) {
    const realBox = document.createElement('div');
    realBox.className = 'sample-strip-container';
    realBox.style.marginTop = '14px';

    const realTitle = document.createElement('div');
    realTitle.className = 'sample-strip-title';
    realTitle.textContent = 'Real-world photos (uncropped / domain shift):';
    realBox.appendChild(realTitle);

    const realStrip = document.createElement('div');
    realStrip.className = 'sample-strip';

    realSamples.forEach((sample) => {
      const img = document.createElement('img');
      img.className = 'sample-thumb';
      img.src = sample.url;
      img.alt = sample.file;
      img.title = sample.file;
      img.setAttribute('tabindex', '0');
      img.setAttribute('role', 'button');

      img.addEventListener('click', async () => {
        try {
          const res = await fetch(sample.url);
          const blob = await res.blob();
          const file = new File([blob], sample.file, { type: 'image/png' });
          currentFile = file;
          currentSampleMeta = sample;
          imageInput.setPreview(sample.url);
          runCompare();
        } catch (e) {
          console.error(e);
        }
      });

      realStrip.appendChild(img);
    });

    realBox.appendChild(realStrip);
    inputContainer.appendChild(realBox);
  }

  // Render initial empty state
  renderEmptyState(resultsPanel);
}

function renderEmptyState(panel) {
  panel.innerHTML = '';
  panel.appendChild(
    createEmptyState({
      title: 'No sign selected for comparison',
      subtitle: 'Upload a sign or select one of the samples to compare Custom CNN, ResNet-18, MobileNetV3, and Robust CNN.',
      icon: '⚖️',
    })
  );
}

/**
 * Execute Compare API call
 */
async function runCompare() {
  const panel = document.getElementById('compare-results-panel');
  if (!panel || !currentFile) return;

  if (abortController) abortController.abort();
  abortController = new AbortController();

  if (compareChart) {
    compareChart.destroy();
    compareChart = null;
  }

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
  loadText.textContent = 'Evaluating image across all 4 architectures...';
  loading.appendChild(loadText);
  panel.appendChild(loading);

  try {
    const data = await api.compare(currentFile, abortController.signal);
    renderCompareResults(panel, data);
  } catch (err) {
    if (err.name === 'AbortError') return;
    panel.innerHTML = '';
    panel.appendChild(createErrorBanner(err.message || 'Comparison failed.'));
  }
}

/**
 * Render Ensemble Banner, 4-Model Cards Grid, Chart, and Speed vs Size Table
 */
function renderCompareResults(panel, data) {
  panel.innerHTML = '';

  const { results, vote } = data;
  const isUnanimous = vote.count === vote.total;

  // 1. Ensemble Banner on top
  const banner = document.createElement('div');
  banner.className = `ensemble-banner ${isUnanimous ? 'unanimous' : 'split'}`;

  const bannerHeader = document.createElement('div');
  bannerHeader.textContent = isUnanimous
    ? `✓ ${vote.count} of ${vote.total} models agree: ${vote.class_name}`
    : `⚠️ Ensemble consensus vote: ${vote.class_name} (${vote.count} of ${vote.total} models agree)`;
  banner.appendChild(bannerHeader);

  if (!isUnanimous) {
    const disagreeing = results
      .filter((r) => r.class_id !== vote.class_id)
      .map((r) => `${r.model} predicted "${r.class_name}"`);
    const disDiv = document.createElement('div');
    disDiv.style.fontSize = '13px';
    disDiv.style.fontWeight = '500';
    disDiv.textContent = `Disagreeing: ${disagreeing.join('; ')}`;
    banner.appendChild(disDiv);
  }

  panel.appendChild(banner);

  // Find max confidence model
  let maxConf = -1;
  let winnerKey = null;
  results.forEach((r) => {
    if (r.confidence > maxConf) {
      maxConf = r.confidence;
      winnerKey = r.model;
    }
  });

  // 2. Responsive Grid of 4 Model Cards
  const grid = document.createElement('div');
  grid.className = 'compare-grid';

  results.forEach((r) => {
    const card = document.createElement('div');
    const isWinner = r.model === winnerKey;
    card.className = `compare-card ${isWinner ? 'winner' : ''}`;

    // Spec info from metadata cache
    const mMeta = modelsCache.find((m) => m.key === r.model) || {};
    const specText = `${mMeta.params_M || '-'}M params | ${mMeta.MFLOPs || '-'} MFLOPs | ${((mMeta.test_accuracy || 0) * 100).toFixed(1)}% test acc.`;

    const titleRow = document.createElement('div');
    titleRow.style.display = 'flex';
    titleRow.style.justifyContent = 'space-between';
    titleRow.style.alignItems = 'center';

    const cardTitle = document.createElement('div');
    cardTitle.className = 'compare-card-title';
    cardTitle.textContent = mMeta.name || r.model;

    titleRow.appendChild(cardTitle);
    if (isWinner) {
      const winBadge = document.createElement('span');
      winBadge.className = 'badge badge-green';
      winBadge.style.fontSize = '11px';
      winBadge.textContent = 'Highest confidence';
      titleRow.appendChild(winBadge);
    }

    const specDiv = document.createElement('div');
    specDiv.className = 'compare-card-spec';
    specDiv.textContent = specText;

    // Grad-CAM Overlay Image (Click to Lightbox)
    const img = document.createElement('img');
    img.className = 'compare-cam-img';
    img.src = r.overlay_b64;
    img.alt = `${mMeta.name || r.model} Grad-CAM overlay`;
    img.title = 'Click to enlarge in Lightbox';
    img.addEventListener('click', () => {
      openLightbox({
        src: r.overlay_b64,
        caption: `${mMeta.name || r.model} Grad-CAM Saliency`,
      });
    });

    // Prediction & Confidence
    const predRow = document.createElement('div');
    predRow.style.display = 'flex';
    predRow.style.justifyContent = 'space-between';
    predRow.style.fontSize = '14px';

    const pClass = document.createElement('strong');
    pClass.textContent = r.class_name;

    const pPct = document.createElement('span');
    pPct.style.color = 'var(--muted)';
    pPct.style.fontWeight = '600';
    pPct.textContent = `${(r.confidence * 100).toFixed(1)}%`;

    predRow.appendChild(pClass);
    predRow.appendChild(pPct);

    const bar = createConfidenceBar(r.confidence);

    const latDiv = document.createElement('div');
    latDiv.style.fontSize = '12px';
    latDiv.style.color = 'var(--muted)';
    latDiv.textContent = `Latency: ${r.latency_ms} ms on CPU`;

    card.appendChild(titleRow);
    card.appendChild(specDiv);
    card.appendChild(img);
    card.appendChild(predRow);
    card.appendChild(bar);
    card.appendChild(latDiv);
    grid.appendChild(card);
  });

  panel.appendChild(grid);

  // 3. Chart.js Horizontal Bar Chart & Speed vs Size Table
  const bottomPanel = document.createElement('div');
  bottomPanel.className = 'panel';

  const chartTitle = document.createElement('h3');
  chartTitle.className = 'panel-title';
  chartTitle.style.marginBottom = '12px';
  chartTitle.textContent = 'Confidence per model';

  const canvasContainer = document.createElement('div');
  canvasContainer.style.height = '180px';
  canvasContainer.style.position = 'relative';
  canvasContainer.style.marginBottom = '24px';

  const canvas = document.createElement('canvas');
  canvasContainer.appendChild(canvas);

  bottomPanel.appendChild(chartTitle);
  bottomPanel.appendChild(canvasContainer);

  // Speed vs Size Table
  const tableTitle = document.createElement('h3');
  tableTitle.className = 'panel-title';
  tableTitle.style.marginBottom = '8px';
  tableTitle.textContent = 'Speed vs size trade-off';

  const tableDesc = document.createElement('p');
  tableDesc.style.fontSize = '13.5px';
  tableDesc.style.color = 'var(--muted)';
  tableDesc.style.marginBottom = '12px';
  tableDesc.textContent = 'MobileNetV3-Small features 28x fewer FLOPs (11.1 MFLOPs) than Custom CNN, making it ideal for edge microcontrollers.';

  const table = document.createElement('table');
  table.className = 'speed-size-table';
  table.innerHTML = `
    <thead>
      <tr>
        <th>Model Architecture</th>
        <th>Latency (ms)</th>
        <th>Parameters (M)</th>
        <th>Compute (MFLOPs)</th>
        <th>Test Accuracy</th>
      </tr>
    </thead>
    <tbody>
      ${results
        .map((r) => {
          const m = modelsCache.find((meta) => meta.key === r.model) || {};
          const isMob = r.model === 'mobilenet';
          return `
            <tr class="${isMob ? 'highlight-mobilenet' : ''}">
              <td><strong>${m.name || r.model}</strong>${isMob ? ' (Edge optimized)' : ''}</td>
              <td>${r.latency_ms} ms</td>
              <td>${m.params_M || '-'} M</td>
              <td>${m.MFLOPs || '-'}</td>
              <td>${((m.test_accuracy || 0) * 100).toFixed(2)}%</td>
            </tr>
          `;
        })
        .join('')}
    </tbody>
  `;

  bottomPanel.appendChild(tableTitle);
  bottomPanel.appendChild(tableDesc);
  bottomPanel.appendChild(table);

  panel.appendChild(bottomPanel);

  // Render Horizontal Bar Chart
  if (window.Chart) {
    const labels = results.map((r) => {
      const m = modelsCache.find((meta) => meta.key === r.model);
      return m ? m.name : r.model;
    });
    const confData = results.map((r) => Math.round(r.confidence * 1000) / 10);

    compareChart = new window.Chart(canvas, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'Confidence (%)',
            data: confData,
            backgroundColor: results.map((r) =>
              r.model === winnerKey ? '#0A4DA2' : '#94A3B8'
            ),
            borderRadius: 4,
          },
        ],
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.parsed.x}% confidence`,
            },
          },
        },
        scales: {
          x: {
            min: 0,
            max: 100,
            ticks: { callback: (val) => `${val}%` },
            grid: { color: '#E2E8F0' },
          },
          y: {
            grid: { display: false },
          },
        },
      },
    });
  }
}

export function destroy() {
  if (compareChart) {
    compareChart.destroy();
    compareChart = null;
  }
  if (abortController) {
    abortController.abort();
    abortController = null;
  }
}
