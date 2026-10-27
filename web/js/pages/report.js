/**
 * SignSense Model Report & Benchmarks Page
 * Modular sub-tabs: Overview | Training | Errors | Explainability | Robustness | Edge | Dataset
 * Deep-linkable as #/report/<tab>
 */
import { api } from '../api.js';
import { openLightbox } from '../components/lightbox.js';
import { createSpinner, createErrorBanner } from '../components/feedback.js';

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'training', label: 'Training' },
  { id: 'errors', label: 'Errors' },
  { id: 'explainability', label: 'Explainability' },
  { id: 'robustness', label: 'Robustness' },
  { id: 'edge', label: 'Edge & ONNX' },
  { id: 'dataset', label: 'Dataset' },
];

const MODEL_NAMES = {
  cnn: 'Custom CNN (Baseline)',
  resnet18: 'ResNet-18',
  mobilenet: 'MobileNetV3-Small',
  cnn_robust: 'Robust CNN (FGSM-Trained)',
};

let activeCharts = [];
let abortController = null;
let reportData = null;
let activeTab = 'overview';
let containerRef = null;

// Tab-specific state
let trainingSelectedModel = 'cnn';
let errorsSelectedModel = 'cnn';
let errorsActiveView = 'confusion';
let explainSelectedModel = 'cnn';
let datasetActiveView = 'distribution';
let errorSearchTerm = '';
let overviewSortKey = 'test_accuracy';
let overviewSortAsc = false;

function triggerDownload(url, filename) {
  const a = document.createElement('a');
  a.href = url;
  a.download = filename || url.split('/').pop();
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

function destroyCharts() {
  activeCharts.forEach((c) => {
    try {
      c.destroy();
    } catch (_) {}
  });
  activeCharts = [];
}

function createImageViewer({ src, alt, caption, downloadName }) {
  const wrapper = document.createElement('div');
  wrapper.className = 'report-img-viewer';
  wrapper.innerHTML = `
    <img src="${src}" alt="${alt}" class="report-img" tabindex="0" role="button" aria-label="Click to enlarge ${alt}" />
    <div class="viewer-action-bar">
      <span>${caption || alt}</span>
      <div style="display: flex; gap: 8px;">
        <button class="btn btn-secondary btn-sm btn-zoom" type="button">Enlarge</button>
        <button class="btn btn-secondary btn-sm btn-download" type="button">Download PNG</button>
      </div>
    </div>
  `;

  const img = wrapper.querySelector('.report-img');
  const zoomBtn = wrapper.querySelector('.btn-zoom');
  const dlBtn = wrapper.querySelector('.btn-download');

  const openZoom = () => openLightbox({ src, alt, caption: caption || alt });
  img.addEventListener('click', openZoom);
  img.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      openZoom();
    }
  });
  zoomBtn.addEventListener('click', openZoom);
  dlBtn.addEventListener('click', () => triggerDownload(src, downloadName || src.split('/').pop()));

  return wrapper;
}

function createModelSelector(selectedKey, onChange) {
  const bar = document.createElement('div');
  bar.style.display = 'flex';
  bar.style.flexWrap = 'wrap';
  bar.style.gap = '8px';
  bar.style.marginBottom = '20px';

  Object.entries(MODEL_NAMES).forEach(([key, name]) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `btn btn-sm ${key === selectedKey ? 'btn-primary' : 'btn-secondary'}`;
    btn.textContent = name;
    btn.addEventListener('click', () => {
      onChange(key);
    });
    bar.appendChild(btn);
  });
  return bar;
}

/* ==========================================================================
   TAB 1: OVERVIEW
   ========================================================================== */
function renderOverviewTab(contentEl) {
  destroyCharts();

  const metrics = reportData.metrics || {};
  const comparison = reportData.comparison || [];

  // Prepare table data
  let tableRows = Object.keys(MODEL_NAMES).map((key) => {
    const csvRow = comparison.find((r) => r.model === key) || {};
    const jsonRow = metrics[key] || {};
    return {
      model: key,
      name: MODEL_NAMES[key],
      params_M: parseFloat(csvRow.params_M || jsonRow.params_M || 0),
      MFLOPs: parseFloat(csvRow.MFLOPs || jsonRow.MFLOPs || 0),
      test_accuracy: parseFloat(csvRow.test_accuracy || jsonRow.test_accuracy || 0),
      macro_f1: parseFloat(csvRow.macro_f1 || jsonRow.macro_f1 || 0),
      latency_ms: parseFloat(csvRow.latency_ms || jsonRow.latency_ms_per_image || 0),
      fgsm_acc_eps02: parseFloat(
        csvRow['fgsm_acc_eps0.02'] || (jsonRow.fgsm_accuracy && jsonRow.fgsm_accuracy['0.02']) || 0
      ),
    };
  });

  // Find best values
  const minParams = Math.min(...tableRows.map((r) => r.params_M));
  const minFlops = Math.min(...tableRows.map((r) => r.MFLOPs));
  const maxAcc = Math.max(...tableRows.map((r) => r.test_accuracy));
  const maxF1 = Math.max(...tableRows.map((r) => r.macro_f1));
  const minLatency = Math.min(...tableRows.map((r) => r.latency_ms));
  const maxFgsm = Math.max(...tableRows.map((r) => r.fgsm_acc_eps02));

  // Sort rows
  tableRows.sort((a, b) => {
    const valA = a[overviewSortKey];
    const valB = b[overviewSortKey];
    if (typeof valA === 'string') {
      return overviewSortAsc ? valA.localeCompare(valB) : valB.localeCompare(valA);
    }
    return overviewSortAsc ? valA - valB : valB - valA;
  });

  const sortArrow = (key) => {
    if (overviewSortKey !== key) return ' ↕';
    return overviewSortAsc ? ' ▲' : ' ▼';
  };

  const overviewHtml = `
    <div style="margin-bottom: 24px;">
      <h3 class="section-title">Model Comparison & Benchmark Summary</h3>
      <p style="color: var(--muted); font-size: 14px; margin-bottom: 16px;">
        Click column headers to sort models by metric. The best performing value in each column is highlighted in green.
      </p>

      <div style="overflow-x: auto;">
        <table class="report-table" id="overview-table">
          <thead>
            <tr>
              <th data-sort="model">Model${sortArrow('model')}</th>
              <th data-sort="params_M">Params (M)${sortArrow('params_M')}</th>
              <th data-sort="MFLOPs">MFLOPs${sortArrow('MFLOPs')}</th>
              <th data-sort="test_accuracy">Test Acc${sortArrow('test_accuracy')}</th>
              <th data-sort="macro_f1">Macro F1${sortArrow('macro_f1')}</th>
              <th data-sort="latency_ms">Latency (ms)${sortArrow('latency_ms')}</th>
              <th data-sort="fgsm_acc_eps02">FGSM ε=0.02 Acc${sortArrow('fgsm_acc_eps02')}</th>
            </tr>
          </thead>
          <tbody>
            ${tableRows
              .map(
                (row) => `
              <tr>
                <td style="font-weight: 600;">${row.name}</td>
                <td class="${row.params_M === minParams ? 'best-cell' : ''}">${row.params_M.toFixed(3)}M</td>
                <td class="${row.MFLOPs === minFlops ? 'best-cell' : ''}">${row.MFLOPs.toFixed(1)}</td>
                <td class="${row.test_accuracy === maxAcc ? 'best-cell' : ''}">${(row.test_accuracy * 100).toFixed(2)}%</td>
                <td class="${row.macro_f1 === maxF1 ? 'best-cell' : ''}">${row.macro_f1.toFixed(4)}</td>
                <td class="${row.latency_ms === minLatency ? 'best-cell' : ''}">${row.latency_ms.toFixed(2)} ms</td>
                <td class="${row.fgsm_acc_eps02 === maxFgsm ? 'best-cell' : ''}">${(row.fgsm_acc_eps02 * 100).toFixed(2)}%</td>
              </tr>
            `
              )
              .join('')}
          </tbody>
        </table>
      </div>
    </div>

    <div style="margin-top: 36px;">
      <h3 class="section-title">Clean vs Adversarial Accuracy Comparison</h3>
      <p style="color: var(--muted); font-size: 14px; margin-bottom: 16px;">
        Comparison of clean test accuracy against perturbed test accuracy under Fast Gradient Sign Method (FGSM, ε = 0.02).
      </p>
      <div style="position: relative; width: 100%; max-width: 820px; height: 360px; margin: 0 auto;">
        <canvas id="overview-chart"></canvas>
      </div>
    </div>
  `;

  contentEl.innerHTML = overviewHtml;

  // Header Sort Listeners
  contentEl.querySelectorAll('#overview-table th').forEach((th) => {
    th.addEventListener('click', () => {
      const sortKey = th.getAttribute('data-sort');
      if (overviewSortKey === sortKey) {
        overviewSortAsc = !overviewSortAsc;
      } else {
        overviewSortKey = sortKey;
        overviewSortAsc = sortKey === 'model' || sortKey === 'params_M' || sortKey === 'MFLOPs' || sortKey === 'latency_ms';
      }
      renderOverviewTab(contentEl);
    });
  });

  // Chart.js Bar Chart
  if (window.Chart) {
    const canvas = contentEl.querySelector('#overview-chart');
    if (canvas) {
      const labels = Object.keys(MODEL_NAMES).map((k) => MODEL_NAMES[k].split(' ')[0]);
      const cleanAcc = Object.keys(MODEL_NAMES).map(
        (k) => (parseFloat(metrics[k]?.test_accuracy || 0) * 100)
      );
      const fgsmAcc = Object.keys(MODEL_NAMES).map(
        (k) => (parseFloat(metrics[k]?.fgsm_accuracy?.['0.02'] || 0) * 100)
      );

      const chart = new window.Chart(canvas, {
        type: 'bar',
        data: {
          labels,
          datasets: [
            {
              label: 'Clean Test Accuracy (%)',
              data: cleanAcc,
              backgroundColor: 'rgba(26, 115, 232, 0.85)',
              borderColor: 'rgb(26, 115, 232)',
              borderWidth: 1,
              borderRadius: 4,
            },
            {
              label: 'FGSM ε = 0.02 Accuracy (%)',
              data: fgsmAcc,
              backgroundColor: 'rgba(217, 48, 37, 0.85)',
              borderColor: 'rgb(217, 48, 37)',
              borderWidth: 1,
              borderRadius: 4,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { position: 'top' },
            tooltip: {
              callbacks: {
                label: (ctx) => ` ${ctx.dataset.label}: ${ctx.raw.toFixed(2)}%`,
              },
            },
          },
          scales: {
            y: {
              min: 50,
              max: 100,
              title: { display: true, text: 'Accuracy (%)' },
              grid: { color: 'rgba(0, 0, 0, 0.05)' },
            },
            x: {
              grid: { display: false },
            },
          },
        },
      });
      activeCharts.push(chart);
    }
  }
}

/* ==========================================================================
   TAB 2: TRAINING
   ========================================================================== */
function renderTrainingTab(contentEl) {
  destroyCharts();

  const metrics = reportData.metrics || {};
  const modelMetrics = metrics[trainingSelectedModel] || {};
  const history = modelMetrics.history || {};

  contentEl.innerHTML = `
    <div style="margin-bottom: 20px;">
      <h3 class="section-title">Training & Validation Histories</h3>
      <p style="color: var(--muted); font-size: 14px; margin-bottom: 16px;">
        Select an architecture to inspect epoch-by-epoch loss convergence, validation accuracy, and track-leakage notes.
      </p>
      <div id="training-model-selector"></div>
    </div>

    <div style="display: flex; justify-content: flex-end; margin-bottom: 16px;">
      <button class="btn btn-secondary btn-sm" id="btn-view-curves">
        View Static Plot in Lightbox
      </button>
    </div>

    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 28px;">
      <div style="background-color: var(--panel); border: 1px solid var(--border); border-radius: var(--radius-panel); padding: 18px;">
        <h4 style="font-size: 15px; font-weight: 600; margin-bottom: 12px; color: var(--ink);">Loss Convergence (Cross-Entropy)</h4>
        <div style="position: relative; width: 100%; height: 280px;">
          <canvas id="training-loss-chart"></canvas>
        </div>
      </div>
      <div style="background-color: var(--panel); border: 1px solid var(--border); border-radius: var(--radius-panel); padding: 18px;">
        <h4 style="font-size: 15px; font-weight: 600; margin-bottom: 12px; color: var(--ink);">Accuracy Progression (%)</h4>
        <div style="position: relative; width: 100%; height: 280px;">
          <canvas id="training-acc-chart"></canvas>
        </div>
      </div>
    </div>

    <div style="background-color: #FEF3C7; border-left: 4px solid #F59E0B; padding: 16px 20px; border-radius: var(--radius-sm); font-size: 14px; color: #78350F; line-height: 1.5;">
      <strong>Academic Note on GTSRB Track-Sequence Leakage:</strong><br>
      Validation accuracy during training climbs rapidly toward ~99–100%. GTSRB was captured from continuous vehicle camera video tracks where consecutive frames share identical lighting, angle, and background. When randomly splitting tracks across train and validation folds, near-identical frames leak into the validation set. True out-of-sample generalization is accurately evaluated only on the track-independent test set (${(parseFloat(modelMetrics.test_accuracy || 0) * 100).toFixed(2)}% for ${MODEL_NAMES[trainingSelectedModel]}).
    </div>
  `;

  // Attach Model Selector
  const selectorContainer = contentEl.querySelector('#training-model-selector');
  selectorContainer.appendChild(
    createModelSelector(trainingSelectedModel, (newModel) => {
      trainingSelectedModel = newModel;
      renderTrainingTab(contentEl);
    })
  );

  // Lightbox Button
  contentEl.querySelector('#btn-view-curves').addEventListener('click', () => {
    openLightbox({
      src: `/outputs/${trainingSelectedModel}_curves.png`,
      alt: `${MODEL_NAMES[trainingSelectedModel]} Training Curves`,
      caption: `${MODEL_NAMES[trainingSelectedModel]} — Training Loss & Accuracy Curves`,
    });
  });

  // Render Charts
  const epochs = (history.train_loss || []).map((_, i) => `Ep ${i + 1}`);

  if (window.Chart && epochs.length > 0) {
    const lossCanvas = contentEl.querySelector('#training-loss-chart');
    if (lossCanvas) {
      const lossChart = new window.Chart(lossCanvas, {
        type: 'line',
        data: {
          labels: epochs,
          datasets: [
            {
              label: 'Train Loss',
              data: history.train_loss || [],
              borderColor: '#1A73E8',
              backgroundColor: 'rgba(26, 115, 232, 0.1)',
              tension: 0.2,
              fill: true,
            },
            {
              label: 'Val Loss',
              data: history.val_loss || [],
              borderColor: '#E37400',
              backgroundColor: 'transparent',
              borderDash: [5, 5],
              tension: 0.2,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { position: 'top' } },
          scales: {
            y: { title: { display: true, text: 'Loss' }, grid: { color: 'rgba(0, 0, 0, 0.05)' } },
            x: { grid: { display: false } },
          },
        },
      });
      activeCharts.push(lossChart);
    }

    const accCanvas = contentEl.querySelector('#training-acc-chart');
    if (accCanvas) {
      const accChart = new window.Chart(accCanvas, {
        type: 'line',
        data: {
          labels: epochs,
          datasets: [
            {
              label: 'Train Acc (%)',
              data: (history.train_acc || []).map((v) => v * 100),
              borderColor: '#1E8E3E',
              backgroundColor: 'rgba(30, 142, 62, 0.1)',
              tension: 0.2,
              fill: true,
            },
            {
              label: 'Val Acc (%)',
              data: (history.val_acc || []).map((v) => v * 100),
              borderColor: '#D93025',
              backgroundColor: 'transparent',
              borderDash: [5, 5],
              tension: 0.2,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { position: 'top' } },
          scales: {
            y: {
              min: 80,
              max: 100,
              title: { display: true, text: 'Accuracy (%)' },
              grid: { color: 'rgba(0, 0, 0, 0.05)' },
            },
            x: { grid: { display: false } },
          },
        },
      });
      activeCharts.push(accChart);
    }
  }
}

/* ==========================================================================
   TAB 3: ERRORS
   ========================================================================== */
function renderErrorsTab(contentEl) {
  contentEl.innerHTML = `
    <div style="margin-bottom: 20px;">
      <h3 class="section-title">Error Analysis & Confusion Matrices</h3>
      <p style="color: var(--muted); font-size: 14px; margin-bottom: 16px;">
        Examine failure modes, multi-class confusion patterns, and per-class precision/recall breakdown across all 43 classes.
      </p>
      <div id="errors-model-selector"></div>
    </div>

    <div style="display: flex; gap: 8px; margin-bottom: 20px; border-bottom: 1px solid var(--border); padding-bottom: 12px;">
      <button class="btn btn-sm ${errorsActiveView === 'confusion' ? 'btn-primary' : 'btn-secondary'}" id="view-confusion-btn">
        Confusion Matrix
      </button>
      <button class="btn btn-sm ${errorsActiveView === 'misclassified' ? 'btn-primary' : 'btn-secondary'}" id="view-misclassified-btn">
        Most Confident Mistakes
      </button>
      <button class="btn btn-sm ${errorsActiveView === 'classes' ? 'btn-primary' : 'btn-secondary'}" id="view-classes-btn">
        Per-Class Scores (43 Classes)
      </button>
    </div>

    <div id="errors-display-area"></div>
  `;

  // Attach Model Selector
  const selectorContainer = contentEl.querySelector('#errors-model-selector');
  selectorContainer.appendChild(
    createModelSelector(errorsSelectedModel, (newModel) => {
      errorsSelectedModel = newModel;
      renderErrorsTab(contentEl);
    })
  );

  // View switch buttons
  contentEl.querySelector('#view-confusion-btn').addEventListener('click', () => {
    errorsActiveView = 'confusion';
    renderErrorsTab(contentEl);
  });
  contentEl.querySelector('#view-misclassified-btn').addEventListener('click', () => {
    errorsActiveView = 'misclassified';
    renderErrorsTab(contentEl);
  });
  contentEl.querySelector('#view-classes-btn').addEventListener('click', () => {
    errorsActiveView = 'classes';
    renderErrorsTab(contentEl);
  });

  const displayArea = contentEl.querySelector('#errors-display-area');

  if (errorsActiveView === 'confusion') {
    displayArea.appendChild(
      createImageViewer({
        src: `/outputs/${errorsSelectedModel}_confusion.png`,
        alt: `${MODEL_NAMES[errorsSelectedModel]} 43-Class Confusion Matrix`,
        caption: `${MODEL_NAMES[errorsSelectedModel]} Confusion Matrix (Normalized along true class rows)`,
        downloadName: `${errorsSelectedModel}_confusion.png`,
      })
    );
  } else if (errorsActiveView === 'misclassified') {
    displayArea.appendChild(
      createImageViewer({
        src: `/outputs/${errorsSelectedModel}_misclassified.png`,
        alt: `${MODEL_NAMES[errorsSelectedModel]} Most Confident Misclassifications`,
        caption: `${MODEL_NAMES[errorsSelectedModel]} Most Confident Errors: True vs Predicted labels with softmax confidence`,
        downloadName: `${errorsSelectedModel}_misclassified.png`,
      })
    );
  } else {
    // Per-class scores table with search
    const classList = (reportData.class_reports && reportData.class_reports[errorsSelectedModel]) || [];
    const worstList = (reportData.worst_classes && reportData.worst_classes[errorsSelectedModel]) || [];
    const worstNames = new Set(worstList.map((w) => w.class_name));

    const renderTable = () => {
      const filtered = classList.filter((c) =>
        c.class_name.toLowerCase().includes(errorSearchTerm.toLowerCase())
      );

      displayArea.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; flex-wrap: wrap; gap: 12px;">
          <input type="text" class="search-input" id="class-search" placeholder="Filter by class name..." value="${errorSearchTerm}" />
          <span style="font-size: 13px; color: var(--muted);">Showing ${filtered.length} of ${classList.length} classes (Amber = 5 lowest F1)</span>
        </div>
        <div style="overflow-x: auto;">
          <table class="report-table">
            <thead>
              <tr>
                <th>Traffic Sign Class</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>Support</th>
              </tr>
            </thead>
            <tbody>
              ${filtered
                .map(
                  (c) => `
                <tr class="${worstNames.has(c.class_name) ? 'worst-class-row' : ''}">
                  <td style="font-weight: 500;">${c.class_name} ${worstNames.has(c.class_name) ? '⚠️' : ''}</td>
                  <td>${(c.precision * 100).toFixed(1)}%</td>
                  <td>${(c.recall * 100).toFixed(1)}%</td>
                  <td style="font-weight: 600;">${(c.f1 * 100).toFixed(1)}%</td>
                  <td>${c.support}</td>
                </tr>
              `
                )
                .join('')}
            </tbody>
          </table>
        </div>
      `;

      const input = displayArea.querySelector('#class-search');
      input.addEventListener('input', (e) => {
        errorSearchTerm = e.target.value;
        renderTable();
        const nextInput = displayArea.querySelector('#class-search');
        if (nextInput) {
          nextInput.focus();
          nextInput.setSelectionRange(errorSearchTerm.length, errorSearchTerm.length);
        }
      });
    };

    renderTable();
  }
}

/* ==========================================================================
   TAB 4: EXPLAINABILITY
   ========================================================================== */
function renderExplainabilityTab(contentEl) {
  contentEl.innerHTML = `
    <div style="margin-bottom: 20px;">
      <h3 class="section-title">Model Interpretability via Grad-CAM</h3>
      <p style="color: var(--muted); font-size: 14px; margin-bottom: 16px;">
        Visual explanations verifying that CNN predictions attend to distinctive sign glyphs and geometric contours rather than background clutter.
      </p>
      <div id="explain-model-selector"></div>
    </div>

    <div style="display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 28px; align-items: start;">
      <div id="explain-viewer-container"></div>
      
      <div style="background-color: var(--panel); border: 1px solid var(--border); border-radius: var(--radius-panel); padding: 22px;">
        <h4 style="font-size: 16px; font-weight: 700; margin-bottom: 14px; color: var(--ink);">How Grad-CAM Computes Heatmaps</h4>
        <ol style="margin: 0; padding-left: 20px; font-size: 14px; color: var(--ink); line-height: 1.6; display: flex; flex-direction: column; gap: 10px;">
          <li>
            <strong>Gradient Flow:</strong> Compute the gradient of the predicted class score $y^c$ with respect to the activation maps $A^k$ of the final convolutional layer: $\\frac{\\partial y^c}{\\partial A^k}$.
          </li>
          <li>
            <strong>Importance Weights:</strong> Global-average-pool gradients over spatial dimensions $(i, j)$ to capture channel importance:
            <br>$\\alpha_k^c = \\frac{1}{Z} \\sum_i \\sum_j \\frac{\\partial y^c}{\\partial A_{i,j}^k}$.
          </li>
          <li>
            <strong>Linear Combination:</strong> Compute a weighted sum of forward activation maps: $\\sum_k \\alpha_k^c A^k$.
          </li>
          <li>
            <strong>Rectification (ReLU):</strong> Apply $\\text{ReLU}$ to isolate features that positively contribute to class $c$:
            <br>$L_{\\text{Grad-CAM}}^c = \\text{ReLU}\\left(\\sum_k \\alpha_k^c A^k\\right)$.
          </li>
          <li>
            <strong>Bilinear Upsampling:</strong> Upsample coarse heatmaps from feature dimensions ($16 \\times 16$) to input dimensions ($64 \\times 64$) and apply the Turbo colormap.
          </li>
        </ol>

        <div style="margin-top: 24px; padding-top: 18px; border-top: 1px solid var(--border); text-align: center;">
          <p style="font-size: 13px; color: var(--muted); margin-bottom: 12px;">
            Want to inspect your own traffic sign image with live adjustable Grad-CAM opacity?
          </p>
          <a href="#/recognise" class="btn btn-primary btn-sm">Try It Live on Recognise Page →</a>
        </div>
      </div>
    </div>
  `;

  // Model Selector
  const selector = contentEl.querySelector('#explain-model-selector');
  selector.appendChild(
    createModelSelector(explainSelectedModel, (newModel) => {
      explainSelectedModel = newModel;
      renderExplainabilityTab(contentEl);
    })
  );

  // Viewer
  const viewerContainer = contentEl.querySelector('#explain-viewer-container');
  viewerContainer.appendChild(
    createImageViewer({
      src: `/outputs/${explainSelectedModel}_gradcam.png`,
      alt: `${MODEL_NAMES[explainSelectedModel]} Grad-CAM Interpretability Grid`,
      caption: `${MODEL_NAMES[explainSelectedModel]} — 8-Sample Grad-CAM Activation Overlays`,
      downloadName: `${explainSelectedModel}_gradcam.png`,
    })
  );
}

/* ==========================================================================
   TAB 5: ROBUSTNESS
   ========================================================================== */
function renderRobustnessTab(contentEl) {
  destroyCharts();

  contentEl.innerHTML = `
    <div style="margin-bottom: 20px;">
      <h3 class="section-title">Adversarial Robustness Evaluation</h3>
      <p style="color: var(--muted); font-size: 14px; margin-bottom: 16px;">
        Empirical evaluation under single-step Fast Gradient Sign Method (FGSM) attacks across perturbation magnitudes $\\epsilon \\in [0, 0.08]$.
      </p>
    </div>

    <div style="display: flex; justify-content: flex-end; margin-bottom: 14px;">
      <button class="btn btn-secondary btn-sm" id="btn-view-robust-plot">
        View Static Curve in Lightbox
      </button>
    </div>

    <div style="background-color: var(--panel); border: 1px solid var(--border); border-radius: var(--radius-panel); padding: 20px; margin-bottom: 24px;">
      <h4 style="font-size: 15px; font-weight: 600; margin-bottom: 12px; color: var(--ink);">Test Accuracy vs FGSM Perturbation Magnitude ($\epsilon$)</h4>
      <div style="position: relative; width: 100%; height: 350px;">
        <canvas id="robustness-chart"></canvas>
      </div>
    </div>

    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px;">
      <div style="background-color: var(--go-green-bg); border-left: 4px solid var(--go-green); padding: 18px; border-radius: var(--radius-sm); font-size: 14px; color: #064E3B; line-height: 1.5;">
        <strong>Key Research Finding:</strong><br>
        Adversarial training (${MODEL_NAMES.cnn_robust}) maintains <strong>98.71% test accuracy</strong> at $\\epsilon = 0.02$, whereas the standard baseline CNN plunges to <strong>82.89%</strong> (and 33.79% at $\\epsilon = 0.08$). Crucially, adversarial training introduced <strong>no penalty on clean test data</strong> (99.69% vs 99.59%).
      </div>
      <div style="background-color: var(--panel); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 18px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;">
        <div style="font-size: 24px; margin-bottom: 8px;">🛡️ ⚔️</div>
        <p style="font-size: 14px; color: var(--ink); margin-bottom: 12px; font-weight: 500;">
          Test live adversarial perturbation on real signs in Attack Lab:
        </p>
        <a href="#/attack" class="btn btn-primary btn-sm">Open Attack Lab →</a>
      </div>
    </div>
  `;

  contentEl.querySelector('#btn-view-robust-plot').addEventListener('click', () => {
    openLightbox({
      src: '/outputs/fgsm_robustness.png',
      alt: 'FGSM Robustness Curve',
      caption: 'Model Accuracy under Fast Gradient Sign Method Perturbation (ε = 0 to 0.08)',
    });
  });

  // Chart.js Multi-line chart
  if (window.Chart) {
    const canvas = contentEl.querySelector('#robustness-chart');
    if (canvas) {
      const metrics = reportData.metrics || {};
      const epsKeys = ['0', '0.005', '0.01', '0.02', '0.04', '0.08'];

      const colors = {
        cnn: { border: '#1A73E8', bg: 'rgba(26, 115, 232, 0.1)' },
        resnet18: { border: '#1E8E3E', bg: 'rgba(30, 142, 62, 0.1)' },
        mobilenet: { border: '#E37400', bg: 'rgba(227, 116, 0, 0.1)' },
        cnn_robust: { border: '#D93025', bg: 'rgba(217, 48, 37, 0.1)' },
      };

      const datasets = Object.keys(MODEL_NAMES).map((key) => {
        const fgsmObj = metrics[key]?.fgsm_accuracy || {};
        const data = epsKeys.map((eps) => (parseFloat(fgsmObj[eps] || 0) * 100));
        return {
          label: MODEL_NAMES[key],
          data,
          borderColor: colors[key].border,
          backgroundColor: colors[key].bg,
          borderWidth: key === 'cnn_robust' ? 3 : 1.8,
          pointRadius: 4,
          tension: 0.2,
        };
      });

      const chart = new window.Chart(canvas, {
        type: 'line',
        data: {
          labels: epsKeys.map((e) => `ε = ${e}`),
          datasets,
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { position: 'top' },
            tooltip: {
              callbacks: {
                label: (ctx) => ` ${ctx.dataset.label}: ${ctx.raw.toFixed(2)}%`,
              },
            },
          },
          scales: {
            y: {
              min: 20,
              max: 100,
              title: { display: true, text: 'Test Accuracy (%)' },
              grid: { color: 'rgba(0, 0, 0, 0.05)' },
            },
            x: {
              title: { display: true, text: 'Perturbation Magnitude (ε)' },
              grid: { display: false },
            },
          },
        },
      });
      activeCharts.push(chart);
    }
  }
}

/* ==========================================================================
   TAB 6: EDGE & ONNX
   ========================================================================== */
function renderEdgeTab(contentEl) {
  destroyCharts();

  const onnxRows = reportData.onnx || [];

  contentEl.innerHTML = `
    <div style="margin-bottom: 20px;">
      <h3 class="section-title">Edge Deployment & ONNX Runtime Benchmark</h3>
      <p style="color: var(--muted); font-size: 14px; margin-bottom: 16px;">
        Evaluation of exported ONNX models across runtime latency, model binary footprint, and numeric parity.
      </p>
    </div>

    <div style="overflow-x: auto; margin-bottom: 28px;">
      <table class="report-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>ONNX Size (MB)</th>
            <th>PyTorch CPU (ms)</th>
            <th>ONNX Runtime CPU (ms)</th>
            <th>Speed-Up</th>
            <th>Max Parity Diff</th>
            <th>Test Acc</th>
          </tr>
        </thead>
        <tbody>
          ${onnxRows
            .map(
              (r) => `
            <tr>
              <td style="font-weight: 600;">${MODEL_NAMES[r.model] || r.model}</td>
              <td class="${parseFloat(r.onnx_size_MB) <= 3.3 ? 'best-cell' : ''}">${parseFloat(r.onnx_size_MB).toFixed(2)} MB</td>
              <td>${parseFloat(r.torch_cpu_ms).toFixed(2)} ms</td>
              <td class="${parseFloat(r.onnxruntime_cpu_ms) <= 1.0 ? 'best-cell' : ''}">${parseFloat(r.onnxruntime_cpu_ms).toFixed(3)} ms</td>
              <td class="${parseFloat(r.speedup) >= 7.0 ? 'best-cell' : ''}"><strong>${parseFloat(r.speedup).toFixed(2)}x</strong></td>
              <td style="font-family: monospace; font-size: 12px;">${r.max_abs_diff}</td>
              <td>${(parseFloat(r.test_accuracy) * 100).toFixed(2)}%</td>
            </tr>
          `
            )
            .join('')}
        </tbody>
      </table>
    </div>

    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px; align-items: start;">
      <div style="background-color: var(--panel); border: 1px solid var(--border); border-radius: var(--radius-panel); padding: 18px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
          <h4 style="font-size: 15px; font-weight: 600; color: var(--ink); margin: 0;">Latency vs Accuracy Trade-off</h4>
          <button class="btn btn-secondary btn-sm" id="btn-view-latency-plot">Lightbox</button>
        </div>
        <div style="position: relative; width: 100%; height: 290px;">
          <canvas id="edge-scatter-chart"></canvas>
        </div>
      </div>

      <div style="background-color: var(--panel); border: 1px solid var(--border); border-radius: var(--radius-panel); padding: 22px;">
        <h4 style="font-size: 16px; font-weight: 700; margin-bottom: 14px; color: var(--ink);">Architectural Trade-off Analysis</h4>
        <div style="font-size: 14px; color: var(--ink); line-height: 1.6; display: flex; flex-direction: column; gap: 12px;">
          <p>
            <strong>MobileNetV3-Small:</strong> Features <strong>28x fewer FLOPs</strong> (11.1 MFLOPs vs 310.1 MFLOPs) and achieves an astonishing <strong>0.74 ms</strong> CPU inference under ONNX Runtime (a <strong>7.94x acceleration</strong> over standard PyTorch).
          </p>
          <p>
            <strong>GPU Batch-1 Caveat:</strong> Because MobileNet utilizes depthwise separable convolutions across numerous fine-grained layers, it incurs higher CUDA kernel launch overhead when evaluated on GPUs at batch size 1. On CPU and microcontrollers, however, its reduced memory bandwidth and FLOP profile make it the clear edge champion.
          </p>
          <p>
            <strong>Custom CNN & ResNet-18:</strong> Standard convolutional kernels map efficiently to modern SIMD units, achieving 2.39x–2.62x speed-ups under ONNX Runtime with numerical differences consistently below $10^{-5}$.
          </p>
        </div>
      </div>
    </div>
  `;

  contentEl.querySelector('#btn-view-latency-plot').addEventListener('click', () => {
    openLightbox({
      src: '/outputs/latency_vs_accuracy.png',
      alt: 'Latency vs Accuracy Trade-off',
      caption: 'Model Latency vs Test Accuracy Trade-off (bubble area proportional to parameters)',
    });
  });

  // Chart.js Bar / Scatter
  if (window.Chart && onnxRows.length > 0) {
    const canvas = contentEl.querySelector('#edge-scatter-chart');
    if (canvas) {
      const labels = onnxRows.map((r) => MODEL_NAMES[r.model]?.split(' ')[0] || r.model);
      const torchMs = onnxRows.map((r) => parseFloat(r.torch_cpu_ms));
      const onnxMs = onnxRows.map((r) => parseFloat(r.onnxruntime_cpu_ms));

      const chart = new window.Chart(canvas, {
        type: 'bar',
        data: {
          labels,
          datasets: [
            {
              label: 'PyTorch CPU (ms)',
              data: torchMs,
              backgroundColor: 'rgba(100, 116, 139, 0.75)',
              borderRadius: 4,
            },
            {
              label: 'ONNX Runtime CPU (ms)',
              data: onnxMs,
              backgroundColor: 'rgba(30, 142, 62, 0.85)',
              borderRadius: 4,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { position: 'top' },
            tooltip: {
              callbacks: {
                label: (ctx) => ` ${ctx.dataset.label}: ${ctx.raw.toFixed(2)} ms`,
              },
            },
          },
          scales: {
            y: {
              title: { display: true, text: 'Latency (ms / image)' },
              grid: { color: 'rgba(0, 0, 0, 0.05)' },
            },
            x: { grid: { display: false } },
          },
        },
      });
      activeCharts.push(chart);
    }
  }
}

/* ==========================================================================
   TAB 7: DATASET
   ========================================================================== */
function renderDatasetTab(contentEl) {
  const plots = {
    distribution: {
      src: '/outputs/eda_class_distribution.png',
      alt: 'GTSRB Class Distribution Histogram',
      caption: 'Distribution of training and test samples across all 43 German Traffic Sign classes',
    },
    samples: {
      src: '/outputs/eda_samples.png',
      alt: 'One Sample Per GTSRB Class',
      caption: 'Representative training image for each of the 43 sign categories',
    },
    sizes: {
      src: '/outputs/eda_image_sizes.png',
      alt: 'GTSRB Raw Image Dimensions',
      caption: 'Raw input image resolution distribution (ranging from 15x15 to 250x250 pixels)',
    },
    augmentation: {
      src: '/outputs/eda_augmentation.png',
      alt: 'Training Data Augmentation Pipeline',
      caption: 'Online data augmentations: random rotation, scaling, translation, perspective, and color jitter',
    },
  };

  contentEl.innerHTML = `
    <div style="margin-bottom: 20px;">
      <h3 class="section-title">Dataset Characteristics & Exploratory Data Analysis</h3>
      <p style="color: var(--muted); font-size: 14px; margin-bottom: 16px;">
        Inspect class distribution imbalances, raw bounding dimensions, and data augmentation transforms applied to GTSRB.
      </p>
    </div>

    <div style="display: flex; gap: 8px; margin-bottom: 20px; border-bottom: 1px solid var(--border); padding-bottom: 12px; flex-wrap: wrap;">
      <button class="btn btn-sm ${datasetActiveView === 'distribution' ? 'btn-primary' : 'btn-secondary'}" id="view-dist-btn">
        Class Distribution
      </button>
      <button class="btn btn-sm ${datasetActiveView === 'samples' ? 'btn-primary' : 'btn-secondary'}" id="view-samples-btn">
        One Example Per Class
      </button>
      <button class="btn btn-sm ${datasetActiveView === 'sizes' ? 'btn-primary' : 'btn-secondary'}" id="view-sizes-btn">
        Image Sizes
      </button>
      <button class="btn btn-sm ${datasetActiveView === 'augmentation' ? 'btn-primary' : 'btn-secondary'}" id="view-aug-btn">
        Augmentation
      </button>
    </div>

    <div id="dataset-viewer-area"></div>
  `;

  contentEl.querySelector('#view-dist-btn').addEventListener('click', () => {
    datasetActiveView = 'distribution';
    renderDatasetTab(contentEl);
  });
  contentEl.querySelector('#view-samples-btn').addEventListener('click', () => {
    datasetActiveView = 'samples';
    renderDatasetTab(contentEl);
  });
  contentEl.querySelector('#view-sizes-btn').addEventListener('click', () => {
    datasetActiveView = 'sizes';
    renderDatasetTab(contentEl);
  });
  contentEl.querySelector('#view-aug-btn').addEventListener('click', () => {
    datasetActiveView = 'augmentation';
    renderDatasetTab(contentEl);
  });

  const viewerArea = contentEl.querySelector('#dataset-viewer-area');
  const activePlot = plots[datasetActiveView];
  viewerArea.appendChild(
    createImageViewer({
      src: activePlot.src,
      alt: activePlot.alt,
      caption: activePlot.caption,
      downloadName: activePlot.src.split('/').pop(),
    })
  );
}

/* ==========================================================================
   MAIN RENDER & SUBNAV ROUTING
   ========================================================================== */
function switchTab(tabId) {
  activeTab = tabId;

  // Update hash without causing recursive router re-render
  history.pushState(null, '', `#/report/${tabId}`);

  // Update subnav buttons active state
  if (containerRef) {
    containerRef.querySelectorAll('.report-tab-btn').forEach((btn) => {
      const match = btn.getAttribute('data-tab') === tabId;
      btn.classList.toggle('active', match);
    });

    const contentEl = containerRef.querySelector('#report-tab-content');
    if (!contentEl) return;

    if (tabId === 'overview') renderOverviewTab(contentEl);
    else if (tabId === 'training') renderTrainingTab(contentEl);
    else if (tabId === 'errors') renderErrorsTab(contentEl);
    else if (tabId === 'explainability') renderExplainabilityTab(contentEl);
    else if (tabId === 'robustness') renderRobustnessTab(contentEl);
    else if (tabId === 'edge') renderEdgeTab(contentEl);
    else if (tabId === 'dataset') renderDatasetTab(contentEl);
  }
}

function handleHashChange() {
  const hash = window.location.hash || '';
  if (hash.startsWith('#/report')) {
    const parts = hash.split('/');
    const tabFromHash = (parts[2] || 'overview').toLowerCase();
    if (TABS.some((t) => t.id === tabFromHash) && tabFromHash !== activeTab) {
      switchTab(tabFromHash);
    }
  }
}

export async function render(container) {
  containerRef = container;
  abortController = new AbortController();

  // Read subtab from hash if present
  const hash = window.location.hash || '';
  if (hash.startsWith('#/report')) {
    const parts = hash.split('/');
    const requested = (parts[2] || 'overview').toLowerCase();
    if (TABS.some((t) => t.id === requested)) {
      activeTab = requested;
    }
  }

  container.innerHTML = `
    <div class="page-header">
      <h2 class="page-title">Model Report & Benchmarks</h2>
      <p class="page-description">Deep learning performance metrics, training histories, confusion matrices, adversarial robustness, and edge ONNX runtime evaluations.</p>
    </div>

    <!-- Sticky Subnav Bar -->
    <nav class="report-subnav" aria-label="Report Subsections">
      ${TABS.map(
        (t) => `
        <button class="report-tab-btn ${t.id === activeTab ? 'active' : ''}" data-tab="${t.id}" type="button">
          ${t.label}
        </button>
      `
      ).join('')}
    </nav>

    <!-- Content Container -->
    <div id="report-tab-content"></div>
  `;

  const contentEl = container.querySelector('#report-tab-content');
  contentEl.innerHTML = '';
  contentEl.appendChild(createSpinner('Loading comprehensive benchmark report...'));

  // Subnav Button Click Listeners
  container.querySelectorAll('.report-tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const tabId = btn.getAttribute('data-tab');
      switchTab(tabId);
    });
  });

  window.addEventListener('hashchange', handleHashChange);

  // Fetch report data
  try {
    reportData = await api.getReport(abortController.signal);
    switchTab(activeTab);
  } catch (err) {
    if (err.name === 'AbortError') return;
    contentEl.innerHTML = '';
    contentEl.appendChild(createErrorBanner(`Failed to load benchmark data: ${err.message}`));
  }
}

export function destroy() {
  if (abortController) {
    abortController.abort();
    abortController = null;
  }
  window.removeEventListener('hashchange', handleHashChange);
  destroyCharts();
  containerRef = null;
}
