import { api } from '../api.js';
import {
  createPanel,
  createButton,
  createSegmentedControl,
  createImageInput,
  createTop5List,
  createSpinner,
  createErrorBanner,
  createEmptyState,
} from '../components/index.js';

let currentMode = 'upload'; // 'upload' | 'camera'
let currentFile = null;
let currentSampleMeta = null;
let currentModelKey = 'resnet18';
let cameraStream = null;
let cameraInterval = null;
let isPredictingCamera = false;
let abortController = null;

export async function render(container) {
  container.innerHTML = `
    <div class="page-header">
      <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
        <div>
          <h1 class="page-title">Recognise</h1>
          <p class="page-description">Inspect individual traffic sign predictions with top-5 confidence distributions and real-time Grad-CAM heatmaps.</p>
        </div>
        <div id="mode-switch-container"></div>
      </div>
    </div>

    <!-- Main Dynamic Area for Upload vs Camera -->
    <div id="recognise-content"></div>
  `;

  // Render Mode Switch (Upload image | Live camera)
  const modeContainer = document.getElementById('mode-switch-container');
  const modeSwitch = createSegmentedControl({
    name: 'recognise-mode',
    options: [
      { value: 'upload', label: 'Upload image' },
      { value: 'camera', label: 'Live camera' },
    ],
    value: currentMode,
    onChange: (val) => {
      currentMode = val;
      renderModeContent();
    },
    ariaLabel: 'Recognition input mode',
  });
  modeContainer.appendChild(modeSwitch);

  renderModeContent();
}

/**
 * Switch between Upload Mode and Live Camera Mode
 */
function renderModeContent() {
  const content = document.getElementById('recognise-content');
  if (!content) return;

  // Stop camera if leaving camera mode
  stopCamera();

  if (currentMode === 'upload') {
    renderUploadMode(content);
  } else {
    renderCameraMode(content);
  }
}

/**
 * Upload Mode: Image Input, Model Selector, Result Panel with Heatmap Slider
 */
async function renderUploadMode(container) {
  container.innerHTML = `
    <div class="recognise-layout">
      <!-- Left Column: Image Input & Model Picker -->
      <div style="display: flex; flex-direction: column; gap: 20px;">
        <div class="panel">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
            <div style="font-weight: 600; font-size: 14px; color: var(--ink);">Select neural network:</div>
            <div id="model-selector-container"></div>
          </div>
          <div id="image-input-container"></div>
        </div>
      </div>

      <!-- Right Column: Result Panel -->
      <div class="panel result-card" id="result-panel">
        <!-- Rendered dynamically -->
      </div>
    </div>
  `;

  const modelSelectorContainer = document.getElementById('model-selector-container');
  const imageInputContainer = document.getElementById('image-input-container');
  const resultPanel = document.getElementById('result-panel');

  // Load Models for Segmented Control
  try {
    const models = await api.getModels();
    const options = models.map((m) => ({ value: m.key, label: m.name }));
    const selector = createSegmentedControl({
      name: 'model-picker',
      options,
      value: currentModelKey,
      onChange: (val) => {
        currentModelKey = val;
        if (currentFile) runUploadPrediction();
      },
      ariaLabel: 'Select model architecture',
    });
    modelSelectorContainer.appendChild(selector);
  } catch (err) {
    console.error('Failed to load models:', err);
  }

  // Load Samples (Standard + Real-World)
  let standardSamples = [];
  let realSamples = [];
  try {
    const allSamples = await api.getSamples();
    standardSamples = allSamples.filter((s) => !s.real);
    realSamples = allSamples.filter((s) => s.real);
  } catch (err) {
    console.error('Failed to load samples:', err);
  }

  // Image Input Component
  const imageInput = createImageInput({
    onImageSelected: (file, sampleMeta) => {
      currentFile = file;
      currentSampleMeta = sampleMeta;
      runUploadPrediction();
    },
    samples: standardSamples,
  });
  imageInputContainer.appendChild(imageInput);

  // Real-world sample strip sub-heading
  if (realSamples.length > 0) {
    const realBox = document.createElement('div');
    realBox.className = 'sample-strip-container';
    realBox.style.marginTop = '12px';

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
          runUploadPrediction();
        } catch (e) {
          console.error(e);
        }
      });

      realStrip.appendChild(img);
    });

    realBox.appendChild(realStrip);
    imageInputContainer.appendChild(realBox);
  }

  // Initial Empty State
  renderUploadEmptyState(resultPanel);
}

function renderUploadEmptyState(panel) {
  panel.innerHTML = '';
  panel.appendChild(
    createEmptyState({
      title: 'No sign selected yet',
      subtitle: 'Drop a traffic sign photo above, or pick one of the sample test signs.',
      icon: '🚦',
    })
  );
}

/**
 * Execute Prediction for Upload Mode
 */
async function runUploadPrediction() {
  const resultPanel = document.getElementById('result-panel');
  if (!resultPanel || !currentFile) return;

  if (abortController) abortController.abort();
  abortController = new AbortController();

  resultPanel.innerHTML = '';
  const loading = document.createElement('div');
  loading.style.display = 'flex';
  loading.style.flexDirection = 'column';
  loading.style.alignItems = 'center';
  loading.style.justifyContent = 'center';
  loading.style.padding = '48px 0';
  loading.style.gap = '12px';
  loading.appendChild(createSpinner());
  const loadText = document.createElement('div');
  loadText.style.color = 'var(--muted)';
  loadText.textContent = 'Classifying sign and generating Grad-CAM...';
  loading.appendChild(loadText);
  resultPanel.appendChild(loading);

  try {
    const res = await api.predict(currentFile, currentModelKey, abortController.signal);
    renderPredictionResult(resultPanel, res);
  } catch (err) {
    if (err.name === 'AbortError') return;
    resultPanel.innerHTML = '';
    resultPanel.appendChild(createErrorBanner(err.message || 'Classification failed.'));
  }
}

/**
 * Render Successful Prediction & Grad-CAM Heatmap Viewer
 */
function renderPredictionResult(panel, pred) {
  panel.innerHTML = '';

  // 1. Prediction Header & Big Class Name
  const header = document.createElement('div');
  header.className = 'prediction-header';

  const title = document.createElement('h2');
  title.className = 'prediction-class-name';
  title.textContent = pred.class_name;

  const badgeRow = document.createElement('div');
  badgeRow.className = 'confidence-badge-row';

  // Confidence Pill
  const pct = (pred.confidence * 100).toFixed(1);
  const confBadge = document.createElement('span');
  if (pred.confidence >= 0.90) {
    confBadge.className = 'badge badge-green';
    confBadge.textContent = `${pct}% confidence (High)`;
  } else if (pred.confidence >= 0.60) {
    confBadge.className = 'badge badge-amber';
    confBadge.textContent = `${pct}% confidence (Medium)`;
  } else {
    confBadge.className = 'badge badge-amber';
    confBadge.textContent = `${pct}% confidence (Low: check manually)`;
  }
  badgeRow.appendChild(confBadge);

  // Ground-truth check if from known test sample
  if (currentSampleMeta && currentSampleMeta.class_id !== null && currentSampleMeta.class_id !== undefined) {
    const truthBadge = document.createElement('span');
    if (pred.class_id === currentSampleMeta.class_id) {
      truthBadge.className = 'badge badge-green';
      truthBadge.textContent = '✓ Correct';
    } else {
      truthBadge.className = 'badge badge-red';
      truthBadge.textContent = `✗ Wrong, expected ${currentSampleMeta.class_name || 'Class ' + currentSampleMeta.class_id}`;
    }
    badgeRow.appendChild(truthBadge);
  }

  // Model & Latency
  const latencyBadge = document.createElement('span');
  latencyBadge.style.fontSize = '13px';
  latencyBadge.style.color = 'var(--muted)';
  latencyBadge.textContent = `${pred.latency_ms} ms on CPU (${pred.model})`;
  badgeRow.appendChild(latencyBadge);

  header.appendChild(title);
  header.appendChild(badgeRow);
  panel.appendChild(header);

  // 2. Heatmap Image Viewer with Opacity Slider
  const camViewerContainer = document.createElement('div');
  camViewerContainer.className = 'cam-viewer-container';

  const viewer = document.createElement('div');
  viewer.className = 'cam-viewer';

  const baseImg = document.createElement('img');
  baseImg.className = 'cam-img-base';
  baseImg.src = pred.input_b64;
  baseImg.alt = 'Input 64x64 sign';

  const heatImg = document.createElement('img');
  heatImg.className = 'cam-img-heatmap';
  heatImg.src = pred.heatmap_b64;
  heatImg.alt = 'Grad-CAM attention heatmap';
  heatImg.style.opacity = '0.5';

  viewer.appendChild(baseImg);
  viewer.appendChild(heatImg);

  // Controls for heatmap opacity
  const camControls = document.createElement('div');
  camControls.className = 'cam-controls';

  const sliderRow = document.createElement('div');
  sliderRow.className = 'slider-row';

  const sliderLabel = document.createElement('label');
  sliderLabel.textContent = 'Heatmap strength:';
  sliderLabel.setAttribute('for', 'opacity-slider');

  const sliderValue = document.createElement('span');
  sliderValue.textContent = '50%';
  sliderValue.style.fontWeight = '600';

  const slider = document.createElement('input');
  slider.type = 'range';
  slider.id = 'opacity-slider';
  slider.className = 'slider-input';
  slider.min = '0';
  slider.max = '100';
  slider.value = '50';

  let currentOpacity = 0.5;
  slider.addEventListener('input', (e) => {
    currentOpacity = parseInt(e.target.value, 10) / 100;
    heatImg.style.opacity = currentOpacity.toString();
    sliderValue.textContent = `${e.target.value}%`;
  });

  sliderRow.appendChild(sliderLabel);
  sliderRow.appendChild(slider);
  sliderRow.appendChild(sliderValue);

  // Toggle Heatmap / Original
  const toggleRow = document.createElement('div');
  toggleRow.style.display = 'flex';
  toggleRow.style.justifyContent = 'center';
  toggleRow.style.gap = '8px';

  const toggleBtn = createButton({
    label: 'Show original only',
    variant: 'secondary',
    onClick: () => {
      if (heatImg.style.opacity === '0') {
        heatImg.style.opacity = currentOpacity.toString();
        toggleBtn.textContent = 'Show original only';
      } else {
        heatImg.style.opacity = '0';
        toggleBtn.textContent = 'Show heatmap';
      }
    },
  });
  toggleRow.appendChild(toggleBtn);

  const caption = document.createElement('div');
  caption.className = 'cam-caption';
  caption.textContent = 'Warm colours show the pixels that drove this prediction.';

  camControls.appendChild(sliderRow);
  camControls.appendChild(toggleRow);
  camControls.appendChild(caption);

  camViewerContainer.appendChild(viewer);
  camViewerContainer.appendChild(camControls);
  panel.appendChild(camViewerContainer);

  // 3. Top-5 Predictions
  const top5Heading = document.createElement('h4');
  top5Heading.style.fontSize = '16px';
  top5Heading.style.marginTop = '8px';
  top5Heading.textContent = 'Top-5 class probabilities:';
  panel.appendChild(top5Heading);

  const top5El = createTop5List(pred.top5);
  panel.appendChild(top5El);
}

/**
 * Live Camera Mode: getUserMedia Video + Center Guide Box + 500ms Interval Prediction
 */
function renderCameraMode(container) {
  container.innerHTML = `
    <div class="recognise-layout">
      <!-- Left Column: Video Feed & Guide Box -->
      <div class="panel camera-view-box">
        <div style="font-weight: 600; font-size: 14px; margin-bottom: 8px;">Dashboard camera stream</div>
        <div class="camera-container" id="camera-box">
          <video class="camera-video" id="camera-video" autoplay playsinline muted></video>
          <div class="camera-guide-box">
            <div class="camera-guide-label">Hold sign inside box</div>
          </div>
          <div class="camera-prediction-badge" id="camera-badge" style="display: none;">
            <span id="camera-pred-class">Detecting...</span>
            <span id="camera-pred-conf">0%</span>
          </div>
        </div>

        <div style="display: flex; gap: 12px; margin-top: 14px;">
          <button id="camera-start-btn" class="btn btn-primary">Start camera</button>
          <button id="camera-stop-btn" class="btn btn-secondary" disabled>Stop camera</button>
        </div>
        <div id="camera-error" style="width: 100%; margin-top: 12px;"></div>
      </div>

      <!-- Right Column: Live Top-5 Probabilities -->
      <div class="panel" id="camera-results-panel">
        <h3 class="panel-title" style="margin-bottom: 12px;">Live class distribution</h3>
        <p style="font-size: 14px; color: var(--muted); margin-bottom: 16px;">
          The camera box is cropped every 500 ms and classified in real time on CPU.
        </p>
        <div id="camera-top5-container">
          <!-- Live top-5 bars -->
        </div>
      </div>
    </div>
  `;

  const video = document.getElementById('camera-video');
  const startBtn = document.getElementById('camera-start-btn');
  const stopBtn = document.getElementById('camera-stop-btn');
  const errorBox = document.getElementById('camera-error');
  const badge = document.getElementById('camera-badge');
  const predClass = document.getElementById('camera-pred-class');
  const predConf = document.getElementById('camera-pred-conf');
  const top5Container = document.getElementById('camera-top5-container');

  startBtn.addEventListener('click', async () => {
    errorBox.innerHTML = '';
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } },
      });
      video.srcObject = cameraStream;
      startBtn.disabled = true;
      stopBtn.disabled = false;
      badge.style.display = 'flex';

      // Offscreen canvas for cropping
      const canvas = document.createElement('canvas');
      canvas.width = 128;
      canvas.height = 128;
      const ctx = canvas.getContext('2d');

      cameraInterval = setInterval(() => {
        if (!cameraStream || isPredictingCamera || video.readyState !== 4) return;
        isPredictingCamera = true;

        const vw = video.videoWidth;
        const vh = video.videoHeight;
        const size = Math.min(vw, vh) * 0.45;
        const sx = (vw - size) / 2;
        const sy = (vh - size) / 2;

        ctx.drawImage(video, sx, sy, size, size, 0, 0, 128, 128);

        canvas.toBlob(async (blob) => {
          if (!blob) {
            isPredictingCamera = false;
            return;
          }
          const file = new File([blob], 'camera_frame.png', { type: 'image/png' });
          try {
            const pred = await api.predict(file, currentModelKey);
            predClass.textContent = pred.class_name;
            predConf.textContent = `${(pred.confidence * 100).toFixed(1)}%`;

            if (top5Container) {
              top5Container.innerHTML = '';
              top5Container.appendChild(createTop5List(pred.top5));
            }
          } catch (e) {
            console.warn('Camera prediction error:', e);
          } finally {
            isPredictingCamera = false;
          }
        }, 'image/png');
      }, 500);
    } catch (err) {
      console.error('Camera access error:', err);
      let msg = 'Could not access camera.';
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        msg = 'Camera permission denied. Please allow camera access in your browser settings.';
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        msg = 'No camera device found on this system.';
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        msg = 'Camera is already in use by another application (e.g., Zoom/Teams).';
      }
      errorBox.innerHTML = '';
      errorBox.appendChild(createErrorBanner(msg));
    }
  });

  stopBtn.addEventListener('click', stopCamera);
}

function stopCamera() {
  if (cameraInterval) {
    clearInterval(cameraInterval);
    cameraInterval = null;
  }
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }
  const startBtn = document.getElementById('camera-start-btn');
  const stopBtn = document.getElementById('camera-stop-btn');
  const badge = document.getElementById('camera-badge');
  if (startBtn) startBtn.disabled = false;
  if (stopBtn) stopBtn.disabled = true;
  if (badge) badge.style.display = 'none';
}

export function destroy() {
  stopCamera();
  if (abortController) {
    abortController.abort();
    abortController = null;
  }
}
