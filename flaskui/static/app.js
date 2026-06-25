/* ═══════════════════════════════════════════════════════════════
   ResistAlert (AMR CDSS) — Frontend Logic
   ═══════════════════════════════════════════════════════════════ */

// ─── THEME ──────────────────────────────────────────────────────
const rootEl = document.getElementById('htmlRoot');
const themeBtn = document.getElementById('themeToggle');
let isDark = localStorage.getItem('dw-dark') !== 'false'; // default dark
if (!isDark) setLight(); else setDark();

themeBtn.addEventListener('click', () => {
  isDark = !isDark;
  if (isDark) { setDark(); localStorage.setItem('dw-dark', 'true'); }
  else { setLight(); localStorage.setItem('dw-dark', 'false'); }
  renderCharts();
});

function setDark() {
  rootEl.setAttribute('data-bs-theme', 'dark');
  Chart.defaults.color = '#94a3b8';
  themeBtn.innerHTML = '<i class="fa-solid fa-sun text-warning"></i>';
}
function setLight() {
  rootEl.setAttribute('data-bs-theme', 'light');
  Chart.defaults.color = '#475569';
  themeBtn.innerHTML = '<i class="fa-solid fa-moon"></i>';
}

// ─── FIELD SETTER (resolves by name=, handles select/checkbox/number) ──
function setField(name, val) {
  const el = document.querySelector(`[name="${name}"]`);
  if (!el || val === undefined || val === null) return;
  if (el.type === 'checkbox') {
    el.checked = (val === 1 || val === '1' || val === true || val === 'Yes' || val === 'on');
  } else {
    el.value = val;
  }
}

// ─── PRESETS ────────────────────────────────────────────────────
function loadPreset(presetId) {
  fetch(`/api/presets/${presetId}`)
    .then(r => r.json())
    .then(data => {
      for (const key in data) setField(key, data[key]);
    });
}

// ─── LOAD PATIENT FROM ROSTER ───────────────────────────────────
function loadPatient(p) {
  setField('patient_name', p.name);
  setField('patient_bed', p.bed);
  setField('patient_zone', p.zone || 'Bay A');
  setField('gender_select', p.gender || 'Male');
  if (p.values) {
    for (const k in p.values) setField(k, p.values[k]);
  }
  // Scroll to top on mobile
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ─── CHARTS ─────────────────────────────────────────────────────
let trendChart, shapChart;
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.weight = 500;

function renderCharts() {
  const dark = rootEl.getAttribute('data-bs-theme') === 'dark';
  const gridColor = dark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)';
  const textColor = dark ? '#94a3b8' : '#475569';

  // TREND CHART
  const probElem = document.getElementById('probData');
  let currentProb = 45;
  if (probElem) currentProb = parseFloat(probElem.textContent) * 100;

  const ctxTrend = document.getElementById('trendChart');
  if (ctxTrend) {
    let pts = [currentProb];
    let v = currentProb;
    for (let i = 0; i < 5; i++) {
      v = Math.max(5, Math.min(95, v + (Math.random() * 20 - 10)));
      pts.unshift(v);
    }
    if (trendChart) trendChart.destroy();
    trendChart = new Chart(ctxTrend, {
      type: 'line',
      data: {
        labels: ['48h', '36h', '24h', '12h', '6h', 'Now'],
        datasets: [{
          label: 'Risk %', data: pts,
          borderColor: '#6366f1',
          backgroundColor: dark ? 'rgba(99,102,241,0.15)' : 'rgba(99,102,241,0.1)',
          borderWidth: 2.5, fill: true, tension: 0.4,
          pointBackgroundColor: '#fff', pointBorderColor: '#6366f1',
          pointRadius: 4, pointHoverRadius: 7
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { color: gridColor }, ticks: { color: textColor, font: { size: 10 } } },
          y: { grid: { color: gridColor }, min: 0, max: 100, ticks: { color: textColor, font: { size: 10 }, callback: v => v + '%' } }
        }
      }
    });
  }

  // SHAP CHART
  const shapElem = document.getElementById('shapData');
  if (shapElem && shapElem.textContent.trim() !== "[]") {
    try {
      const sData = JSON.parse(shapElem.textContent);
      if (sData.length > 0) {
        const labels = sData.map(d => d.name
          .replace(/_last_value$/, '').replace(/_max$/, ' (max)').replace(/_min$/, ' (min)')
          .replace(/_/g, ' ').toUpperCase());
        const values = sData.map(d => d.value);
        const bgCols = values.map(v => v >= 0
          ? (dark ? 'rgba(239,68,68,0.7)' : 'rgba(239,68,68,0.8)')
          : (dark ? 'rgba(34,197,94,0.7)' : 'rgba(34,197,94,0.8)')
        );
        const ctxShap = document.getElementById('shapChart');
        if (ctxShap) {
          if (shapChart) shapChart.destroy();
          shapChart = new Chart(ctxShap, {
            type: 'bar',
            data: { labels, datasets: [{ data: values, backgroundColor: bgCols, borderRadius: 6 }] },
            options: {
              indexAxis: 'y', responsive: true, maintainAspectRatio: false,
              plugins: {
                legend: { display: false },
                tooltip: { callbacks: { label: ctx => `Shift: ${ctx.raw.toFixed(3)}` } }
              },
              scales: {
                x: { grid: { color: gridColor }, title: { display: true, text: 'Impact on Prediction', color: textColor, font: { size: 10 } } },
                y: { grid: { display: false }, ticks: { font: { size: 10, weight: 600 } } }
              }
            }
          });
        }
      }
    } catch (e) { /* silent */ }
  }
}

// ─── TELEMETRY ANIMATION ────────────────────────────────────────
const telCanvas = document.getElementById('telemetryCanvas');
if (telCanvas) {
  const tCtx = telCanvas.getContext('2d');
  const tW = telCanvas.offsetWidth || 900;
  const tH = telCanvas.offsetHeight || 100;
  telCanvas.width = tW * 2;
  telCanvas.height = tH * 2;
  tCtx.scale(2, 2);

  let offset = 0;
  const ecg = new Array(tW).fill(0);
  const eeg = new Array(tW).fill(0);

  function drawTelemetry() {
    tCtx.fillStyle = 'rgba(6,10,20,0.35)';
    tCtx.fillRect(0, 0, tW, tH);
    offset += 3;

    // ECG
    ecg.shift();
    if (Math.random() < 0.013) ecg.push(-25, 40, -12);
    else ecg.push((Math.random() - 0.5) * 2.5);

    // EEG
    eeg.shift();
    eeg.push(Math.sin(offset / 12) * 6 + Math.cos(offset / 20) * 4 + (Math.random() - 0.5) * 4);

    // Grid
    tCtx.strokeStyle = 'rgba(0,255,100,0.04)';
    tCtx.lineWidth = 0.5;
    tCtx.beginPath();
    for (let i = (offset * 2) % 40; i < tW; i += 40) { tCtx.moveTo(i, 0); tCtx.lineTo(i, tH); }
    tCtx.stroke();

    // ECG line
    tCtx.strokeStyle = '#00ff66';
    tCtx.lineWidth = 1.8;
    tCtx.beginPath();
    for (let i = 0; i < ecg.length; i++) tCtx.lineTo(i, tH / 3 - (ecg[i] || 0));
    tCtx.stroke();

    // EEG line
    tCtx.strokeStyle = '#06b6d4';
    tCtx.lineWidth = 1.2;
    tCtx.beginPath();
    for (let i = 0; i < eeg.length; i++) tCtx.lineTo(i, (tH / 3) * 2.2 - (eeg[i] || 0));
    tCtx.stroke();

    requestAnimationFrame(drawTelemetry);
  }
  drawTelemetry();
}

// ─── INIT ───────────────────────────────────────────────────────
renderCharts();
