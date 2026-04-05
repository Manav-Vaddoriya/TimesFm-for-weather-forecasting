/**
 * Weather Forecaster & Anomaly Detection — Main Application Script
 *
 * Handles:
 *  - City search & geocoding
 *  - Fetching predictions from the Flask backend
 *  - Rendering the Chart.js temperature chart
 *  - Rendering the anomaly probability grid
 *  - Loading / toast UI helpers
 */

// ── State ───────────────────────────────────────────────────────────
let chart = null;

// ── Event Listeners ─────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('cityInput').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') handleSearch();
    });
});

// ── Search & Predict ────────────────────────────────────────────────

async function handleSearch() {
    const city = document.getElementById('cityInput').value.trim();
    if (!city) return;

    const btn = document.getElementById('searchBtn');
    btn.disabled = true;
    showLoading(true);

    try {
        // 1. Geocode the city
        const geoResp = await fetch(`/geocode?city=${encodeURIComponent(city)}`);
        const geo = await geoResp.json();
        if (geo.error) throw new Error(geo.error);

        // Show location tag
        document.getElementById('locationText').textContent =
            `📍 ${geo.name}, ${geo.country} (${geo.latitude.toFixed(2)}°, ${geo.longitude.toFixed(2)}°)`;
        document.getElementById('locationTag').classList.add('visible');

        // 2. Run prediction + anomaly detection
        const predResp = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ latitude: geo.latitude, longitude: geo.longitude }),
        });
        const data = await predResp.json();
        if (data.error) throw new Error(data.error);

        // 3. Render all results
        renderResults(data);

    } catch (err) {
        showToast(err.message);
    } finally {
        btn.disabled = false;
        showLoading(false);
    }
}

// ── Render Results ──────────────────────────────────────────────────

function renderResults(data) {
    document.getElementById('emptyState').style.display = 'none';
    document.getElementById('results').style.display = 'block';

    // Current weather cards
    const c = data.current;
    document.getElementById('currentTemp').textContent =
        c.temperature != null ? c.temperature.toFixed(1) : '--';
    document.getElementById('currentHumidity').textContent =
        c.humidity != null ? Math.round(c.humidity) : '--';
    document.getElementById('currentWind').textContent =
        c.wind_speed != null ? c.wind_speed.toFixed(1) : '--';

    // Model info
    const m = data.model_info;
    document.getElementById('forecastParams').textContent = m.forecast_parameters;
    document.getElementById('anomalyParams').textContent = m.anomaly_parameters;
    document.getElementById('modelContext').textContent = m.context_window;
    document.getElementById('modelHorizon').textContent = m.forecast_horizon;

    // Chart (no anomaly markers)
    renderChart(data.history, data.prediction);

    // Anomaly probability grid
    renderAnomalyProbabilities(data.prediction, data.anomaly_probabilities);
}

// ── Chart ───────────────────────────────────────────────────────────

function renderChart(history, prediction) {
    const ctx = document.getElementById('forecastChart').getContext('2d');

    const formatTime = (iso) => {
        const d = new Date(iso);
        const month = d.toLocaleString('en', { month: 'short' });
        const day = d.getDate();
        const hour = d.getHours().toString().padStart(2, '0');
        return `${month} ${day}, ${hour}:00`;
    };

    const allLabels = [
        ...history.times.map(formatTime),
        ...prediction.times.map(formatTime),
    ];

    // Historical data (nulls where prediction sits)
    const histData = [
        ...history.temps,
        ...new Array(prediction.temps.length).fill(null),
    ];

    // Prediction data with a bridge point from the last historical value
    const predData = [
        ...new Array(history.temps.length - 1).fill(null),
        history.temps[history.temps.length - 1],
        ...prediction.temps,
    ];

    if (chart) chart.destroy();

    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: allLabels,
            datasets: [
                {
                    label: 'Historical',
                    data: histData,
                    borderColor: '#06b6d4',
                    backgroundColor: 'rgba(6, 182, 212, 0.08)',
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    fill: true,
                    tension: 0.3,
                    spanGaps: false,
                },
                {
                    label: 'AI Prediction',
                    data: predData,
                    borderColor: '#f97316',
                    backgroundColor: 'rgba(249, 115, 22, 0.08)',
                    borderWidth: 2.5,
                    borderDash: [6, 3],
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    fill: true,
                    tension: 0.3,
                    spanGaps: false,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(17, 24, 39, 0.95)',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1,
                    titleColor: '#f1f5f9',
                    bodyColor: '#94a3b8',
                    padding: 12,
                    cornerRadius: 10,
                    callbacks: {
                        label: (ctx) => {
                            if (ctx.raw == null) return null;
                            return `${ctx.dataset.label}: ${ctx.raw.toFixed(1)}°C`;
                        },
                    },
                },
            },
            scales: {
                x: {
                    ticks: {
                        color: '#64748b',
                        font: { size: 11 },
                        maxTicksLimit: 12,
                        maxRotation: 45,
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                },
                y: {
                    ticks: {
                        color: '#64748b',
                        font: { size: 11 },
                        callback: (v) => v.toFixed(0) + '°C',
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                },
            },
        },
    });
}

// ── Anomaly Probability Grid ────────────────────────────────────────

function renderAnomalyProbabilities(prediction, probabilities) {
    const grid = document.getElementById('anomalyGrid');
    grid.innerHTML = '';

    // Render each hourly cell with just its probability
    probabilities.forEach((prob, i) => {
        const cell = document.createElement('div');
        cell.className = 'anomaly-cell';

        const time = prediction.times[i];
        const d = new Date(time);
        const hourStr = d.getHours().toString().padStart(2, '0') + ':00';
        const dayStr = d.toLocaleString('en', { month: 'short' }) + ' ' + d.getDate();

        cell.innerHTML = `
            <div class="anomaly-hour">${dayStr}<br>${hourStr}</div>
            <div class="anomaly-temp">${prediction.temps[i].toFixed(1)}°</div>
            <div class="anomaly-prob">${(prob * 100).toFixed(1)}%</div>
        `;
        grid.appendChild(cell);
    });
}

// ── UI Helpers ──────────────────────────────────────────────────────

function showLoading(on) {
    document.getElementById('loading').classList.toggle('active', on);
}

function showToast(msg) {
    const el = document.getElementById('toast');
    el.textContent = '⚠ ' + msg;
    el.classList.add('visible');
    setTimeout(() => el.classList.remove('visible'), 4000);
}
