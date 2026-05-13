
// ── Navigation ──────────────────────────────────────────────
const navItems = document.querySelectorAll('.nav-item');
const pages    = document.querySelectorAll('.page');

function showPage(id) {
  pages.forEach(p => p.classList.remove('active'));
  navItems.forEach(n => n.classList.remove('active'));
  const page = document.getElementById(id);
  const nav  = document.querySelector(`[data-page="${id}"]`);
  if (page) page.classList.add('active');
  if (nav)  nav.classList.add('active');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

navItems.forEach(item => {
  item.addEventListener('click', () => showPage(item.dataset.page));
});

// ── Tabs ──────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const group = btn.dataset.group;
    document.querySelectorAll(`.tab-btn[data-group="${group}"]`)
      .forEach(b => b.classList.remove('active'));
    document.querySelectorAll(`.tab-panel[data-group="${group}"]`)
      .forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    const panel = document.getElementById(btn.dataset.tab);
    if (panel) panel.classList.add('active');
  });
});

// ── Distance Calculator ──────────────────────────────────────
function calcDistances() {
  const p1 = [+document.getElementById('p1x').value, +document.getElementById('p1y').value];
  const p2 = [+document.getElementById('p2x').value, +document.getElementById('p2y').value];

  const euclidean = Math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2);
  const manhattan = Math.abs(p2[0]-p1[0]) + Math.abs(p2[1]-p1[1]);

  const dot  = p1[0]*p2[0] + p1[1]*p2[1];
  const mag1 = Math.sqrt(p1[0]**2 + p1[1]**2);
  const mag2 = Math.sqrt(p2[0]**2 + p2[1]**2);
  const cosine = (mag1 && mag2) ? dot/(mag1*mag2) : 0;

  document.getElementById('res-euclidean').textContent = euclidean.toFixed(3);
  document.getElementById('res-manhattan').textContent = manhattan.toFixed(3);
  document.getElementById('res-cosine').textContent    = cosine.toFixed(3);
  drawDistanceViz(p1, p2);
}

function drawDistanceViz(p1, p2) {
  const canvas = document.getElementById('distCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 1;
  for (let x = 0; x < W; x += 40) { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke(); }
  for (let y = 0; y < H; y += 40) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke(); }

  const scale = 30, ox = W/2, oy = H/2;
  const sx1 = ox + p1[0]*scale, sy1 = oy - p1[1]*scale;
  const sx2 = ox + p2[0]*scale, sy2 = oy - p2[1]*scale;

  // Axes
  ctx.strokeStyle = 'rgba(255,255,255,0.2)'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, oy); ctx.lineTo(W, oy); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(ox, 0); ctx.lineTo(ox, H); ctx.stroke();

  // Euclidean line
  ctx.strokeStyle = '#6366f1'; ctx.lineWidth = 2;
  ctx.setLineDash([]);
  ctx.beginPath(); ctx.moveTo(sx1, sy1); ctx.lineTo(sx2, sy2); ctx.stroke();

  // Manhattan path
  ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 2;
  ctx.setLineDash([6,3]);
  ctx.beginPath(); ctx.moveTo(sx1, sy1); ctx.lineTo(sx2, sy1); ctx.lineTo(sx2, sy2); ctx.stroke();
  ctx.setLineDash([]);

  // Points
  [[sx1,sy1,'#10b981','P1'],[sx2,sy2,'#f43f5e','P2']].forEach(([x,y,c,lbl]) => {
    ctx.beginPath(); ctx.arc(x, y, 8, 0, Math.PI*2);
    ctx.fillStyle = c; ctx.fill();
    ctx.strokeStyle = 'white'; ctx.lineWidth = 2; ctx.stroke();
    ctx.fillStyle = 'white'; ctx.font = 'bold 12px Inter';
    ctx.fillText(lbl, x+10, y-8);
  });
}

// ── Association Rule Calculator ──────────────────────────────
function calcRule() {
  const total  = +document.getElementById('r-total').value  || 1000;
  const antAB  = +document.getElementById('r-AB').value     || 200;
  const antA   = +document.getElementById('r-A').value      || 300;
  const antB   = +document.getElementById('r-B').value      || 400;

  const support    = antAB / total;
  const confidence = antAB / antA;
  const lift       = confidence / (antB / total);
  const conviction = (1 - antB/total) / (1 - confidence + 0.0001);

  document.getElementById('r-support').textContent    = support.toFixed(3);
  document.getElementById('r-confidence').textContent = confidence.toFixed(3);
  document.getElementById('r-lift').textContent       = lift.toFixed(2);
  document.getElementById('r-conviction').textContent = conviction.toFixed(2);

  const q = document.getElementById('lift-quality');
  if (lift > 1.5) { q.textContent = '🟢 Strong positive association'; q.style.color = '#10b981'; }
  else if (lift > 1) { q.textContent = '🟡 Mild positive association'; q.style.color = '#f59e0b'; }
  else { q.textContent = '🔴 Negative / no association'; q.style.color = '#f43f5e'; }
}

// ── Silhouette / Metric Helpers ───────────────────────────────
function updateMetricExplain() {
  const sel = document.getElementById('metric-select')?.value;
  const el  = document.getElementById('metric-explain');
  if (!el) return;
  const texts = {
    silhouette: '<strong>Silhouette Score</strong>: Measures how similar a point is to its own cluster vs other clusters. Range [-1,1]. Higher is better.',
    davies: '<strong>Davies–Bouldin Index</strong>: Ratio of within-cluster scatter to between-cluster separation. Lower is better.',
    inertia: '<strong>Inertia (WCSS)</strong>: Sum of squared distances from each point to its cluster centroid. Lower = more compact clusters.',
    calinski: '<strong>Calinski–Harabász</strong>: Ratio of between-cluster to within-cluster dispersion. Higher is better.'
  };
  el.innerHTML = texts[sel] || '';
}

// ── Init ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  showPage('home');
  calcDistances();
  calcRule();
  updateMetricExplain();

  document.getElementById('metric-select')?.addEventListener('change', updateMetricExplain);
  document.getElementById('btn-calc-dist')?.addEventListener('click', calcDistances);
  document.getElementById('btn-calc-rule')?.addEventListener('click', calcRule);
});
