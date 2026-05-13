
// ═══════════════════════════════════════════════════════════════
//  viz.js — All Interactive Visualizations
// ═══════════════════════════════════════════════════════════════

const COLORS = ['#6366f1','#10b981','#f59e0b','#f43f5e','#06b6d4','#8b5cf6','#ec4899'];

// ── Utility ────────────────────────────────────────────────────
function randRange(a, b) { return a + Math.random() * (b - a); }
function dist2(a, b) { return (a[0]-b[0])**2 + (a[1]-b[1])**2; }
function drawCircle(ctx, x, y, r, fill, stroke, lw=2) {
  ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2);
  ctx.fillStyle = fill; ctx.fill();
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke(); }
}

// ══════════════════════════════════════════════════════════════
//  1. K-MEANS VISUALIZATION
// ══════════════════════════════════════════════════════════════
const KMeans = (() => {
  let points = [], centroids = [], assignments = [], k = 3, running = false, step = 0;

  function generateData(n=120) {
    points = [];
    const blobs = k;
    for (let b = 0; b < blobs; b++) {
      const cx = randRange(80,320), cy = randRange(60,240);
      const count = Math.floor(n / blobs);
      for (let i = 0; i < count; i++) {
        points.push([cx + randRange(-40,40), cy + randRange(-40,40)]);
      }
    }
  }

  function initCentroids(mode='kmeans++') {
    centroids = [];
    if (mode === 'random') {
      for (let i = 0; i < k; i++) {
        centroids.push([...points[Math.floor(Math.random()*points.length)]]);
      }
    } else {
      // k-means++
      centroids.push([...points[Math.floor(Math.random()*points.length)]]);
      while (centroids.length < k) {
        const dists = points.map(p => Math.min(...centroids.map(c => dist2(p,c))));
        const sum = dists.reduce((a,b)=>a+b,0);
        let r = Math.random()*sum, cum = 0;
        for (let i = 0; i < points.length; i++) {
          cum += dists[i];
          if (cum >= r) { centroids.push([...points[i]]); break; }
        }
      }
    }
  }

  function assign() {
    assignments = points.map(p => {
      let best = 0, bestD = Infinity;
      centroids.forEach((c,i) => { const d = dist2(p,c); if(d<bestD){bestD=d;best=i;} });
      return best;
    });
  }

  function update() {
    const sums = Array.from({length:k}, ()=>[0,0]);
    const counts = new Array(k).fill(0);
    points.forEach((p,i) => {
      sums[assignments[i]][0] += p[0];
      sums[assignments[i]][1] += p[1];
      counts[assignments[i]]++;
    });
    let moved = false;
    centroids = centroids.map((c,i) => {
      if (!counts[i]) return c;
      const nc = [sums[i][0]/counts[i], sums[i][1]/counts[i]];
      if (dist2(c,nc) > 0.01) moved = true;
      return nc;
    });
    return moved;
  }

  function computeWCSS() {
    return points.reduce((sum,p,i) => sum + dist2(p, centroids[assignments[i]]), 0);
  }

  function draw(canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0,0,W,H);

    // Background grid
    ctx.strokeStyle = 'rgba(255,255,255,0.04)'; ctx.lineWidth = 1;
    for(let x=0;x<W;x+=30){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}
    for(let y=0;y<H;y+=30){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}

    // Voronoi-like background tint
    const idata = ctx.createImageData(W,H);
    for (let py=0;py<H;py+=4){
      for (let px=0;px<W;px+=4){
        let best=0, bd=Infinity;
        centroids.forEach((c,i)=>{ const d=(px-c[0])**2+(py-c[1])**2; if(d<bd){bd=d;best=i;} });
        const col = hexToRgb(COLORS[best % COLORS.length]);
        for (let dy=0;dy<4&&py+dy<H;dy++){
          for (let dx=0;dx<4&&px+dx<W;dx++){
            const idx = ((py+dy)*W+(px+dx))*4;
            idata.data[idx]=col[0]; idata.data[idx+1]=col[1];
            idata.data[idx+2]=col[2]; idata.data[idx+3]=25;
          }
        }
      }
    }
    ctx.putImageData(idata,0,0);

    // Points
    points.forEach((p,i) => {
      const c = assignments.length ? COLORS[assignments[i] % COLORS.length] : '#475569';
      drawCircle(ctx, p[0], p[1], 4, c, 'rgba(255,255,255,0.3)', 1);
    });

    // Centroids
    centroids.forEach((c,i) => {
      drawCircle(ctx, c[0], c[1], 12, COLORS[i % COLORS.length], 'white', 3);
      ctx.fillStyle = 'white'; ctx.font = 'bold 11px Inter';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('✕', c[0], c[1]);
    });

    // WCSS display
    if (assignments.length) {
      const wcss = computeWCSS().toFixed(0);
      document.getElementById('wcss-val') && (document.getElementById('wcss-val').textContent = wcss);
      document.getElementById('step-val') && (document.getElementById('step-val').textContent = step);
    }
  }

  function hexToRgb(hex) {
    const r = parseInt(hex.slice(1,3),16);
    const g = parseInt(hex.slice(3,5),16);
    const b = parseInt(hex.slice(5,7),16);
    return [r,g,b];
  }

  return {
    init(canvasId) {
      k = +document.getElementById('km-k')?.value || 3;
      step = 0;
      generateData(150);
      initCentroids(document.getElementById('km-init')?.value || 'kmeans++');
      assign();
      draw(canvasId);
    },
    stepOnce(canvasId) {
      if (!points.length) this.init(canvasId);
      assign(); update(); step++;
      draw(canvasId);
    },
    run(canvasId) {
      if (running) { running = false; return; }
      running = true;
      const loop = () => {
        if (!running) return;
        assign();
        const moved = update(); step++;
        draw(canvasId);
        if (moved && step < 50) setTimeout(loop, 300);
        else running = false;
      };
      loop();
    },
    stop() { running = false; }
  };
})();

// ══════════════════════════════════════════════════════════════
//  2. ELBOW METHOD CHART
// ══════════════════════════════════════════════════════════════
function drawElbow(canvasId) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0,0,W,H);

  const ks     = [1,2,3,4,5,6,7,8];
  const wcss   = [8200,4100,1800,1550,1400,1300,1230,1190];
  const padL=60, padR=20, padT=20, padB=40;
  const gW = W-padL-padR, gH = H-padT-padB;
  const maxY = Math.max(...wcss);

  // Axes
  ctx.strokeStyle = '#475569'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(padL,padT); ctx.lineTo(padL,padT+gH); ctx.lineTo(padL+gW,padT+gH); ctx.stroke();

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  for (let i=0;i<=4;i++) {
    const y = padT + gH - (i/4)*gH;
    ctx.beginPath(); ctx.moveTo(padL,y); ctx.lineTo(padL+gW,y); ctx.stroke();
    ctx.fillStyle = '#475569'; ctx.font = '10px Inter'; ctx.textAlign = 'right';
    ctx.fillText(Math.round(maxY*i/4), padL-6, y+4);
  }

  // Area fill
  const pts = ks.map((kv,i) => [padL+(i/(ks.length-1))*gW, padT+gH-(wcss[i]/maxY)*gH]);
  ctx.beginPath(); ctx.moveTo(pts[0][0], padT+gH);
  pts.forEach(p => ctx.lineTo(p[0],p[1]));
  ctx.lineTo(pts[pts.length-1][0], padT+gH); ctx.closePath();
  const grad = ctx.createLinearGradient(0,padT,0,padT+gH);
  grad.addColorStop(0,'rgba(99,102,241,0.4)'); grad.addColorStop(1,'rgba(99,102,241,0.02)');
  ctx.fillStyle = grad; ctx.fill();

  // Line
  ctx.strokeStyle = '#6366f1'; ctx.lineWidth = 2.5; ctx.setLineDash([]);
  ctx.beginPath(); pts.forEach((p,i) => i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1])); ctx.stroke();

  // Elbow marker (k=3)
  const elbow = pts[2];
  ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 1.5; ctx.setLineDash([5,3]);
  ctx.beginPath(); ctx.moveTo(elbow[0], padT); ctx.lineTo(elbow[0], padT+gH); ctx.stroke();
  ctx.setLineDash([]);
  drawCircle(ctx, elbow[0], elbow[1], 8, '#f59e0b', 'white', 2);
  ctx.fillStyle = '#f59e0b'; ctx.font = 'bold 11px Inter'; ctx.textAlign = 'center';
  ctx.fillText('Elbow (k=3)', elbow[0], padT+gH+20);

  // X labels
  ks.forEach((kv,i) => {
    ctx.fillStyle = '#94a3b8'; ctx.font = '11px Inter'; ctx.textAlign = 'center';
    ctx.fillText('k='+kv, pts[i][0], padT+gH+20);
  });

  // Points
  pts.forEach(p => drawCircle(ctx, p[0], p[1], 5, '#6366f1', 'white', 1.5));
}

// ══════════════════════════════════════════════════════════════
//  3. DBSCAN VISUALIZATION
// ══════════════════════════════════════════════════════════════
const DBSCAN = (() => {
  let points = [], labels = [];

  function generateData() {
    points = [];
    // Cluster 1
    for(let i=0;i<40;i++) points.push([randRange(60,140), randRange(60,130)]);
    // Cluster 2
    for(let i=0;i<35;i++) points.push([randRange(220,300), randRange(80,140)]);
    // Cluster 3 (ring-like)
    for(let i=0;i<30;i++) {
      const a=Math.random()*Math.PI*2, r=randRange(45,65);
      points.push([200+r*Math.cos(a), 220+r*Math.sin(a)*0.6]);
    }
    // Noise
    for(let i=0;i<12;i++) points.push([randRange(20,370), randRange(20,280)]);
  }

  function run(eps, minPts) {
    labels = new Array(points.length).fill(-2); // -2 = unvisited
    let clusterIdx = -1;

    function neighborhood(idx) {
      return points.reduce((acc,p,i) => {
        if (Math.sqrt(dist2(points[idx],p)) <= eps) acc.push(i);
        return acc;
      }, []);
    }

    for (let i=0;i<points.length;i++) {
      if (labels[i] !== -2) continue;
      const nb = neighborhood(i);
      if (nb.length < minPts) { labels[i] = -1; continue; } // noise
      clusterIdx++;
      labels[i] = clusterIdx;
      const seed = [...nb];
      for (let si=0;si<seed.length;si++) {
        const q = seed[si];
        if (labels[q] === -1) labels[q] = clusterIdx;
        if (labels[q] !== -2) continue;
        labels[q] = clusterIdx;
        const qnb = neighborhood(q);
        if (qnb.length >= minPts) seed.push(...qnb.filter(x=>!seed.includes(x)));
      }
    }
  }

  function draw(canvasId, eps) {
    const canvas = document.getElementById(canvasId);
    if(!canvas) return;
    const ctx = canvas.getContext('2d');
    const W=canvas.width, H=canvas.height;
    ctx.clearRect(0,0,W,H);

    // Grid
    ctx.strokeStyle='rgba(255,255,255,0.04)'; ctx.lineWidth=1;
    for(let x=0;x<W;x+=30){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}
    for(let y=0;y<H;y+=30){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}

    const maxCluster = Math.max(...labels);

    points.forEach((p,i) => {
      const lbl = labels[i];
      if (lbl === -1) {
        drawCircle(ctx, p[0], p[1], 4, '#374151', '#6b7280', 1);
        ctx.fillStyle='#9ca3af'; ctx.font='9px Inter'; ctx.textAlign='center';
        ctx.fillText('×', p[0], p[1]+3);
      } else {
        const col = COLORS[lbl % COLORS.length];
        drawCircle(ctx, p[0], p[1], 5, col, 'rgba(255,255,255,0.3)', 1);
      }
    });

    // Epsilon circle on hover (first core point)
    const coreIdx = labels.findIndex(l=>l>=0);
    if (coreIdx>=0 && eps) {
      ctx.strokeStyle='rgba(99,102,241,0.3)'; ctx.lineWidth=1.5; ctx.setLineDash([4,3]);
      ctx.beginPath(); ctx.arc(points[coreIdx][0], points[coreIdx][1], eps, 0, Math.PI*2); ctx.stroke();
      ctx.setLineDash([]);
    }

    // Stats
    const noise = labels.filter(l=>l===-1).length;
    document.getElementById('dbscan-clusters') && (document.getElementById('dbscan-clusters').textContent = maxCluster+1);
    document.getElementById('dbscan-noise') && (document.getElementById('dbscan-noise').textContent = noise);
  }

  return {
    init(canvasId) { generateData(); },
    run(canvasId) {
      const eps    = +document.getElementById('db-eps')?.value || 35;
      const minPts = +document.getElementById('db-minpts')?.value || 5;
      if(!points.length) generateData();
      run(eps, minPts);
      draw(canvasId, eps);
    },
    reset(canvasId) { generateData(); labels=[]; draw(canvasId, 0); }
  };
})();

// ══════════════════════════════════════════════════════════════
//  4. PCA vs t-SNE PROJECT DASHBOARD
// ══════════════════════════════════════════════════════════════
const ProjectViz = (() => {
  let dataset = 'digits';

  // Generate synthetic 2D projections mimicking PCA & t-SNE
  function generateProjections(ds) {
    const numClasses = ds === 'digits' ? 10 : 5;
    const n = 200;
    const pca = [], tsne = [];
    const labels = [];

    for (let c=0; c<numClasses; c++) {
      const count = Math.floor(n/numClasses);
      // PCA: clusters overlap more (linear projection)
      const pcaCx = (c % 5)*70 - 140 + randRange(-20,20);
      const pcaCy = Math.floor(c/5)*80 - 40 + randRange(-20,20);
      // t-SNE: well-separated clusters
      const angle = (c/numClasses)*Math.PI*2;
      const rad = 130;
      const tCx = rad*Math.cos(angle);
      const tCy = rad*Math.sin(angle)*0.85;

      for (let i=0;i<count;i++) {
        pca.push([pcaCx+randRange(-30,30), pcaCy+randRange(-25,25)]);
        tsne.push([tCx+randRange(-22,22), tCy+randRange(-22,22)]);
        labels.push(c);
      }
    }
    return { pca, tsne, labels };
  }

  function drawProjection(canvasId, pts, labels, title) {
    const canvas = document.getElementById(canvasId);
    if(!canvas) return;
    const ctx = canvas.getContext('2d');
    const W=canvas.width, H=canvas.height;
    ctx.clearRect(0,0,W,H);

    // Background
    ctx.fillStyle='#111827'; ctx.fillRect(0,0,W,H);

    // Grid
    ctx.strokeStyle='rgba(255,255,255,0.04)'; ctx.lineWidth=1;
    for(let x=0;x<W;x+=30){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}
    for(let y=0;y<H;y+=30){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}

    // Find bounds
    const xs=pts.map(p=>p[0]), ys=pts.map(p=>p[1]);
    const xMin=Math.min(...xs),xMax=Math.max(...xs);
    const yMin=Math.min(...ys),yMax=Math.max(...ys);
    const pad=30;

    function toScreen(x,y) {
      return [
        pad + (x-xMin)/(xMax-xMin+0.01)*(W-2*pad),
        pad + (y-yMin)/(yMax-yMin+0.01)*(H-2*pad)
      ];
    }

    pts.forEach((p,i) => {
      const [sx,sy] = toScreen(p[0],p[1]);
      const col = COLORS[labels[i] % COLORS.length];
      drawCircle(ctx,sx,sy,4,col,'rgba(255,255,255,0.2)',1);
    });

    // Title
    ctx.fillStyle='rgba(241,245,249,0.9)'; ctx.font='bold 13px Inter';
    ctx.textAlign='center'; ctx.fillText(title, W/2, H-10);
  }

  return {
    setDataset(ds) { dataset = ds; },
    draw() {
      const data = generateProjections(dataset);
      drawProjection('pca-canvas', data.pca, data.labels, 'PCA Projection');
      drawProjection('tsne-canvas', data.tsne, data.labels, 't-SNE Projection');
      updateStats(data);
    }
  };

  function updateStats(data) {
    // Simulated metrics
    const pcaSep  = (Math.random()*0.2+0.35).toFixed(2);
    const tsneSep = (Math.random()*0.1+0.82).toFixed(2);
    document.getElementById('pca-sil') && (document.getElementById('pca-sil').textContent = pcaSep);
    document.getElementById('tsne-sil') && (document.getElementById('tsne-sil').textContent = tsneSep);
  }
})();

// ══════════════════════════════════════════════════════════════
//  5. ISOLATION FOREST VIZ
// ══════════════════════════════════════════════════════════════
function drawIsolationForest(canvasId) {
  const canvas = document.getElementById(canvasId);
  if(!canvas) return;
  const ctx = canvas.getContext('2d');
  const W=canvas.width, H=canvas.height;
  ctx.clearRect(0,0,W,H);

  ctx.fillStyle='#111827'; ctx.fillRect(0,0,W,H);
  ctx.strokeStyle='rgba(255,255,255,0.04)'; ctx.lineWidth=1;
  for(let x=0;x<W;x+=30){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}
  for(let y=0;y<H;y+=30){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}

  // Normal points cluster
  const normals=[], anomalies=[];
  for(let i=0;i<80;i++) normals.push([randRange(120,260), randRange(80,200)]);
  // Anomalies
  const apos=[[40,40],[330,230],[60,220],[310,50],[180,260],[20,150]];
  apos.forEach(p=>anomalies.push([p[0]+randRange(-10,10),p[1]+randRange(-10,10)]));

  normals.forEach(p => drawCircle(ctx,p[0],p[1],5,'#10b981','rgba(16,185,129,0.4)',1));
  anomalies.forEach(p => {
    drawCircle(ctx,p[0],p[1],8,'#f43f5e','rgba(244,63,94,0.4)',2);
    ctx.strokeStyle='#f43f5e'; ctx.lineWidth=1.5;
    ctx.beginPath(); ctx.moveTo(p[0]-12,p[1]-12); ctx.lineTo(p[0]+12,p[1]+12); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(p[0]+12,p[1]-12); ctx.lineTo(p[0]-12,p[1]+12); ctx.stroke();
  });

  // Legend
  ctx.font='12px Inter'; ctx.textAlign='left';
  drawCircle(ctx,16,H-36,6,'#10b981','rgba(16,185,129,0.4)',1);
  ctx.fillStyle='#94a3b8'; ctx.fillText('Normal (short path)', 28, H-31);
  drawCircle(ctx,16,H-16,6,'#f43f5e','rgba(244,63,94,0.4)',1);
  ctx.fillText('Anomaly (isolated quickly)', 28, H-11);
}

// ══════════════════════════════════════════════════════════════
//  6. SILHOUETTE CHART
// ══════════════════════════════════════════════════════════════
function drawSilhouette(canvasId) {
  const canvas = document.getElementById(canvasId);
  if(!canvas) return;
  const ctx = canvas.getContext('2d');
  const W=canvas.width, H=canvas.height;
  ctx.clearRect(0,0,W,H);

  const clusters = [
    { label:'Cluster A', values:[0.82,0.75,0.71,0.68,0.65,0.62,0.58] },
    { label:'Cluster B', values:[0.78,0.73,0.69,0.64,0.55,0.52,0.48,0.41] },
    { label:'Cluster C', values:[0.88,0.84,0.79,0.76,0.71,0.68] },
  ];

  const pad=50, barH=12, gap=3;
  let y=pad;

  clusters.forEach((cl,ci) => {
    ctx.fillStyle='rgba(255,255,255,0.4)'; ctx.font='11px Inter';
    ctx.textAlign='right'; ctx.fillText(cl.label, pad-8, y+8);

    cl.values.forEach(v => {
      const bw=(W-pad-20)*Math.abs(v);
      const grad=ctx.createLinearGradient(pad,0,pad+bw,0);
      grad.addColorStop(0, COLORS[ci]+'cc');
      grad.addColorStop(1, COLORS[ci]+'44');
      ctx.fillStyle=grad;
      ctx.fillRect(pad, y, bw, barH);
      y += barH+gap;
    });
    y+=8;
  });

  // Axis
  ctx.strokeStyle='#475569'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad,pad-10); ctx.lineTo(pad,y); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(pad,y); ctx.lineTo(W-20,y); ctx.stroke();

  ctx.fillStyle='#6b7280'; ctx.font='10px Inter'; ctx.textAlign='center';
  ctx.fillText('Silhouette Coefficient →', (pad+W-20)/2, y+16);
}

// ══════════════════════════════════════════════════════════════
//  GLOBAL INIT
// ══════════════════════════════════════════════════════════════
window.KMeans      = KMeans;
window.DBSCAN      = DBSCAN;
window.ProjectViz  = ProjectViz;

window.initKMeans = function() {
  KMeans.init('kmeans-canvas');
};
window.stepKMeans = function() {
  KMeans.stepOnce('kmeans-canvas');
};
window.runKMeans = function() {
  KMeans.run('kmeans-canvas');
};
window.runDBSCAN = function() {
  DBSCAN.run('dbscan-canvas');
};
window.resetDBSCAN = function() {
  DBSCAN.reset('dbscan-canvas');
};
window.drawProjections = function() {
  ProjectViz.draw();
};
window.setDataset = function(ds) {
  ProjectViz.setDataset(ds);
  ProjectViz.draw();
};
window.drawElbow         = drawElbow;
window.drawIsolationForest = drawIsolationForest;
window.drawSilhouette    = drawSilhouette;

document.addEventListener('DOMContentLoaded', () => {
  setTimeout(()=>{
    KMeans.init('kmeans-canvas');
    DBSCAN.init('dbscan-canvas');
    drawElbow('elbow-canvas');
    drawIsolationForest('iforest-canvas');
    drawSilhouette('silhouette-canvas');
    ProjectViz.draw();
  }, 200);
});
