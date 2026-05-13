# Architecture — Unsupervised Learning Interactive Model

## Overview
A zero-dependency, single-page web application covering all 6 units of Unsupervised Learning with live interactive visualizations. Runs directly in any modern browser — no server needed.

---

## File Structure
```
unsupervised_learning_model/
+-- index.html       ? SPA shell with 8 page sections (Home + 6 Units + Project)
+-- styles.css       ? Design system: tokens, components, layout, responsive
+-- viz.js           ? All canvas visualizations (k-Means, DBSCAN, PCA/t-SNE, etc.)
+-- app.js           ? Navigation, tab system, distance/rule calculators
+-- ARCHITECTURE.md  ? This file
+-- PROGRESS.md      ? Feature tracker & what is covered
```

---

## Design System (styles.css)

### Color Tokens
| Token              | Value     | Usage                          |
|--------------------|-----------|--------------------------------|
| --bg-primary       | #0a0e1a   | Page background                |
| --bg-secondary     | #111827   | Sidebar, canvas backgrounds    |
| --bg-card          | #1a2235   | Card backgrounds               |
| --accent-primary   | #6366f1   | Indigo — nav, CTAs             |
| --accent-cyan      | #06b6d4   | Formulas, info alerts          |
| --accent-emerald   | #10b981   | Success, normal data points    |
| --accent-amber     | #f59e0b   | Warnings, elbow marker         |
| --accent-rose      | #f43f5e   | Anomalies, errors              |

### Layout
- Fixed 260px sidebar + flex main content area
- Grid utilities: .grid-2 / .grid-3 / .grid-4
- Fully responsive: sidebar hides on mobile, grids collapse to 1 col

### Key Components
- .card, .hero, .concept-box, .formula-box
- .compare-table, .metric-card, .badge (5 color variants)
- .btn (primary/secondary/emerald/rose)
- .tabs / .tab-btn / .tab-panel (multi-group support)
- .info-alert / .warn-alert

---

## Navigation System (app.js)

### Page Router
```js
showPage(id)
  ? hides all .page elements
  ? shows #id section
  ? sets matching .nav-item as active
```

### Tab System
Each .tab-btn carries [data-group] and [data-tab].
Clicking activates the matching .tab-panel within the same group.
Multiple independent tab groups can exist on a single page.

### Calculators
| Function             | What it does                                        |
|----------------------|-----------------------------------------------------|
| calcDistances()      | Euclidean, Manhattan, Cosine from 2 DOM input points|
| drawDistanceViz()    | Canvas rendering showing metric paths               |
| calcRule()           | Support, Confidence, Lift, Conviction from counts   |
| updateMetricExplain()| Swaps explanation text from metric dropdown         |

---

## Visualization Engine (viz.js)

All visualizations use the HTML5 Canvas 2D API — no third-party libraries.

### KMeans (IIFE Module)
- .init(canvasId)    ? generate 150 points, k-Means++ initialization
- .stepOnce(canvasId)? one assign + update iteration
- .run(canvasId)     ? animated convergence (300ms per step, max 50)
- Renders: Voronoi tint, color-coded points, centroid X markers
- Live metrics: WCSS and iteration count written to DOM

### DBSCAN (IIFE Module)
- .init()            ? generate 3 clusters + 12 noise points
- .run(canvasId)     ? run with eps/minPts from sliders
- .reset(canvasId)   ? regenerate data, clear labels
- Renders: clusters in COLORS palette, noise as × marks

### ProjectViz (IIFE Module — Project Component)
- .setDataset(ds)    ? 'digits' (10 classes) or 'customers' (5 classes)
- .draw()            ? renders PCA + t-SNE side-by-side canvases
- Simulates realistic projection: PCA=overlapping, t-SNE=well-separated

### Standalone Draw Functions
| Function                  | Canvas ID         | Description                  |
|---------------------------|-------------------|------------------------------|
| drawElbow(id)             | elbow-canvas      | WCSS vs k with elbow marker  |
| drawIsolationForest(id)   | iforest-canvas    | Normal vs anomaly scatter    |
| drawSilhouette(id)        | silhouette-canvas | Horizontal bars by cluster   |

---

## Data Flow
```
User action (slider / button click)
        ?
app.js reads DOM inputs
        ?
viz.js function called (canvasId + params)
        ?
Algorithm runs on in-memory JS arrays
        ?
Canvas cleared and redrawn
        ?
Metric outputs written back to DOM spans
```

---

## How to Extend

1. Add nav item in index.html sidebar with data-page="unitN"
2. Add <section id="unitN" class="page"> in index.html
3. Add .unit-N { --unit-color: #hex; } in styles.css
4. Add visualization function in viz.js, expose via window.myFn
5. Call it in the DOMContentLoaded block at bottom of viz.js

---

## Performance Notes
- Voronoi bg computed at 4x downsample ? 16x fewer pixel ops
- DBSCAN O(n²) fine for n < 300; needs KD-tree for n > 10k
- k-Means uses setTimeout (not rAF) for readable step-by-step view
- Fixed canvas dimensions prevent layout reflow on redraw

## Browser Requirements
HTML5 Canvas, CSS Grid, CSS Custom Properties, ES6+
Tested: Chrome 90+, Firefox 88+, Edge 90+, Safari 14+
