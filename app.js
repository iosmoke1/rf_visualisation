const canvas = document.getElementById("plotCanvas");
const ctx = canvas.getContext("2d");

const controls = {
  dataset: document.getElementById("datasetSelect"),
  sampleCount: document.getElementById("sampleCount"),
  noise: document.getElementById("noise"),
  trainSplit: document.getElementById("trainSplit"),
  seed: document.getElementById("seedInput"),
  treeCount: document.getElementById("treeCount"),
  maxDepth: document.getElementById("maxDepth"),
  minSplit: document.getElementById("minSplit"),
  minLeaf: document.getElementById("minLeaf"),
  sampleRatio: document.getElementById("sampleRatio"),
  featureRatio: document.getElementById("featureRatio"),
  visibleTrees: document.getElementById("visibleTrees"),
  showLastTree: document.getElementById("showLastTree")
};

const state = {
  data: [],
  train: [],
  test: [],
  forest: null,
  selectedTreeIndex: -1
};

function makeRandom(seed) {
  let value = seed % 2147483647;
  if (value <= 0) value += 2147483646;
  return function random() {
    value = (value * 16807) % 2147483647;
    return (value - 1) / 2147483646;
  };
}

function normalRandom(random) {
  const u = Math.max(random(), 0.000001);
  const v = random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function shuffle(items, random) {
  for (let i = items.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [items[i], items[j]] = [items[j], items[i]];
  }
}

function makePoint(x, y, label) {
  return { x, y, label, train: true };
}

function generateData() {
  const random = makeRandom(Number(controls.seed.value));
  const n = Number(controls.sampleCount.value);
  const noise = Number(controls.noise.value) / 100;
  const data = [];

  for (let i = 0; i < n; i++) {
    let x = 0;
    let y = 0;
    let label = 0;
    const half = i < n / 2 ? 0 : 1;

    if (controls.dataset.value === "moons") {
      const t = random() * Math.PI;
      if (half === 0) {
        x = Math.cos(t) - 0.25;
        y = Math.sin(t) - 0.35;
        label = 0;
      } else {
        x = 1 - Math.cos(t) - 0.25;
        y = -Math.sin(t) + 0.35;
        label = 1;
      }
    }

    if (controls.dataset.value === "circles") {
      const angle = random() * Math.PI * 2;
      const radius = half === 0 ? 0.45 + random() * 0.18 : 1.05 + random() * 0.2;
      x = Math.cos(angle) * radius;
      y = Math.sin(angle) * radius;
      label = half;
    }

    if (controls.dataset.value === "xor") {
      x = random() * 2.4 - 1.2;
      y = random() * 2.4 - 1.2;
      label = x * y > 0 ? 1 : 0;
    }

    if (controls.dataset.value === "blobs") {
      const centerX = half === 0 ? -0.55 : 0.55;
      const centerY = half === 0 ? -0.15 : 0.2;
      x = centerX + normalRandom(random) * 0.34;
      y = centerY + normalRandom(random) * 0.34;
      label = half;
    }

    x += normalRandom(random) * noise;
    y += normalRandom(random) * noise;
    if (random() < noise * 0.18) label = 1 - label;
    data.push(makePoint(x, y, label));
  }

  shuffle(data, random);
  const trainSize = Math.floor(data.length * Number(controls.trainSplit.value) / 100);
  for (let i = 0; i < data.length; i++) {
    data[i].train = i < trainSize;
  }

  state.data = data;
  state.train = data.filter(point => point.train);
  state.test = data.filter(point => !point.train);
  state.forest = new RandomForest(getForestOptions());
  state.selectedTreeIndex = -1;
  updateAll();
}

function gini(labels) {
  if (labels.length === 0) return 0;
  let ones = 0;
  for (const label of labels) {
    if (label === 1) ones++;
  }
  const p1 = ones / labels.length;
  const p0 = 1 - p1;
  return 1 - p0 * p0 - p1 * p1;
}

class DecisionTree {
  constructor(options, random) {
    this.maxDepth = options.maxDepth;
    this.minSplit = options.minSplit;
    this.minLeaf = options.minLeaf;
    this.featureRatio = options.featureRatio;
    this.random = random;
    this.root = null;
    this.featureGain = [0, 0];
  }

  fit(points) {
    this.root = this.build(points, 0);
  }

  build(points, depth) {
    const counts = this.countClasses(points);
    const majority = counts[1] >= counts[0] ? 1 : 0;
    const probability = points.length === 0 ? 0 : counts[1] / points.length;

    if (
      depth >= this.maxDepth ||
      points.length < this.minSplit ||
      counts[0] === 0 ||
      counts[1] === 0
    ) {
      return { type: "leaf", prediction: majority, probability, count: points.length };
    }

    const split = this.findBestSplit(points);
    if (!split) {
      return { type: "leaf", prediction: majority, probability, count: points.length };
    }

    const left = [];
    const right = [];
    for (const point of points) {
      if (point[split.feature] < split.threshold) left.push(point);
      else right.push(point);
    }

    this.featureGain[split.featureIndex] += split.gain;
    return {
      type: "node",
      feature: split.feature,
      featureIndex: split.featureIndex,
      threshold: split.threshold,
      gain: split.gain,
      count: points.length,
      left: this.build(left, depth + 1),
      right: this.build(right, depth + 1)
    };
  }

  findBestSplit(points) {
    const parentLabels = points.map(point => point.label);
    const parentGini = gini(parentLabels);
    const features = this.pickFeatures();
    let best = null;

    for (const feature of features) {
      const sorted = [...points].sort((a, b) => a[feature.name] - b[feature.name]);
      for (let i = 1; i < sorted.length; i++) {
        const previous = sorted[i - 1][feature.name];
        const current = sorted[i][feature.name];
        if (previous === current) continue;

        const threshold = (previous + current) / 2;
        const leftLabels = [];
        const rightLabels = [];

        for (const point of sorted) {
          if (point[feature.name] < threshold) leftLabels.push(point.label);
          else rightLabels.push(point.label);
        }

        if (leftLabels.length < this.minLeaf || rightLabels.length < this.minLeaf) continue;

        const leftWeight = leftLabels.length / points.length;
        const rightWeight = rightLabels.length / points.length;
        const gain = parentGini - leftWeight * gini(leftLabels) - rightWeight * gini(rightLabels);

        if (!best || gain > best.gain || (gain === best.gain && threshold < best.threshold)) {
          best = {
            feature: feature.name,
            featureIndex: feature.index,
            threshold,
            gain
          };
        }
      }
    }

    if (!best || best.gain <= 0) return null;
    return best;
  }

  pickFeatures() {
    const features = [
      { name: "x", index: 0 },
      { name: "y", index: 1 }
    ];
    if (this.featureRatio >= 1) return features;
    return this.random() < 0.5 ? [features[0]] : [features[1]];
  }

  countClasses(points) {
    const counts = [0, 0];
    for (const point of points) {
      counts[point.label]++;
    }
    return counts;
  }

  predict(point) {
    let node = this.root;
    while (node.type !== "leaf") {
      node = point[node.feature] < node.threshold ? node.left : node.right;
    }
    return node.prediction;
  }

  predictProba(point) {
    let node = this.root;
    while (node.type !== "leaf") {
      node = point[node.feature] < node.threshold ? node.left : node.right;
    }
    return node.probability;
  }
}

class RandomForest {
  constructor(options) {
    this.options = options;
    this.random = makeRandom(options.seed + 1000);
    this.trees = [];
    this.oobIndices = [];
  }

  trainOne(points) {
    if (this.trees.length >= this.options.treeCount) return;
    const sample = [];
    const inBag = new Set();
    const size = Math.max(2, Math.floor(points.length * this.options.sampleRatio));

    for (let i = 0; i < size; i++) {
      const index = Math.floor(this.random() * points.length);
      sample.push(points[index]);
      inBag.add(index);
    }

    const oob = [];
    for (let i = 0; i < points.length; i++) {
      if (!inBag.has(i)) oob.push(i);
    }

    const tree = new DecisionTree(this.options, this.random);
    tree.fit(sample);
    this.trees.push(tree);
    this.oobIndices.push(oob);
  }

  removeLast() {
    this.trees.pop();
    this.oobIndices.pop();
  }

  predict(point) {
    if (this.trees.length === 0) return 0;
    let votes = 0;
    for (const tree of this.trees) {
      votes += tree.predict(point);
    }
    return votes >= this.trees.length / 2 ? 1 : 0;
  }

  predictProba(point) {
    if (this.trees.length === 0) return 0;
    let sum = 0;
    for (const tree of this.trees) {
      sum += tree.predictProba(point);
    }
    return sum / this.trees.length;
  }

  oobAccuracy(allPoints) {
    if (this.trees.length === 0) return null;
    let correct = 0;
    let total = 0;
    for (let i = 0; i < allPoints.length; i++) {
      const point = allPoints[i];
      let votes = 0;
      let count = 0;
      for (let t = 0; t < this.trees.length; t++) {
        if (this.oobIndices[t].includes(i)) {
          votes += this.trees[t].predict(point);
          count++;
        }
      }
      if (count > 0) {
        const pred = votes >= count / 2 ? 1 : 0;
        if (pred === point.label) correct++;
        total++;
      }
    }
    if (total === 0) return null;
    return correct / total;
  }

  featureImportance() {
    const gains = [0, 0];
    for (const tree of this.trees) {
      gains[0] += tree.featureGain[0];
      gains[1] += tree.featureGain[1];
    }
    const total = gains[0] + gains[1];
    if (total === 0) return [0, 0];
    return [gains[0] / total, gains[1] / total];
  }

  treeStats() {
    if (this.trees.length === 0) return null;
    let totalDepth = 0;
    let totalLeaves = 0;
    for (const tree of this.trees) {
      totalDepth += this.nodeDepth(tree.root);
      totalLeaves += this.nodeLeafCount(tree.root);
    }
    return {
      avgDepth: totalDepth / this.trees.length,
      totalLeaves: totalLeaves,
      avgLeaves: totalLeaves / this.trees.length
    };
  }

  nodeDepth(node) {
    if (!node || node.type === "leaf") return 0;
    return 1 + Math.max(this.nodeDepth(node.left), this.nodeDepth(node.right));
  }

  nodeLeafCount(node) {
    if (!node) return 0;
    if (node.type === "leaf") return 1;
    return this.nodeLeafCount(node.left) + this.nodeLeafCount(node.right);
  }
}

function getForestOptions() {
  return {
    seed: Number(controls.seed.value),
    treeCount: Number(controls.treeCount.value),
    maxDepth: Number(controls.maxDepth.value),
    minSplit: Number(controls.minSplit.value),
    minLeaf: Number(controls.minLeaf.value),
    sampleRatio: Number(controls.sampleRatio.value) / 100,
    featureRatio: Number(controls.featureRatio.value) / 100
  };
}

function resetForest() {
  state.forest = new RandomForest(getForestOptions());
  state.selectedTreeIndex = -1;
  updateAll();
}

function targetTreeCount() {
  return Number(controls.treeCount.value);
}

function trainedTreeCount() {
  return state.forest ? state.forest.trees.length : 0;
}

function syncForestTarget() {
  const target = targetTreeCount();
  if (state.forest) {
    state.forest.options.treeCount = target;
  }

  const playbackSlider = document.getElementById("playbackSlider");
  playbackSlider.max = target;

  while (trainedTreeCount() > target) {
    state.forest.removeLast();
  }

  if (state.selectedTreeIndex >= trainedTreeCount()) {
    state.selectedTreeIndex = -1;
  }
}

function setTrainedTreeCount(targetCount) {
  syncForestTarget();
  const boundedTarget = Math.max(0, Math.min(targetCount, targetTreeCount()));

  while (trainedTreeCount() < boundedTarget) {
    state.forest.trainOne(state.train);
  }

  while (trainedTreeCount() > boundedTarget) {
    state.forest.removeLast();
  }

  if (state.selectedTreeIndex >= trainedTreeCount()) {
    state.selectedTreeIndex = -1;
  }
}

function fitCanvas() {
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.floor(rect.width * scale);
  canvas.height = Math.floor(rect.height * scale);
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
}

function toScreen(point) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (point.x + 1.7) / 3.4 * rect.width,
    y: (1.7 - point.y) / 3.4 * rect.height
  };
}

function toWorld(screenX, screenY) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: screenX / rect.width * 3.4 - 1.7,
    y: 1.7 - screenY / rect.height * 3.4
  };
}

function drawPlot() {
  fitCanvas();
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);

  if (state.forest && state.forest.trees.length > 0) {
    const cell = 8;
    for (let y = 0; y < rect.height; y += cell) {
      for (let x = 0; x < rect.width; x += cell) {
        const world = toWorld(x + cell / 2, y + cell / 2);
        const p = state.forest.predictProba(world);
        ctx.fillStyle = p >= 0.5
          ? `rgba(214, 90, 74, ${0.12 + p * 0.22})`
          : `rgba(20, 138, 130, ${0.18 + (1 - p) * 0.18})`;
        ctx.fillRect(x, y, cell + 1, cell + 1);
      }
    }
  }

  drawAxes(rect);
  
  if (state.selectedTreeIndex >= 0 && state.selectedTreeIndex < trainedTreeCount()) {
    drawTreeOverlay(state.forest.trees[state.selectedTreeIndex], 0.3, 2, [5, 3], "rgba(20, 32, 30, 0.75)");
  } else if (controls.showLastTree.checked) {
    drawLastTreeOverlay();
  }

  for (const point of state.data) {
    const screen = toScreen(point);
    ctx.beginPath();
    ctx.arc(screen.x, screen.y, point.train ? 4.2 : 5.2, 0, Math.PI * 2);
    ctx.fillStyle = point.label === 0 ? "#148a82" : "#d65a4a";
    ctx.fill();
    if (!point.train) {
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#d59624";
      ctx.stroke();
    }
  }
}

function drawLastTreeOverlay() {
  if (!controls.showLastTree.checked || !state.forest || trainedTreeCount() === 0) return;
  const lastTree = state.forest.trees[trainedTreeCount() - 1];
  drawTreeOverlay(lastTree, 0.25, 1.5, [4, 3], "rgba(20, 32, 30, 0.6)");
}

function drawTreeOverlay(tree, fillAlpha, lineWidth, lineDash, strokeStyle) {
  const bounds = { xMin: -1.7, xMax: 1.7, yMin: -1.7, yMax: 1.7 };
  const cell = 8;
  const rect = canvas.getBoundingClientRect();

  ctx.save();

  for (let y = 0; y < rect.height; y += cell) {
    for (let x = 0; x < rect.width; x += cell) {
      const world = toWorld(x + cell / 2, y + cell / 2);
      const pred = tree.predict(world);
      ctx.fillStyle = pred === 1
        ? `rgba(214, 90, 74, ${fillAlpha})`
        : `rgba(20, 138, 130, ${fillAlpha})`;
      ctx.fillRect(x, y, cell + 1, cell + 1);
    }
  }

  ctx.lineWidth = lineWidth;
  ctx.strokeStyle = strokeStyle;
  ctx.setLineDash(lineDash);
  drawTreeSplitNode(tree.root, bounds);
  
  ctx.restore();
}

function drawTreeSplitNode(node, bounds) {
  if (!node || node.type === "leaf") return;

  if (node.feature === "x") {
    const from = toScreen({ x: node.threshold, y: bounds.yMin });
    const to = toScreen({ x: node.threshold, y: bounds.yMax });
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();

    drawTreeSplitNode(node.left, {
      xMin: bounds.xMin,
      xMax: Math.min(bounds.xMax, node.threshold),
      yMin: bounds.yMin,
      yMax: bounds.yMax
    });
    drawTreeSplitNode(node.right, {
      xMin: Math.max(bounds.xMin, node.threshold),
      xMax: bounds.xMax,
      yMin: bounds.yMin,
      yMax: bounds.yMax
    });
  } else {
    const from = toScreen({ x: bounds.xMin, y: node.threshold });
    const to = toScreen({ x: bounds.xMax, y: node.threshold });
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();

    drawTreeSplitNode(node.left, {
      xMin: bounds.xMin,
      xMax: bounds.xMax,
      yMin: bounds.yMin,
      yMax: Math.min(bounds.yMax, node.threshold)
    });
    drawTreeSplitNode(node.right, {
      xMin: bounds.xMin,
      xMax: bounds.xMax,
      yMin: Math.max(bounds.yMin, node.threshold),
      yMax: bounds.yMax
    });
  }
}

function drawAxes(rect) {
  const zero = toScreen({ x: 0, y: 0 });
  ctx.strokeStyle = "rgba(102, 115, 111, 0.25)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, zero.y);
  ctx.lineTo(rect.width, zero.y);
  ctx.moveTo(zero.x, 0);
  ctx.lineTo(zero.x, rect.height);
  ctx.stroke();
}

function accuracy(points) {
  if (!state.forest || state.forest.trees.length === 0 || points.length === 0) return null;
  let correct = 0;
  for (const point of points) {
    if (state.forest.predict(point) === point.label) correct++;
  }
  return correct / points.length;
}

function confusionMatrix(points) {
  const matrix = { tn: 0, fp: 0, fn: 0, tp: 0 };
  if (!state.forest || state.forest.trees.length === 0) return matrix;
  for (const point of points) {
    const pred = state.forest.predict(point);
    if (point.label === 0 && pred === 0) matrix.tn++;
    if (point.label === 0 && pred === 1) matrix.fp++;
    if (point.label === 1 && pred === 0) matrix.fn++;
    if (point.label === 1 && pred === 1) matrix.tp++;
  }
  return matrix;
}

function safeDivide(numerator, denominator) {
  if (denominator === 0) return null;
  return numerator / denominator;
}

function selectedMetricNames() {
  const toggles = document.querySelectorAll(".metric-toggle");
  const selected = [];
  for (const toggle of toggles) {
    if (toggle.checked) selected.push(toggle.value);
  }
  return selected;
}

function aucRoc(points) {
  if (!state.forest || state.forest.trees.length === 0 || points.length === 0) return null;
  const pairs = points.map(point => ({
    score: state.forest.predictProba(point),
    label: point.label
  })).sort((a, b) => b.score - a.score);

  const positives = pairs.filter(pair => pair.label === 1).length;
  const negatives = pairs.length - positives;
  if (positives === 0 || negatives === 0) return null;

  let tp = 0;
  let fp = 0;
  let prevFpr = 0;
  let prevTpr = 0;
  let area = 0;

  for (const pair of pairs) {
    if (pair.label === 1) tp++;
    else fp++;

    const tpr = tp / positives;
    const fpr = fp / negatives;
    area += (fpr - prevFpr) * (tpr + prevTpr) / 2;
    prevFpr = fpr;
    prevTpr = tpr;
  }

  return area;
}

function aucPr(points) {
  if (!state.forest || state.forest.trees.length === 0 || points.length === 0) return null;
  const pairs = points.map(point => ({
    score: state.forest.predictProba(point),
    label: point.label
  })).sort((a, b) => b.score - a.score);

  const positives = pairs.filter(pair => pair.label === 1).length;
  if (positives === 0) return null;

  let tp = 0;
  let fp = 0;
  let prevRecall = 0;
  let prevPrecision = 1;
  let area = 0;

  for (const pair of pairs) {
    if (pair.label === 1) tp++;
    else fp++;

    const recall = tp / positives;
    const precision = tp / (tp + fp);
    area += (recall - prevRecall) * (precision + prevPrecision) / 2;
    prevRecall = recall;
    prevPrecision = precision;
  }

  return area;
}

function testMetricValues(matrix) {
  const precision = safeDivide(matrix.tp, matrix.tp + matrix.fp);
  const recall = safeDivide(matrix.tp, matrix.tp + matrix.fn);
  const fScore = precision === null || recall === null || precision + recall === 0
    ? null
    : 2 * precision * recall / (precision + recall);

  return {
    precision,
    recall,
    fScore,
    aucRoc: aucRoc(state.test),
    aucPr: aucPr(state.test)
  };
}

function renderExtraMetrics(values) {
  const labels = {
    precision: "Precision test",
    recall: "Recall test",
    fScore: "F-метрика test",
    aucRoc: "AUC-ROC test",
    aucPr: "AUC-PR test"
  };
  const container = document.getElementById("extraMetrics");
  container.innerHTML = "";

  for (const name of selectedMetricNames()) {
    const row = document.createElement("div");
    row.className = "metric-row";
    row.innerHTML = `<span>${labels[name]}</span><strong>${percent(values[name])}</strong>`;
    container.appendChild(row);
  }
}

function percent(value) {
  if (value === null) return "-";
  return `${Math.round(value * 1000) / 10}%`;
}

function renderForestStats() {
  const stats = state.forest.treeStats();
  const container = document.getElementById("forestStats");
  if (!stats) {
    container.innerHTML = "<p class=\"empty-state\">Лес пока не обучен</p>";
    return;
  }
  
  container.innerHTML = `
    <div class="stat-item">
      <div class="stat-label">Ср. глубина</div>
      <div class="stat-value">${stats.avgDepth.toFixed(1)}</div>
    </div>
    <div class="stat-item">
      <div class="stat-label">Всего листьев</div>
      <div class="stat-value">${stats.totalLeaves}</div>
    </div>
    <div class="stat-item">
      <div class="stat-label">Ср. листьев</div>
      <div class="stat-value">${stats.avgLeaves.toFixed(1)}</div>
    </div>
    <div class="stat-item">
      <div class="stat-label">Деревьев</div>
      <div class="stat-value">${trainedTreeCount()}</div>
    </div>
  `;
}

function updateMetrics() {
  document.getElementById("trainedTrees").textContent = `${trainedTreeCount()}/${targetTreeCount()}`;
  document.getElementById("trainAccuracy").textContent = percent(accuracy(state.train));
  document.getElementById("testAccuracy").textContent = percent(accuracy(state.test));
  
  const oobAcc = state.forest.oobAccuracy(state.train);
  const oobElement = document.getElementById("oobAccuracy");
  if (oobElement) oobElement.textContent = percent(oobAcc);

  const matrix = confusionMatrix(state.test);
  const extraValues = testMetricValues(matrix);
  document.getElementById("tn").textContent = matrix.tn;
  document.getElementById("fp").textContent = matrix.fp;
  document.getElementById("fn").textContent = matrix.fn;
  document.getElementById("tp").textContent = matrix.tp;
  renderExtraMetrics(extraValues);

  const importance = state.forest.featureImportance();
  document.getElementById("impX").style.width = `${importance[0] * 100}%`;
  document.getElementById("impY").style.width = `${importance[1] * 100}%`;
  document.getElementById("impXText").textContent = percent(importance[0]);
  document.getElementById("impYText").textContent = percent(importance[1]);
  
  renderForestStats();
}

function renderNode(node, depth) {
  const maxVisibleDepth = 4;
  const line = document.createElement("div");
  line.className = `tree-node depth-${depth <= 4 ? depth : "more"}`;

  if (node.type === "leaf") {
    line.innerHTML = `<strong>лист:</strong> класс ${node.prediction}, p=${node.probability.toFixed(2)}, n=${node.count}`;
    return line;
  }

  if (depth >= maxVisibleDepth) {
    line.classList.add("tree-cut");
    line.textContent = `... поддерево, n=${node.count}`;
    return line;
  }

  line.innerHTML = `<strong>${node.feature} &lt; ${node.threshold.toFixed(2)}</strong>, gain=${node.gain.toFixed(3)}, n=${node.count}`;
  line.appendChild(renderNode(node.left, depth + 1));
  line.appendChild(renderNode(node.right, depth + 1));
  return line;
}

function updateTrees() {
  const list = document.getElementById("treeList");
  const limit = Math.min(Number(controls.visibleTrees.value), state.forest.trees.length);
  document.getElementById("treeLimitNote").textContent = `Показаны первые ${limit}`;
  list.innerHTML = "";

  if (state.forest.trees.length === 0) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "Лес пока не обучен.";
    list.appendChild(empty);
    return;
  }

  for (let i = 0; i < limit; i++) {
    const card = document.createElement("article");
    card.className = `tree-card ${i === state.selectedTreeIndex ? "selected" : ""}`;
    card.id = `tree-card-${i}`;
    card.innerHTML = `<h3>Дерево ${i + 1}</h3>`;
    card.appendChild(renderNode(state.forest.trees[i].root, 0));
    
    card.addEventListener("click", () => {
      state.selectedTreeIndex = state.selectedTreeIndex === i ? -1 : i;
      updateTrees();
      drawPlot();
    });
    
    list.appendChild(card);
  }
}

function updateLabels() {
  syncForestTarget();
  document.getElementById("sampleCountValue").textContent = controls.sampleCount.value;
  document.getElementById("noiseValue").textContent = `${controls.noise.value}%`;
  document.getElementById("trainSplitValue").textContent = `${controls.trainSplit.value}%`;
  document.getElementById("treeCountValue").textContent = controls.treeCount.value;
  document.getElementById("maxDepthValue").textContent = controls.maxDepth.value;
  document.getElementById("minSplitValue").textContent = controls.minSplit.value;
  document.getElementById("minLeafValue").textContent = controls.minLeaf.value;
  document.getElementById("sampleRatioValue").textContent = `${controls.sampleRatio.value}%`;
  document.getElementById("featureRatioValue").textContent = `${controls.featureRatio.value}%`;
  document.getElementById("visibleTreesValue").textContent = controls.visibleTrees.value;
  syncPlaybackSlider();
}

function syncPlaybackSlider() {
  const playbackSlider = document.getElementById("playbackSlider");
  playbackSlider.max = targetTreeCount();
  playbackSlider.value = trainedTreeCount();
  document.getElementById("playbackValue").textContent = `${trainedTreeCount()}/${targetTreeCount()}`;
}

function updateAll() {
  updateLabels();
  drawPlot();
  updateMetrics();
  updateTrees();
  updateButtonStates();
}

function updateButtonStates() {
  const trainOneBtn = document.getElementById("trainOneBtn");
  const trainAllBtn = document.getElementById("trainAllBtn");
  const removeOneBtn = document.getElementById("removeOneBtn");
  const isComplete = trainedTreeCount() >= targetTreeCount();
  const isEmpty = trainedTreeCount() === 0;
  trainOneBtn.disabled = isComplete;
  trainAllBtn.disabled = isComplete;
  removeOneBtn.disabled = isEmpty;
}

document.getElementById("resetDataBtn").addEventListener("click", generateData);
document.getElementById("removeOneBtn").addEventListener("click", () => {
  setTrainedTreeCount(trainedTreeCount() - 1);
  updateAll();
});
document.getElementById("trainOneBtn").addEventListener("click", () => {
  setTrainedTreeCount(trainedTreeCount() + 1);
  updateAll();
});
document.getElementById("trainAllBtn").addEventListener("click", () => {
  setTrainedTreeCount(targetTreeCount());
  updateAll();
});

document.getElementById("playbackSlider").addEventListener("input", (e) => {
  setTrainedTreeCount(Number(e.target.value));
  updateAll();
});

for (const key of Object.keys(controls)) {
  const control = controls[key];
  control.addEventListener("input", () => {
    if (["dataset", "sampleCount", "noise", "trainSplit", "seed"].includes(key)) {
      generateData();
    } else if (key === "treeCount") {
      updateAll();
    } else if (key === "visibleTrees") {
      updateLabels();
      updateTrees();
    } else if (key === "showLastTree") {
      drawPlot();
    } else {
      resetForest();
    }
  });
}

for (const toggle of document.querySelectorAll(".metric-toggle")) {
  toggle.addEventListener("input", updateMetrics);
}

window.addEventListener("resize", drawPlot);
generateData();
