import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

const VIEWER_HTML = `<!DOCTYPE html>
<html>
<head>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #16182a; color: #ccc; font: 12px monospace; overflow: hidden; display: flex; flex-direction: column; height: 100vh; user-select: none; }
  canvas { flex: 1; display: block; cursor: grab; min-height: 0; }
  canvas.dragging { cursor: grabbing; }
  #controls { padding: 6px 10px; background: #1e2030; display: flex; align-items: center; gap: 8px; flex-shrink: 0; border-top: 1px solid #2a2d42; }
  button { background: #2a2d42; color: #ccc; border: 1px solid #444; padding: 3px 10px; cursor: pointer; border-radius: 3px; font: 12px monospace; }
  button:hover { background: #3a3d52; }
  button.active { background: #2a52a0; color: #fff; border-color: #5080dd; }
  input[type=range] { flex: 1; accent-color: #5080dd; }
  #status { position: absolute; top: 8px; left: 10px; color: #778; pointer-events: none; max-width: 80%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="status"></div>
<div id="controls">
  <button id="btnPlay">▶ Play</button>
  <button id="btnFollow">⊙ Stay</button>
  <input type="range" id="scrubber" min="0" max="100" value="0">
  <span id="frameLabel" style="white-space:nowrap;min-width:60px;text-align:right">0 / 0</span>
</div>
<script>
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const scrubber = document.getElementById('scrubber');
const frameLabel = document.getElementById('frameLabel');
const btnPlay = document.getElementById('btnPlay');
const btnFollow = document.getElementById('btnFollow');
const statusEl = document.getElementById('status');

let xyz = null, bones = [], numFrames = 0, numJoints = 22, fps = 30;
let currentFrame = 0, playing = false, animId = null, lastTime = 0;
let rotY = 0.4, rotX = -0.15, zoom = 1.0;
let dragging = false, lastMX = 0, lastMY = 0, dragIsPan = false;
let panX = 0, panY = 0;
let norm = { cx: 0, cy: 0, cz: 0, scale: 1, floorY: 0 };
let followMode = false;
let gizmoHovered = false;
let lerpAnimId = null;
let lerpFromY = 0, lerpFromX = 0, lerpToY = 0, lerpToX = 0, lerpStartTime = null;
const LERP_DUR = 350;
let velY = 0, velX = 0, velPanX = 0, velPanY = 0;
let momentumId = null;
let dragFrames = 0;

function resize() {
  canvas.width = canvas.clientWidth || 512;
  canvas.height = canvas.clientHeight || 400;
  render();
}

function computeNorm() {
  if (!xyz) return null;
  let minX=Infinity, maxX=-Infinity, minY=Infinity, maxY=-Infinity, minZ=Infinity, maxZ=-Infinity;
  for (let j = 0; j < numJoints; j++) {
    const [x, y, z] = getJoint(0, j);
    if (x < minX) minX=x; if (x > maxX) maxX=x;
    if (y < minY) minY=y; if (y > maxY) maxY=y;
    if (z < minZ) minZ=z; if (z > maxZ) maxZ=z;
  }
  const range = Math.max(maxX-minX, maxY-minY, maxZ-minZ, 0.01);
  const [rx0, , rz0] = getJoint(0, 0);
  return { cx: rx0, cy: (minY+maxY)/2, cz: rz0, scale: 1/range, floorY: minY };
}

function drawGrid() {
  const { floorY, scale, cx: baseCX, cz: baseCZ } = norm;
  let gcx = baseCX, gcz = baseCZ;
  if (followMode && numFrames > 0) {
    const root = getJoint(Math.floor(currentFrame) % numFrames, 0);
    gcx = root[0]; gcz = root[2];
  }
  const step = 0.3 / scale;
  const nZ = 7;
  const nX = nZ * 3;
  const halfZ = nZ * step;
  const halfX = nX * step;
  ctx.save();
  ctx.strokeStyle = '#252840';
  ctx.lineWidth = 0.8;
  for (let i = -nX; i <= nX; i++) {
    const x = gcx + i * step;
    const [ax0, ay0] = project(x, floorY, gcz - halfZ);
    const [ax1, ay1] = project(x, floorY, gcz + halfZ);
    ctx.beginPath(); ctx.moveTo(ax0, ay0); ctx.lineTo(ax1, ay1); ctx.stroke();
  }
  for (let i = -nZ; i <= nZ; i++) {
    const z = gcz + i * step;
    const [bx0, by0] = project(gcx - halfX, floorY, z);
    const [bx1, by1] = project(gcx + halfX, floorY, z);
    ctx.beginPath(); ctx.moveTo(bx0, by0); ctx.lineTo(bx1, by1); ctx.stroke();
  }
  ctx.restore();
}

function autoFrame() {
  if (!norm) return;
  const savedZoom = zoom;
  zoom = 1.0;
  let minSX=Infinity, maxSX=-Infinity, minSY=Infinity, maxSY=-Infinity;
  for (let j = 0; j < numJoints; j++) {
    const [sx, sy] = project(...getJoint(0, j));
    if (sx < minSX) minSX=sx; if (sx > maxSX) maxSX=sx;
    if (sy < minSY) minSY=sy; if (sy > maxSY) maxSY=sy;
  }
  const cW = maxSX - minSX || 1, cH = maxSY - minSY || 1;
  zoom = Math.min((canvas.width * 0.75) / cW, (canvas.height * 0.75) / cH);
  if (!isFinite(zoom) || zoom <= 0) zoom = savedZoom;
}

function getJoint(frame, joint) {
  const i = (frame * numJoints + joint) * 3;
  return [xyz[i], xyz[i+1], xyz[i+2]];
}

function project(x, y, z) {
  const { scale, cx: baseCX, cy, cz: baseCZ } = norm;
  let cx = baseCX, cz = baseCZ;
  if (followMode) {
    const f = Math.floor(currentFrame) % numFrames;
    const root = getJoint(f, 0);
    cx = root[0]; cz = root[2];
  }
  let px = (x-cx)*scale, py = (y-cy)*scale, pz = (z-cz)*scale;
  const cosY = Math.cos(rotY), sinY = Math.sin(rotY);
  const rx = px*cosY - pz*sinY;
  const rz = px*sinY + pz*cosY;
  const cosX = Math.cos(rotX), sinX = Math.sin(rotX);
  const ry = py*cosX - rz*sinX;
  const s = Math.min(canvas.width, canvas.height) * 0.55 * zoom;
  return [rx*s + canvas.width/2 + panX, -ry*s + canvas.height/2 + panY];
}

const GIZMO_ARM = 36;
const GIZMO_R = 50;
const GIZMO_AXES = [
  { id: 'X',  vec: [ 1, 0, 0], color: '#e05555', snapY: -Math.PI/2, snapX: 0 },
  { id: 'Y',  vec: [ 0, 1, 0], color: '#55c055', snapY: 0, snapX: -Math.PI/2 },
  { id: 'Z',  vec: [ 0, 0, 1], color: '#5599e0', snapY: 0, snapX: 0 },
  { id: '-X', vec: [-1, 0, 0], color: '#703030', snapY:  Math.PI/2, snapX: 0 },
  { id: '-Y', vec: [ 0,-1, 0], color: '#306030', snapY: 0, snapX:  Math.PI/2 },
  { id: '-Z', vec: [ 0, 0,-1], color: '#304870', snapY: Math.PI, snapX: 0 },
];

function gizmoCenter() {
  return [canvas.width - GIZMO_R - 10, GIZMO_R + 10];
}

function projectAxisDir(ax, ay, az) {
  const cosY = Math.cos(rotY), sinY = Math.sin(rotY);
  const rx = ax*cosY - az*sinY;
  const rz = ax*sinY + az*cosY;
  const cosX = Math.cos(rotX), sinX = Math.sin(rotX);
  const ry = ay*cosX - rz*sinX;
  const depth = rz*cosX + ay*sinX;
  return [rx, -ry, depth];
}

function getGizmoEndpoints() {
  const [gx, gy] = gizmoCenter();
  return GIZMO_AXES.map(axis => {
    const [sx, sy, depth] = projectAxisDir(...axis.vec);
    return { ...axis, ex: gx + sx*GIZMO_ARM, ey: gy + sy*GIZMO_ARM, depth };
  });
}

function drawGizmo() {
  const [gx, gy] = gizmoCenter();
  const eps = getGizmoEndpoints().sort((a, b) => a.depth - b.depth);
  ctx.save();
  ctx.beginPath();
  ctx.arc(gx, gy, GIZMO_R, 0, Math.PI*2);
  ctx.fillStyle = 'rgba(22,24,42,0.45)';
  ctx.fill();
  for (const e of eps) {
    const isPos = !e.id.startsWith('-');
    ctx.globalAlpha = isPos ? 0.85 : 0.4;
    ctx.beginPath();
    ctx.moveTo(gx, gy);
    ctx.lineTo(e.ex, e.ey);
    ctx.strokeStyle = e.color;
    ctx.lineWidth = isPos ? 2 : 1.5;
    ctx.stroke();
  }
  for (const e of eps) {
    const isPos = !e.id.startsWith('-');
    ctx.globalAlpha = isPos ? 1.0 : 0.55;
    ctx.beginPath();
    ctx.arc(e.ex, e.ey, isPos ? 10 : 6, 0, Math.PI*2);
    ctx.fillStyle = e.color;
    ctx.fill();
    if (isPos) {
      ctx.globalAlpha = 1.0;
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 8px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(e.id, e.ex, e.ey);
    }
  }
  ctx.globalAlpha = 1.0;
  ctx.beginPath();
  ctx.arc(gx, gy, 4, 0, Math.PI*2);
  ctx.fillStyle = '#aaaaaa';
  ctx.fill();
  ctx.restore();
}

function gizmoHitTest(mx, my) {
  const eps = getGizmoEndpoints();
  for (const e of eps) {
    if (Math.hypot(mx - e.ex, my - e.ey) <= (!e.id.startsWith('-') ? 12 : 8)) return e;
  }
  return null;
}

function angleMatch(a, b) {
  let d = Math.abs(a - b) % (2 * Math.PI);
  if (d > Math.PI) d = 2 * Math.PI - d;
  return d < 0.05;
}

function lerpAngle(a, b, t) {
  let diff = b - a;
  while (diff >  Math.PI) diff -= 2 * Math.PI;
  while (diff < -Math.PI) diff += 2 * Math.PI;
  return a + diff * t;
}

function animateLerp(ts) {
  if (!lerpStartTime) lerpStartTime = ts;
  const t = Math.min(1, (ts - lerpStartTime) / LERP_DUR);
  const ease = t < 0.5 ? 2*t*t : -1 + (4 - 2*t) * t;
  rotY = lerpAngle(lerpFromY, lerpToY, ease);
  rotX = lerpFromX + (lerpToX - lerpFromX) * ease;
  render();
  if (t < 1) lerpAnimId = requestAnimationFrame(animateLerp);
  else lerpAnimId = null;
}

function startLerp(toY, toX) {
  if (lerpAnimId) cancelAnimationFrame(lerpAnimId);
  if (momentumId) { cancelAnimationFrame(momentumId); momentumId = null; }
  lerpFromY = rotY; lerpFromX = rotX;
  lerpToY = toY; lerpToX = toX;
  lerpStartTime = null;
  lerpAnimId = requestAnimationFrame(animateLerp);
}

const FRICTION = 0.85;
function momentumStep() {
  velY *= FRICTION; velX *= FRICTION;
  velPanX *= FRICTION; velPanY *= FRICTION;
  if (Math.abs(velY) + Math.abs(velX) + Math.abs(velPanX) + Math.abs(velPanY) < 0.0003) {
    momentumId = null; return;
  }
  rotY += velY;
  rotX = Math.max(-1.4, Math.min(1.4, rotX + velX));
  panX += velPanX;
  panY += velPanY;
  render();
  momentumId = requestAnimationFrame(momentumStep);
}

function render() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawGrid();
  if (xyz) {
    const f = Math.floor(currentFrame) % numFrames;
    const pts = [];
    for (let j = 0; j < numJoints; j++) pts.push(project(...getJoint(f, j)));
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = '#4a70cc';
    for (const [a, b] of bones) {
      if (a >= pts.length || b >= pts.length) continue;
      ctx.beginPath();
      ctx.moveTo(pts[a][0], pts[a][1]);
      ctx.lineTo(pts[b][0], pts[b][1]);
      ctx.stroke();
    }
    for (const [sx, sy] of pts) {
      ctx.beginPath();
      ctx.arc(sx, sy, 3.5, 0, Math.PI*2);
      ctx.fillStyle = '#ccd';
      ctx.fill();
    }
  }
  drawGizmo();
}

function step(ts) {
  if (!playing) return;
  if (!lastTime) lastTime = ts;
  currentFrame = (currentFrame + (ts - lastTime) / 1000 * fps) % numFrames;
  lastTime = ts;
  const f = Math.floor(currentFrame);
  scrubber.value = f;
  frameLabel.textContent = f + ' / ' + numFrames;
  render();
  animId = requestAnimationFrame(step);
}

function setPlaying(v) {
  playing = v;
  btnPlay.textContent = v ? '⏸ Pause' : '▶ Play';
  if (v) { lastTime = 0; animId = requestAnimationFrame(step); }
  else if (animId) { cancelAnimationFrame(animId); animId = null; }
}

function setFollow(v) {
  if (!v && followMode && norm && numFrames > 0) {
    const root = getJoint(Math.floor(currentFrame) % numFrames, 0);
    norm.cx = root[0];
    norm.cz = root[2];
  }
  followMode = v;
  btnFollow.classList.toggle('active', v);
  render();
}

btnPlay.addEventListener('click', () => setPlaying(!playing));
btnFollow.addEventListener('click', () => setFollow(!followMode));
scrubber.addEventListener('input', () => {
  currentFrame = +scrubber.value;
  frameLabel.textContent = Math.floor(currentFrame) + ' / ' + numFrames;
  render();
});

canvas.addEventListener('mousedown', e => {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / (rect.width || 1);
  const scaleY = canvas.height / (rect.height || 1);
  const mx = (e.clientX - rect.left) * scaleX;
  const my = (e.clientY - rect.top) * scaleY;
  const hit = gizmoHitTest(mx, my);
  if (hit) {
    if (angleMatch(rotY, hit.snapY) && angleMatch(rotX, hit.snapX)) {
      const oppId = hit.id.startsWith('-') ? hit.id.slice(1) : '-' + hit.id;
      const opp = GIZMO_AXES.find(a => a.id === oppId);
      startLerp(opp.snapY, opp.snapX);
    } else {
      startLerp(hit.snapY, hit.snapX);
    }
    return;
  }
  if (momentumId) { cancelAnimationFrame(momentumId); momentumId = null; }
  velY = 0; velX = 0; velPanX = 0; velPanY = 0;
  dragging = true; dragIsPan = e.shiftKey; dragFrames = 0;
  lastMX = e.clientX; lastMY = e.clientY;
  canvas.classList.add('dragging');
});

window.addEventListener('mousemove', e => {
  if (!dragging) {
    const rect = canvas.getBoundingClientRect();
    if (rect.width && rect.height) {
      const mx = (e.clientX - rect.left) * (canvas.width / rect.width);
      const my = (e.clientY - rect.top) * (canvas.height / rect.height);
      const nowHovered = !!gizmoHitTest(mx, my);
      if (nowHovered !== gizmoHovered) {
        gizmoHovered = nowHovered;
        canvas.style.cursor = gizmoHovered ? 'pointer' : '';
      }
    }
    return;
  }
  const dx = e.clientX - lastMX, dy = e.clientY - lastMY;
  const ease = Math.min(1, ++dragFrames / 6);
  if (dragIsPan) {
    const pdx = dx * ease, pdy = dy * ease;
    panX += pdx; panY += pdy;
    velPanX = pdx; velPanY = pdy; velY = 0; velX = 0;
  } else {
    const rdx = dx * 0.007 * ease, rdy = dy * 0.007 * ease;
    rotY += rdx;
    rotX = Math.max(-1.4, Math.min(1.4, rotX - rdy));
    velY = rdx; velX = -rdy; velPanX = 0; velPanY = 0;
  }
  lastMX = e.clientX; lastMY = e.clientY;
  render();
});

window.addEventListener('mouseup', () => {
  dragging = false;
  canvas.classList.remove('dragging');
  if (Math.abs(velY) + Math.abs(velX) + Math.abs(velPanX) + Math.abs(velPanY) > 0.0003) {
    if (!momentumId) momentumId = requestAnimationFrame(momentumStep);
  }
});

canvas.addEventListener('wheel', e => {
  e.preventDefault();
  zoom = Math.max(0.2, Math.min(8.0, zoom * (e.deltaY < 0 ? 1.1 : 0.9)));
  render();
}, { passive: false });

window.addEventListener('message', e => {
  if (e.data?.type === 'LOAD_MOTION') {
    const d = e.data.motionData;
    xyz = new Float32Array(d.xyz);
    numFrames = d.num_frames;
    numJoints = d.num_joints || 22;
    fps = d.fps || 30;
    bones = d.bones || [];
    currentFrame = 0;
    panX = 0; panY = 0;
    norm = computeNorm();
    autoFrame();
    setFollow(followMode);
    scrubber.max = numFrames - 1;
    scrubber.value = 0;
    frameLabel.textContent = '0 / ' + numFrames;
    statusEl.textContent = d.text || '';
    render();
    setPlaying(true);
  } else if (e.data?.type === 'RESIZE') {
    resize();
  }
});

new ResizeObserver(resize).observe(canvas);
resize();
if (window.parent) window.parent.postMessage({ type: 'VIEWER_READY' }, '*');
</script>
</body>
</html>`;

const NODE_CONFIG = {
  "HybridMaMoMask Generate": {
    extensionName: "hybrid_mamomask.generate",
    textWidgets: { text: "Motion description" },
  },
  "HybridMaMoMask Export FBX": {
    extensionName: "hybrid_mamomask.exportfbx",
  },
  "HybridMaMoMask Preview Animation (3D)": {
    extensionName: "hybrid_mamomask.previewanimation",
    widgetType: "MS_MOTION_PREVIEW",
  },
};

function wireTextWidgets(nodeType, labels) {
  const onNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function() {
    const r = onNodeCreated?.apply(this, arguments);
    const applyHeights = () => {
      for (const [name, placeholder] of Object.entries(labels)) {
        const widget = this.widgets?.find(w => w.name === name);
        if (!widget) continue;
        widget.computeSize = width => [width, 58];
        if (widget.inputEl) widget.inputEl.placeholder = placeholder;
      }
      this.setSize(this.computeSize());
      this.setDirtyCanvas(true, true);
    };
    requestAnimationFrame(applyHeights);
    return r;
  };
}

function wireExportButton(node) {
  const btn = node.addWidget("button", "No File", null, () => {
    if (!node._fbxDownloadUrl) return;
    const a = document.createElement("a");
    a.href = api.apiURL(node._fbxDownloadUrl);
    a.download = node._fbxFilename || "motion.fbx";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  });
  btn.serialize = false;
  btn.disabled = true;
  node._fbxDownloadButton = btn;

  const onExecuted = node.onExecuted;
  node.onExecuted = function(message) {
    onExecuted?.apply(this, arguments);
    const text = message?.text?.[0];
    if (!text) return;
    const urlMatch = text.match(/href="([^"]+)"/);
    const nameMatch = text.match(/Download:\s*([^<]+)/);
    if (!urlMatch) return;
    node._fbxDownloadUrl = urlMatch[1];
    node._fbxFilename = nameMatch ? nameMatch[1].trim() : "motion.fbx";
    btn.label = `\u{1F4BE} ${node._fbxFilename}`;
    btn.disabled = false;
  };
}

function wirePreviewNode(nodeType, widgetType = "MS_MOTION_PREVIEW") {
  const onNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function() {
    const r = onNodeCreated?.apply(this, arguments);

    const iframe = document.createElement("iframe");
    iframe.style.cssText = "width:100%;height:100%;border:none;background:#16182a;display:block;";
    const blob = new Blob([VIEWER_HTML], { type: "text/html" });
    iframe.src = URL.createObjectURL(blob);
    iframe.addEventListener("load", () => { iframe._blobUrl = iframe.src; });

    const widget = this.addDOMWidget("preview", widgetType, iframe, {
      getValue() { return ""; },
      setValue() {}
    });
    widget.computeSize = w => [w || 512, (w || 512) * 1.05];
    widget.element = iframe;
    this.motionViewerIframe = iframe;
    this.motionViewerReady = false;

    window.addEventListener("message", e => {
      if (e.data?.type === "VIEWER_READY") this.motionViewerReady = true;
    });

    const notifyResize = () => iframe.contentWindow?.postMessage({ type: "RESIZE" }, "*");
    this.onResize = () => requestAnimationFrame(notifyResize);

    const ro = new ResizeObserver(() => requestAnimationFrame(notifyResize));
    ro.observe(iframe);

    const onRemoved = this.onRemoved;
    this.onRemoved = function() {
      ro.disconnect();
      if (iframe._blobUrl) URL.revokeObjectURL(iframe._blobUrl);
      onRemoved?.apply(this, arguments);
    };

    this.setSize([512, 570]);

    const onExecuted = this.onExecuted;
    this.onExecuted = function(message) {
      onExecuted?.apply(this, arguments);
      if (!message?.motion_data?.[0]) return;
      try {
        const motionData = JSON.parse(message.motion_data[0]);
        const send = () => iframe.contentWindow?.postMessage({ type: "LOAD_MOTION", motionData }, "*");
        if (this.motionViewerReady) {
          send();
        } else {
          const t = setInterval(() => {
            if (this.motionViewerReady) {
              clearInterval(t);
              send();
            }
          }, 50);
          setTimeout(() => {
            clearInterval(t);
            send();
          }, 2000);
        }
      } catch (e) {
        console.error("[HybridMaMoMask] Failed to parse motion data:", e);
      }
    };

    return r;
  };
}

app.registerExtension({
  name: NODE_CONFIG["HybridMaMoMask Generate"].extensionName,
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "HybridMaMoMask Generate") return;
    wireTextWidgets(nodeType, NODE_CONFIG["HybridMaMoMask Generate"].textWidgets);
  }
});

app.registerExtension({
  name: NODE_CONFIG["HybridMaMoMask Export FBX"].extensionName,
  async nodeCreated(node) {
    if (node.comfyClass !== "HybridMaMoMask Export FBX") return;
    wireExportButton(node);
  }
});

app.registerExtension({
  name: NODE_CONFIG["HybridMaMoMask Preview Animation (3D)"].extensionName,
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "HybridMaMoMask Preview Animation (3D)") return;
    wirePreviewNode(nodeType, NODE_CONFIG["HybridMaMoMask Preview Animation (3D)"].widgetType);
  }
});
