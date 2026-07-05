import { app } from "../../scripts/app.js";

const NODE_NAME = "Sprite Scale Calculator";
const PREVIEW_MIN_HEIGHT = 270;
const NODE_HEIGHT_PADDING = 150;
const SPIRE_REFERENCE = {
  widthInches: 27.5,
  heightInches: 62.5,
  pixelWidth: 55,
  pixelHeight: 125,
  defaultTargetWidthInches: 28,
  defaultTargetHeightInches: 63,
};
const PRESET_DIMENSIONS = {
  Spirie: {
    widthInches: SPIRE_REFERENCE.defaultTargetWidthInches,
    heightInches: SPIRE_REFERENCE.defaultTargetHeightInches,
  },
};

let stylesInjected = false;
let silhouetteImagePromise = null;

const ensureStyles = () => {
  if (stylesInjected || typeof document === "undefined") {
    return;
  }
  stylesInjected = true;

  const style = document.createElement("style");
  style.textContent = `
    .lc-sprite-scale {
      box-sizing: border-box;
      width: 100%;
      min-height: ${PREVIEW_MIN_HEIGHT}px;
      height: 100%;
      padding: 12px;
      background: linear-gradient(180deg, #2e2b27 0%, #24211e 100%);
      border: 1px solid rgba(196, 181, 160, 0.12);
      border-radius: 14px;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
    }

    .lc-sprite-scale__canvas-wrap {
      box-sizing: border-box;
      height: 100%;
      padding: 8px;
      border-radius: 12px;
      background: rgba(20, 18, 17, 0.42);
      border: 1px solid rgba(196, 181, 160, 0.1);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
    }

    .lc-sprite-scale__canvas {
      display: block;
      width: 100%;
      height: 100%;
      min-height: 230px;
      border-radius: 8px;
      background: linear-gradient(180deg, #35312d 0%, #26231f 100%);
    }
  `;
  document.head.appendChild(style);
};

const loadSilhouetteImage = () => {
  if (silhouetteImagePromise) {
    return silhouetteImagePromise;
  }

  silhouetteImagePromise = new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);
    img.src = new URL("./spriteScaleSilhouette.png", import.meta.url).href;
  });

  return silhouetteImagePromise;
};

const getWidget = (node, name) =>
  node.widgets?.find((widget) => widget.name === name) ?? null;

const getWidgetValue = (node, name, fallback = 0) => {
  const widget = getWidget(node, name);
  return widget?.value ?? fallback;
};

const formatNumber = (value) => {
  if (!Number.isFinite(value)) {
    return "-";
  }
  if (Math.abs(value - Math.round(value)) < 0.001) {
    return `${Math.round(value)}`;
  }
  return value.toFixed(2);
};

const formatDimension = (inches) => {
  const wholeInches = Math.max(0, Math.round(inches));
  if (wholeInches < 12) {
    return `${wholeInches}"`;
  }

  const feet = Math.floor(wholeInches / 12);
  const remainingInches = wholeInches % 12;
  if (remainingInches === 0) {
    return `${feet}'`;
  }
  return `${feet}'${remainingInches}"`;
};

const drawPreview = (node) => {
  const state = node._spriteScaleState;
  if (!state) {
    return;
  }

  const { canvas, silhouetteImage } = state;
  const bounds = canvas.getBoundingClientRect();
  const width = Math.max(260, Math.round(bounds.width || canvas.clientWidth || 260));
  const height = Math.max(220, Math.round(bounds.height || canvas.clientHeight || 220));
  const dpr = window.devicePixelRatio || 1;

  if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
    canvas.width = width * dpr;
    canvas.height = height * dpr;
  }

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const ref = SPIRE_REFERENCE;
  const targetWidthInches = Number(
    getWidgetValue(node, "target_width_inches", ref.defaultTargetWidthInches),
  );
  const targetHeightInches = Number(
    getWidgetValue(node, "target_height_inches", ref.defaultTargetHeightInches),
  );

  const pixelsPerInchWidth = ref.pixelWidth / ref.widthInches;
  const pixelsPerInchHeight = ref.pixelHeight / ref.heightInches;
  const targetPixelWidth = Math.max(1, Math.round(targetWidthInches * pixelsPerInchWidth));
  const targetPixelHeight = Math.max(1, Math.round(targetHeightInches * pixelsPerInchHeight));

  const margin = 18;
  const groundY = height - 28;
  const maxVisualHeight = Math.max(ref.pixelHeight, targetPixelHeight, 1);
  const maxVisualWidth = Math.max(ref.pixelWidth, targetPixelWidth, 1);
  const availableHeight = Math.max(80, groundY - 18);
  const availableWidth = Math.max(80, width - (margin * 2) - 18);
  const visualScale = Math.max(
    0.25,
    Math.min(
      availableHeight / maxVisualHeight,
      availableWidth / maxVisualWidth,
    ),
  );
  const refDrawW = Math.max(12, ref.pixelWidth * visualScale);
  const refDrawH = Math.max(24, ref.pixelHeight * visualScale);
  const targetDrawW = Math.max(12, targetPixelWidth * visualScale);
  const targetDrawH = Math.max(24, targetPixelHeight * visualScale);
  const centerX = width / 2;
  const targetX = centerX - targetDrawW / 2;
  const refX = centerX - refDrawW / 2;
  const targetY = groundY - targetDrawH;
  const refY = groundY - refDrawH;

  ctx.fillStyle = "#2f2b27";
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "#7c7266";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(margin, groundY + 0.5);
  ctx.lineTo(width - margin, groundY + 0.5);
  ctx.stroke();

  ctx.fillStyle = "rgba(104, 149, 169, 0.18)";
  ctx.fillRect(targetX, targetY, targetDrawW, targetDrawH);
  ctx.strokeStyle = "#5e93a8";
  ctx.lineWidth = 2;
  ctx.setLineDash([7, 4]);
  ctx.strokeRect(targetX, targetY, targetDrawW, targetDrawH);
  ctx.setLineDash([]);

  if (silhouetteImage) {
    ctx.save();
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(silhouetteImage, refX, refY, refDrawW, refDrawH);
    ctx.restore();
  } else {
    ctx.fillStyle = "#d6cdc0";
    ctx.fillRect(refX, refY, refDrawW, refDrawH);
  }

  ctx.fillStyle = "#d0c5b6";
  ctx.font = '12px "Trebuchet MS", "Segoe UI", sans-serif';
  ctx.textAlign = "center";
  ctx.fillText(
    `${targetPixelWidth}×${targetPixelHeight}px  |  ${formatDimension(targetWidthInches)} × ${formatDimension(targetHeightInches)}`,
    centerX,
    groundY + 18,
  );
  ctx.textAlign = "left";
};

const updateNodeLayout = (node) => {
  node.computeSize?.();
  node.setDirtyCanvas?.(true, true);
  requestAnimationFrame(() => drawPreview(node));
};

app.registerExtension({
  name: "Comfy.LinkComfy.SpriteScaleCalculator",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) {
      return;
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      ensureStyles();

      const wrapper = document.createElement("div");
      wrapper.className = "lc-sprite-scale";

      const canvasWrap = document.createElement("div");
      canvasWrap.className = "lc-sprite-scale__canvas-wrap";
      const canvas = document.createElement("canvas");
      canvas.className = "lc-sprite-scale__canvas";
      canvasWrap.appendChild(canvas);
      wrapper.appendChild(canvasWrap);

      const previewWidget = this.addDOMWidget(
        "preview",
        "sprite_scale_preview",
        wrapper,
        {
          serialize: false,
          hideOnZoom: false,
        },
      );

      previewWidget.computeSize = function (width) {
        const available = Math.max(
          PREVIEW_MIN_HEIGHT,
          (this.node?.size?.[1] ?? PREVIEW_MIN_HEIGHT + NODE_HEIGHT_PADDING) -
            NODE_HEIGHT_PADDING,
        );
        this.computedHeight = available;
        return [width, available];
      };

      if (
        (this.size?.[0] ?? 0) < 340 ||
        (this.size?.[1] ?? 0) < NODE_HEIGHT_PADDING + PREVIEW_MIN_HEIGHT
      ) {
        this.setSize([
          Math.max(this.size?.[0] ?? 0, 340),
          Math.max(this.size?.[1] ?? 0, NODE_HEIGHT_PADDING + PREVIEW_MIN_HEIGHT),
        ]);
      }

      this._spriteScaleState = {
        canvas,
        silhouetteImage: null,
        applyingPreset: false,
      };
      const presetWidget = getWidget(this, "preset");
      const widthWidget = getWidget(this, "target_width_inches");
      const heightWidget = getWidget(this, "target_height_inches");

      if (presetWidget) {
        const originalPresetCallback = presetWidget.callback;
        presetWidget.callback = (...args) => {
          const callbackResult = originalPresetCallback?.apply(presetWidget, args);
          const presetName = presetWidget.value;
          const presetValues = PRESET_DIMENSIONS[presetName];
          if (presetValues) {
            this._spriteScaleState.applyingPreset = true;
            if (widthWidget) {
              widthWidget.value = presetValues.widthInches;
            }
            if (heightWidget) {
              heightWidget.value = presetValues.heightInches;
            }
            this._spriteScaleState.applyingPreset = false;
          }
          updateNodeLayout(this);
          return callbackResult;
        };
      }

      for (const widget of this.widgets ?? []) {
        if (widget === presetWidget) {
          continue;
        }
        const originalCallback = widget.callback;
        widget.callback = (...args) => {
          const callbackResult = originalCallback?.apply(widget, args);
          if (
            (widget === widthWidget || widget === heightWidget) &&
            presetWidget &&
            !this._spriteScaleState.applyingPreset &&
            presetWidget.value !== "Custom"
          ) {
            presetWidget.value = "Custom";
          }
          updateNodeLayout(this);
          return callbackResult;
        };
      }

      const originalOnResize = this.onResize;
      this.onResize = function () {
        const resizeResult = originalOnResize?.apply(this, arguments);
        drawPreview(this);
        return resizeResult;
      };

      const originalOnConfigure = this.onConfigure;
      this.onConfigure = function () {
        const configureResult = originalOnConfigure?.apply(this, arguments);
        updateNodeLayout(this);
        return configureResult;
      };

      loadSilhouetteImage().then((image) => {
        if (this._spriteScaleState) {
          this._spriteScaleState.silhouetteImage = image;
          drawPreview(this);
        }
      });

      updateNodeLayout(this);
      return result;
    };
  },
});
