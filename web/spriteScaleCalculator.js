import { app } from "../../scripts/app.js";

const NODE_NAME = "Sprite Scale Calculator";
const MIN_NODE_WIDTH = 350;
const MIN_NODE_HEIGHT = 550;
const SPIRE_REFERENCE = {
  widthInches: 27.5,
  heightInches: 62.5,
  pixelWidth: 55,
  pixelHeight: 125,
  width: 35,
  height: 79,
};
const PRESET_DIMENSIONS = {
  Spirie: {
    width: SPIRE_REFERENCE.width,
    height: SPIRE_REFERENCE.height,
  },
};

let silhouetteImagePromise = null;

const loadSilhouetteImage = () => {
  if (silhouetteImagePromise) {
    return silhouetteImagePromise;
  }

  silhouetteImagePromise = new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);
    img.src = new URL("./spriteScaleSilhouette.svg", import.meta.url).href;
  });

  return silhouetteImagePromise;
};

const getWidget = (node, name) =>
  node.widgets?.find((widget) => widget.name === name) ?? null;

const getWidgetValue = (node, name, fallback = 0) => {
  const widget = getWidget(node, name);
  return widget?.value ?? fallback;
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

const getPreviewRect = (node) => {
  const horizontalPadding = 12;
  const bottomPadding = 12;
  const widgets = node?.widgets ?? [];
  const widgetYs = widgets
    .map((widget) => widget?.last_y)
    .filter((value) => Number.isFinite(value));
  const widgetsBottom = widgetYs.length
    ? Math.max(...widgetYs) + 24
    : 44 + widgets.length * 24;
  const x = horizontalPadding;
  const y = Math.max(44, widgetsBottom + 8);
  const width = Math.max(1, (node?.size?.[0] ?? 0) - horizontalPadding * 2);
  const height = Math.max(1, (node?.size?.[1] ?? 0) - y - bottomPadding);
  return { x, y, width, height };
};

const drawPreview = (node, ctx) => {
  const state = node._spriteScaleState;
  if (!state) {
    return;
  }

  const { silhouetteImage } = state;
  const rect = getPreviewRect(node);
  const width = rect.width;
  const height = rect.height;
  if (width <= 1 || height <= 1) {
    return;
  }

  const ref = SPIRE_REFERENCE;
  const targetWidthInches = Number(
    getWidgetValue(node, "target_width_inches", ref.width),
  );
  const targetHeightInches = Number(
    getWidgetValue(node, "target_height_inches", ref.height),
  );
  const widthKnown = targetWidthInches > 0;
  const heightKnown = targetHeightInches > 0;

  const pixelsPerInchWidth = ref.pixelWidth / ref.widthInches;
  const pixelsPerInchHeight = ref.pixelHeight / ref.heightInches;

  // Either (or both) dimensions may be unset (0) — the user may only know
  // one measurement. Keep the unknown side as null so we never draw a box
  // that implies a size we don't actually have.
  const targetPixelWidth = widthKnown
    ? Math.max(1, Math.round(targetWidthInches * pixelsPerInchWidth))
    : null;
  const targetPixelHeight = heightKnown
    ? Math.max(1, Math.round(targetHeightInches * pixelsPerInchHeight))
    : null;

  const margin = 18;
  const innerX = rect.x;
  const innerY = rect.y;
  const groundY = innerY + height - 28;
  const maxVisualHeight = Math.max(ref.pixelHeight, targetPixelHeight ?? 0, 1);
  const maxVisualWidth = Math.max(ref.pixelWidth, targetPixelWidth ?? 0, 1);
  const availableHeight = Math.max(80, height - 62);
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
  const targetDrawW = targetPixelWidth !== null ? Math.max(12, targetPixelWidth * visualScale) : null;
  const targetDrawH = targetPixelHeight !== null ? Math.max(24, targetPixelHeight * visualScale) : null;
  const centerX = innerX + width / 2;
  const refX = centerX - refDrawW / 2;
  const refY = groundY - refDrawH;
  const targetX = targetDrawW !== null ? centerX - targetDrawW / 2 : null;
  const targetY = targetDrawH !== null ? groundY - targetDrawH : null;

  ctx.save();
  ctx.beginPath();
  ctx.roundRect(rect.x, rect.y, rect.width, rect.height, 14);
  ctx.clip();

  const background = ctx.createLinearGradient(0, rect.y, 0, rect.y + rect.height);
  background.addColorStop(0, "#2e2b27");
  background.addColorStop(1, "#24211e");
  ctx.fillStyle = background;
  ctx.fillRect(rect.x, rect.y, rect.width, rect.height);

  ctx.fillStyle = "rgba(20, 18, 17, 0.42)";
  ctx.fillRect(rect.x + 8, rect.y + 8, rect.width - 16, rect.height - 16);

  ctx.strokeStyle = "#7c7266";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(innerX + margin, groundY + 0.5);
  ctx.lineTo(innerX + width - margin, groundY + 0.5);
  ctx.stroke();

  const CAP_TICK = 14;
  ctx.strokeStyle = "#5e93a8";
  ctx.fillStyle = "rgba(104, 149, 169, 0.18)";

  if (widthKnown && heightKnown) {
    ctx.fillRect(targetX, targetY, targetDrawW, targetDrawH);
    ctx.lineWidth = 2;
    ctx.setLineDash([7, 4]);
    ctx.strokeRect(targetX, targetY, targetDrawW, targetDrawH);
    ctx.setLineDash([]);
  } else if (heightKnown) {
    // Width unknown: show the height extent as two end-cap ticks joined by
    // a connector, instead of a box that would falsely imply a width.
    ctx.lineWidth = 2;
    ctx.setLineDash([7, 4]);
    ctx.beginPath();
    ctx.moveTo(centerX - CAP_TICK, targetY);
    ctx.lineTo(centerX + CAP_TICK, targetY);
    ctx.moveTo(centerX - CAP_TICK, targetY + targetDrawH);
    ctx.lineTo(centerX + CAP_TICK, targetY + targetDrawH);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(centerX, targetY);
    ctx.lineTo(centerX, targetY + targetDrawH);
    ctx.stroke();
  } else if (widthKnown) {
    // Height unknown: mirror the above with vertical end-cap ticks joined
    // by a horizontal connector at the reference figure's midline.
    const midY = refY + refDrawH / 2;
    ctx.lineWidth = 2;
    ctx.setLineDash([7, 4]);
    ctx.beginPath();
    ctx.moveTo(targetX, midY - CAP_TICK);
    ctx.lineTo(targetX, midY + CAP_TICK);
    ctx.moveTo(targetX + targetDrawW, midY - CAP_TICK);
    ctx.lineTo(targetX + targetDrawW, midY + CAP_TICK);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(targetX, midY);
    ctx.lineTo(targetX + targetDrawW, midY);
    ctx.stroke();
  }

  if (silhouetteImage) {
    ctx.save();
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(silhouetteImage, refX, refY, refDrawW, refDrawH);
    ctx.restore();
  } else {
    ctx.fillStyle = "#d6cdc0";
    ctx.fillRect(refX, refY, refDrawW, refDrawH);
  }

  const pixelParts = [];
  if (widthKnown) pixelParts.push(`${targetPixelWidth}`);
  if (heightKnown) pixelParts.push(`${targetPixelHeight}`);
  const pixelLabel = pixelParts.length ? `${pixelParts.join("×")}px` : "";

  const dimensionParts = [];
  if (widthKnown) dimensionParts.push(formatDimension(targetWidthInches));
  if (heightKnown) dimensionParts.push(formatDimension(targetHeightInches));
  const dimensionLabel = dimensionParts.join(" × ");

  const label = [pixelLabel, dimensionLabel].filter(Boolean).join("  |  ");

  ctx.fillStyle = "#d0c5b6";
  ctx.font = '12px "Trebuchet MS", "Segoe UI", sans-serif';
  ctx.textAlign = "center";
  if (label) {
    ctx.fillText(label, centerX, groundY + 18);
  }
  ctx.textAlign = "left";
  ctx.restore();
};

const updateNodeLayout = (node) => {
  node.setDirtyCanvas?.(true, true);
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

      if (
        (this.size?.[0] ?? 0) < MIN_NODE_WIDTH ||
        (this.size?.[1] ?? 0) < MIN_NODE_HEIGHT
      ) {
        this.setSize([
          Math.max(this.size?.[0] ?? 0, MIN_NODE_WIDTH),
          Math.max(this.size?.[1] ?? 0, MIN_NODE_HEIGHT),
        ]);
      }

      this._spriteScaleState = {
        silhouetteImage: null,
        applyingPreset: false,
      };

      const originalOnDrawBackground = this.onDrawBackground;
      this.onDrawBackground = function (ctx) {
        originalOnDrawBackground?.apply(this, arguments);
        drawPreview(this, ctx);
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
              widthWidget.value = presetValues.width;
            }
            if (heightWidget) {
              heightWidget.value = presetValues.height;
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
        this.setDirtyCanvas?.(true, true);
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
          this.setDirtyCanvas?.(true, true);
        }
      });

      updateNodeLayout(this);
      return result;
    };
  },
});
