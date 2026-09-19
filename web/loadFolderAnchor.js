import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "Load Folder with Anchor";
const PREVIEW_HEIGHT = 220;
const PREVIEW_FPS = 12;

function fitContain(srcW, srcH, maxW, maxH) {
  if (!srcW || !srcH || !maxW || !maxH) return { x: 0, y: 0, w: 0, h: 0 };
  const scale = Math.min(maxW / srcW, maxH / srcH);
  const w = Math.max(1, Math.floor(srcW * scale));
  const h = Math.max(1, Math.floor(srcH * scale));
  return { x: Math.floor((maxW - w) / 2), y: Math.floor((maxH - h) / 2), w, h };
}

function widgetValue(node, name, fallback) {
  const widget = node.widgets?.find((item) => item.name === name);
  return widget?.value ?? fallback;
}

function hideWidget(widget) {
  if (!widget) return;
  widget.hidden = true;
  widget.computeSize = () => [0, -4];
}

function imageUrl(frame) {
  const params = new URLSearchParams({
    filename: frame.filename ?? "",
    subfolder: frame.subfolder ?? "",
    type: frame.type ?? "input",
  });
  return api.apiURL("/view?" + params.toString());
}

function checkerboard(ctx, x, y, width, height, cell = 10) {
  ctx.fillStyle = "#252525";
  ctx.fillRect(x, y, width, height);
  ctx.fillStyle = "#363636";
  for (let row = 0; row * cell < height; row += 1) {
    for (let col = 0; col * cell < width; col += 1) {
      if ((row + col) % 2 === 0) {
        ctx.fillRect(x + col * cell, y + row * cell,
          Math.min(cell, width - col * cell),
          Math.min(cell, height - row * cell));
      }
    }
  }
}

app.registerExtension({
  name: "LinkComfyNodes.LoadFolderAnchor",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      originalOnNodeCreated?.apply(this, arguments);

      hideWidget(this.widgets?.find((widget) => widget.name === "anchor_x"));
      hideWidget(this.widgets?.find((widget) => widget.name === "anchor_y"));

      this.size = [380, Math.max(this.size?.[1] ?? 0, 430)];
      const state = this._lcAnchorPreview = {
        frames: [], image: null, frameIndex: 0, imageCache: new Map(),
        timer: null, loading: false, requestKey: "", pendingSrc: "", error: "",
      };

      const previewGeometry = () => {
        const bottom = (this.widgets ?? []).reduce((value, widget) => {
          if (widget.hidden) return value;
          const y = Number(widget.y) || 0;
          const h = Number(widget.height) ||
            Number(widget.computeSize?.(this.size[0])?.[1]) || 20;
          return Math.max(value, y + h);
        }, 70);
        return {
          drawX: 10, drawY: bottom + 10,
          drawW: Math.max(120, this.size[0] - 20), drawH: PREVIEW_HEIGHT,
        };
      };

      const stopAnimation = () => {
        if (state.timer) clearInterval(state.timer);
        state.timer = null;
      };

      const loadFrame = (index) => {
        if (!state.frames.length) {
          state.image = null;
          return;
        }
        state.frameIndex = ((index % state.frames.length) + state.frames.length) % state.frames.length;
        const src = imageUrl(state.frames[state.frameIndex]);
        state.pendingSrc = src;
        const cached = state.imageCache.get(src);
        if (cached?.complete) {
          state.image = cached;
          this.setDirtyCanvas(true, true);
          return;
        }
        const image = cached || new Image();
        if (!cached) state.imageCache.set(src, image);
        image.onload = () => {
          if (state.pendingSrc !== src) return;
          state.image = image;
          state.loading = false;
          this.setDirtyCanvas(true, true);
        };
        image.onerror = () => {
          if (state.pendingSrc !== src) return;
          state.image = null;
          state.loading = false;
          state.error = "Could not load preview frame";
          this.setDirtyCanvas(true, true);
        };
        state.loading = true;
        state.error = "";
        if (!cached) image.src = src;
      };

      const renderFrames = (frames) => {
        stopAnimation();
        state.frames = Array.isArray(frames) ? frames : [];
        state.frameIndex = 0;
        state.image = null;
        state.imageCache.clear();
        state.error = "";
        if (!state.frames.length) {
          state.loading = false;
          this.setDirtyCanvas(true, true);
          return;
        }
        loadFrame(0);
        if (state.frames.length > 1) {
          state.timer = setInterval(() => loadFrame(state.frameIndex + 1),
            1000 / PREVIEW_FPS);
        }
        this.setDirtyCanvas(true, true);
      };

      const refreshPreview = async () => {
        const directory = String(widgetValue(this, "directory", ""));
        const cap = Number(widgetValue(this, "image_load_cap", 0)) || 0;
        const skip = Number(widgetValue(this, "skip_first_images", 0)) || 0;
        const every = Number(widgetValue(this, "select_every_nth", 1)) || 1;
        const key = [directory, cap, skip, every].join("|");
        if (!directory || key === state.requestKey) return;
        state.requestKey = key;
        state.loading = true;
        state.error = "";
        this.setDirtyCanvas(true, true);
        try {
          const params = new URLSearchParams({
            directory, image_load_cap: String(cap),
            skip_first_images: String(skip), select_every_nth: String(every),
          });
          const response = await fetch(api.apiURL(
            "/link_comfy/load_folder_preview?" + params.toString()));
          if (!response.ok) throw new Error("Preview request failed");
          const payload = await response.json();
          renderFrames(payload.frames ?? []);
        } catch (error) {
          stopAnimation();
          state.loading = false;
          state.error = error?.message || "Could not load folder preview";
          this.setDirtyCanvas(true, true);
        }
      };

      ["directory", "image_load_cap", "skip_first_images", "select_every_nth"]
        .forEach((name) => {
          const widget = this.widgets?.find((item) => item.name === name);
          if (!widget) return;
          const originalCallback = widget.callback;
          widget.callback = (...args) => {
            const result = originalCallback?.apply(widget, args);
            state.requestKey = "";
            refreshPreview();
            return result;
          };
        });

      this.onMouseDown = function (_event, pos) {
        const geometry = previewGeometry();
        const image = state.image;
        if (!image?.width || !image?.height) return false;
        const rect = fitContain(image.width, image.height,
          geometry.drawW, geometry.drawH);
        const imageX = geometry.drawX + rect.x;
        const imageY = geometry.drawY + rect.y;
        if (pos[0] < imageX || pos[0] > imageX + rect.w ||
            pos[1] < imageY || pos[1] > imageY + rect.h) return false;

        const anchorX = Math.max(0, Math.min(image.width - 1,
          Math.round((pos[0] - imageX) * image.width / rect.w)));
        const anchorY = Math.max(0, Math.min(image.height - 1,
          Math.round((pos[1] - imageY) * image.height / rect.h)));
        const xWidget = this.widgets?.find((item) => item.name === "anchor_x");
        const yWidget = this.widgets?.find((item) => item.name === "anchor_y");
        if (xWidget) xWidget.value = anchorX;
        if (yWidget) yWidget.value = anchorY;
        this.setDirtyCanvas(true, true);
        return true;
      };

      this.onDrawForeground = function (ctx) {
        const geometry = previewGeometry();
        const image = state.image;
        ctx.save();
        checkerboard(ctx, geometry.drawX, geometry.drawY,
          geometry.drawW, geometry.drawH);
        ctx.strokeStyle = "#454545";
        ctx.strokeRect(geometry.drawX, geometry.drawY, geometry.drawW, geometry.drawH);

        if (image?.width && image?.height) {
          const rect = fitContain(image.width, image.height,
            geometry.drawW, geometry.drawH);
          const imageX = geometry.drawX + rect.x;
          const imageY = geometry.drawY + rect.y;
          ctx.imageSmoothingEnabled = false;
          ctx.drawImage(image, imageX, imageY, rect.w, rect.h);

          const anchorX = Number(widgetValue(this, "anchor_x", -1));
          const anchorY = Number(widgetValue(this, "anchor_y", -1));
          if (anchorX >= 0 && anchorY >= 0) {
            const crossX = imageX + anchorX / image.width * rect.w;
            const crossY = imageY + anchorY / image.height * rect.h;
            ctx.strokeStyle = "#00e0ff";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(crossX - 9, crossY); ctx.lineTo(crossX + 9, crossY);
            ctx.moveTo(crossX, crossY - 9); ctx.lineTo(crossX, crossY + 9);
            ctx.stroke();
            ctx.fillStyle = "#00e0ff";
            ctx.font = "12px sans-serif";
            ctx.fillText("(" + anchorX + ", " + anchorY + ")", crossX + 8, crossY - 8);
          }
        } else {
          ctx.fillStyle = "#b0b0b0";
          ctx.font = "12px sans-serif";
          ctx.fillText(state.error || (state.loading ? "Loading preview..." : "Choose a folder"),
            geometry.drawX + 10, geometry.drawY + 24);
        }
        ctx.restore();
      };

      const originalOnExecuted = this.onExecuted;
      this.onExecuted = function (message) {
        originalOnExecuted?.apply(this, arguments);
        if (message?.fast_images) renderFrames(message.fast_images);
      };

      const originalOnRemoved = this.onRemoved;
      this.onRemoved = function () {
        stopAnimation();
        originalOnRemoved?.apply(this, arguments);
      };

      setTimeout(refreshPreview, 0);
    };
  },
});
