import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS = "Preview (webm)";
const MIN_HEIGHT = 160;
const NODE_HEIGHT_PADDING = 54;
const MAX_AUTO_HEIGHT = 800;

function getPreviewsFromMessage(message) {
  const direct = message?.webm_preview;
  if (Array.isArray(direct)) {
    return direct;
  }
  const nested = message?.ui?.webm_preview;
  if (Array.isArray(nested)) {
    return nested;
  }
  return [];
}

function buildViewUrl(item) {
  if (!item?.filename) {
    return null;
  }
  if (typeof item.url === "string" && item.url.length > 0) {
    return item.url.startsWith("/") ? api.apiURL(item.url) : item.url;
  }

  const params = new URLSearchParams({
    filename: item.filename,
    type: item.type || "temp",
    subfolder: item.subfolder || "",
    t: Date.now().toString(),
  });

  return api.apiURL(`/view?${params.toString()}`);
}

app.registerExtension({
  name: "LinkComfyNodes.WebmPreview",

  async nodeCreated(node) {
    if (node.comfyClass !== NODE_CLASS) return;

    const video = document.createElement("video");
    video.loop = true;
    video.muted = true;
    video.setAttribute("muted", "");
    video.autoplay = true;
    video.playsInline = true;
    video.controls = true;
    video.preload = "metadata";
    video.style.cssText =
      "width: 100%; height: 100%; object-fit: contain; display: block;";

    const previewWidget = node.addDOMWidget(
      "webm_preview",
      "webm_preview",
      video,
      { serialize: false, hideOnZoom: false },
    );

    previewWidget.computeSize = function (width) {
      const available = Math.max(
        0,
        (node?.size?.[1] ?? 0) - NODE_HEIGHT_PADDING,
      );
      const height = Math.max(MIN_HEIGHT, available || MIN_HEIGHT);
      this.computedHeight = height;
      return [width, height];
    };

    let hasAutoSized = false;
    video.addEventListener("loadedmetadata", () => {
      if (!hasAutoSized && video.videoWidth && video.videoHeight) {
        hasAutoSized = true;
        const ratio = video.videoHeight / video.videoWidth;
        const nodeWidth = node.size[0];
        const contentWidth = Math.max(1, nodeWidth - 20);
        const targetHeight = Math.round(contentWidth * ratio);
        const clampedHeight = Math.max(
          MIN_HEIGHT,
          Math.min(targetHeight, MAX_AUTO_HEIGHT),
        );
        node.setSize([nodeWidth, clampedHeight + NODE_HEIGHT_PADDING]);
        app.graph?.setDirtyCanvas?.(true, false);
      }
    });

    video.addEventListener("error", () => {
      const details = video.error
        ? `${video.error.code} ${video.error.message || ""}`.trim()
        : "unknown error";
      console.warn(`[WebmPreview] failed to load video: ${details}`);
    });

    let currentUrl = null;
    const originalGetExtraMenuOptions = node.getExtraMenuOptions;

    node.getExtraMenuOptions = function (_, options) {
      if (originalGetExtraMenuOptions) {
        originalGetExtraMenuOptions.apply(this, arguments);
      }
      if (currentUrl) {
        options.unshift({
          content: "Open video in new tab",
          callback: () => window.open(currentUrl, "_blank"),
        });
      }
    };

    const originalOnExecuted = node.onExecuted;
    node.onExecuted = function (message) {
      if (originalOnExecuted) originalOnExecuted.apply(this, arguments);

      const previews = getPreviewsFromMessage(message);
      if (previews.length === 0) {
        currentUrl = null;
        video.pause();
        video.removeAttribute("src");
        video.load();
        return;
      }

      const item = previews[0];
      const nextUrl = buildViewUrl(item);
      if (!nextUrl) return;

      currentUrl = nextUrl;
      video.src = currentUrl;
      void video.play().catch(() => {});
    };

    if ((node?.size?.[1] ?? 0) < MIN_HEIGHT + NODE_HEIGHT_PADDING) {
      node.setSize([node.size?.[0] ?? 240, MIN_HEIGHT + NODE_HEIGHT_PADDING]);
    }
  },
});
