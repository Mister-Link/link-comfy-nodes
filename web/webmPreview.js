import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS = "Preview (webm)";
const MIN_HEIGHT = 160;
const NODE_HEIGHT_PADDING = 54;

app.registerExtension({
  name: "LinkComfyNodes.WebmPreview",

  async nodeCreated(node) {
    if (node.comfyClass !== NODE_CLASS) return;

    const wrapper = document.createElement("div");
    wrapper.style.cssText = `
      width: 100%;
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #1a1a1a;
      box-sizing: border-box;
      overflow: hidden;
      pointer-events: none;
    `;

    const video = document.createElement("video");
    video.loop = true;
    video.muted = true;
    video.setAttribute("muted", "");
    video.playsInline = true;
    video.autoplay = true;
    video.controls = true;
    video.style.cssText = `
      max-width: 100%;
      max-height: 100%;
      display: none;
      pointer-events: auto;
    `;

    const placeholder = document.createElement("div");
    placeholder.textContent = "No preview";
    placeholder.style.cssText = `
      color: #666;
      font-style: italic;
      font-size: 12px;
      pointer-events: none;
    `;

    wrapper.appendChild(video);
    wrapper.appendChild(placeholder);

    const previewWidget = node.addDOMWidget(
      "webm_preview",
      "webm_preview",
      wrapper,
      {
        serialize: false,
        hideOnZoom: false,
      },
    );

    previewWidget.computeSize = function (width) {
      const available = Math.max(
        0,
        (node?.size?.[1] ?? 0) - NODE_HEIGHT_PADDING,
      );
      const height = Math.max(MIN_HEIGHT, available);
      this.computedHeight = height;
      return [width, height];
    };

    // Resize node once video dimensions are known
    video.addEventListener("stalled", () =>
      console.warn(
        "[webmPreview] stalled, readyState:",
        video.readyState,
        "networkState:",
        video.networkState,
      ),
    );
    video.addEventListener("waiting", () =>
      console.warn(
        "[webmPreview] waiting, readyState:",
        video.readyState,
        "networkState:",
        video.networkState,
      ),
    );
    video.addEventListener("canplay", () =>
      console.log("[webmPreview] canplay, readyState:", video.readyState),
    );
    video.addEventListener("error", (e) => {
      console.error(
        "[webmPreview] video error:",
        video.error?.code,
        video.error?.message,
        e,
      );
    });
    video.addEventListener("loadedmetadata", () => {
      if (video.videoWidth && video.videoHeight) {
        const ratio = video.videoHeight / video.videoWidth;
        const nodeWidth = node.size[0];
        const targetHeight = Math.round(nodeWidth * ratio);
        const clampedHeight = Math.max(MIN_HEIGHT, Math.min(targetHeight, 600));
        node.setSize([nodeWidth, clampedHeight + NODE_HEIGHT_PADDING]);
        app.graph.setDirtyCanvas(true, false);
      }
    });

    // Track current URL for context menu
    let currentUrl = null;

    node.getExtraMenuOptions = function (_, options) {
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

      const previews = message?.webm_preview;
      if (!previews || previews.length === 0) {
        video.style.display = "none";
        placeholder.style.display = "block";
        return;
      }

      const item = previews[0];
      const streamParams = new URLSearchParams({
        filename: item.filename,
        t: Date.now(),
      });
      const url = api.apiURL(`/webm_preview/stream?${streamParams.toString()}`);

      currentUrl = url;
      video.pause();
      video.src = url;
      video.style.display = "block";
      placeholder.style.display = "none";
      video.play().catch(() => {});
    };

    // Set initial node size
    if ((node?.size?.[1] ?? 0) < MIN_HEIGHT + NODE_HEIGHT_PADDING) {
      node.setSize([node.size?.[0] ?? 240, MIN_HEIGHT + NODE_HEIGHT_PADDING]);
    }
  },
});
