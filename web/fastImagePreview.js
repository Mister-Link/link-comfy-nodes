import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const STYLE_ID = "lc_fast_image_preview_styles";
const PREVIEW_MIN_HEIGHT = 140;
const NODE_HEIGHT_PADDING = 54;

function ensureStyles() {
  if (document.getElementById(STYLE_ID)) {
    return;
  }

  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
    .lc-fast-preview {
      width: 100%;
      height: 100%;
      display: grid;
      align-content: center;
      justify-content: center;
    }
    .lc-fast-preview-empty {
      color: #9a9a9a;
      font-style: italic;
      font-size: 12px;
      padding: 6px;
    }
    .lc-fast-preview-item {
      box-sizing: border-box;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      background: transparent;
      border: none;
      cursor: pointer;
    }
    .lc-fast-preview-item img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
      transition: filter 0.15s ease;
    }
    .lc-fast-preview-item:hover img {
      filter: brightness(1.25);
    }
    .lc-fast-preview-overlay {
      position: absolute;
      inset: 0;
      background: #2a2a2a;
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 5;
      box-sizing: border-box;
    }
    .lc-fast-preview-overlay img {
      max-width: 100%;
      max-height: 100%;
      width: auto;
      height: auto;
      object-fit: contain;
      display: block;
    }
    .lc-fast-preview-overlay button {
      position: absolute;
      top: 8px;
      right: 8px;
      width: 28px;
      height: 28px;
      border-radius: 6px;
      border: none;
      background: rgba(20, 20, 20, 0.8);
      color: #f0f0f0;
      font-size: 18px;
      cursor: pointer;
      line-height: 1;
    }
  `;

  document.head.appendChild(style);
}

function findBestFit(containerW, containerH, itemCount, aspectRatio) {
  let bestLayout = null;
  let bestArea = 0;
  let bestCoverage = 0;

  for (let cols = 1; cols <= itemCount; cols += 1) {
    const rows = Math.ceil(itemCount / cols);
    const maxItemW = containerW / cols;
    const maxItemH = containerH / rows;
    let itemW = maxItemW;
    let itemH = itemW / aspectRatio;
    if (itemH > maxItemH) {
      itemH = maxItemH;
      itemW = itemH * aspectRatio;
    }

    const itemWInt = Math.floor(itemW);
    const itemHInt = Math.floor(itemH);
    const totalWInt = itemWInt * cols;
    const totalHInt = itemHInt * rows;
    if (itemWInt < 1 || itemHInt < 1) {
      continue;
    }
    if (totalWInt > containerW || totalHInt > containerH) {
      continue;
    }

    const area = itemWInt * itemHInt;
    const coverage = (totalWInt / containerW) * (totalHInt / containerH);

    if (coverage > bestCoverage + 0.001) {
      bestCoverage = coverage;
      bestArea = area;
      bestLayout = { cols, rows, itemW: itemWInt, itemH: itemHInt };
      continue;
    }

    if (Math.abs(coverage - bestCoverage) <= 0.001 && area > bestArea) {
      bestArea = area;
      bestLayout = { cols, rows, itemW: itemWInt, itemH: itemHInt };
    }
  }

  return bestLayout;
}

function buildImageUrl(data) {
  const params = new URLSearchParams({
    filename: data.filename ?? "",
    subfolder: data.subfolder ?? "",
    type: data.type ?? "temp",
  });
  return api.apiURL(`/view?${params.toString()}`);
}

function buildFullImageUrl(data) {
  const params = new URLSearchParams({
    filename: data.full_filename ?? data.filename ?? "",
    subfolder: data.subfolder ?? "",
    type: data.type ?? "temp",
  });
  return api.apiURL(`/view?${params.toString()}`);
}

function scheduleLayout(state) {
  if (state.layoutRaf) {
    cancelAnimationFrame(state.layoutRaf);
  }
  state.layoutRaf = requestAnimationFrame(() => updateLayout(state));
}

function updateAspectRatio(state) {
  const ratios = state.items
    .map((item) => item.ratio)
    .filter((ratio) => Number.isFinite(ratio) && ratio > 0);
  if (!ratios.length) {
    state.aspectRatio = 1;
    return;
  }
  const sum = ratios.reduce((acc, ratio) => acc + ratio, 0);
  state.aspectRatio = sum / ratios.length;
}

function updateLayout(state) {
  const container = state.container;
  if (!container) {
    return;
  }

  const rect = container.getBoundingClientRect();
  const styles = getComputedStyle(container);
  const rawW = container.clientWidth || rect.width;
  const rawH = container.clientHeight || rect.height;
  const paddingX =
    parseFloat(styles.paddingLeft || "0") +
    parseFloat(styles.paddingRight || "0");
  const paddingY =
    parseFloat(styles.paddingTop || "0") +
    parseFloat(styles.paddingBottom || "0");
  const containerW = Math.max(0, rawW - paddingX);
  const containerH = Math.max(0, rawH - paddingY);

  if (!containerW || !containerH) {
    return;
  }

  const itemCount = state.items.length;
  if (!itemCount) {
    return;
  }

  const layout = findBestFit(
    containerW,
    containerH,
    itemCount,
    state.aspectRatio || 1,
  );

  const finalLayout =
    layout ||
    (containerW >= 1 && containerH >= 1
      ? {
          cols: 1,
          rows: itemCount,
          itemW: Math.max(1, Math.floor(containerW)),
          itemH: Math.max(1, Math.floor(containerH / itemCount)),
        }
      : null);

  if (!finalLayout) {
    return;
  }

  const itemW = Math.max(1, finalLayout.itemW);
  const itemH = Math.max(1, finalLayout.itemH);

  container.style.gridTemplateColumns = `repeat(${finalLayout.cols}, ${itemW}px)`;
  container.style.gridAutoRows = `${itemH}px`;

  state.items.forEach((item) => {
    item.wrapper.style.width = `${itemW}px`;
    item.wrapper.style.height = `${itemH}px`;
  });
}

function resetPreview(state, message) {
  state.container.innerHTML = "";
}

function preloadImages(images, startIndex) {
  // Preload images starting from startIndex, prioritizing nearby images
  const preloadOrder = [];

  // Add current image first
  preloadOrder.push(startIndex);

  // Add alternating next/prev images
  for (let offset = 1; offset < images.length; offset++) {
    const nextIdx = (startIndex + offset) % images.length;
    const prevIdx = (startIndex - offset + images.length) % images.length;

    if (nextIdx !== startIndex && !preloadOrder.includes(nextIdx)) {
      preloadOrder.push(nextIdx);
    }
    if (prevIdx !== startIndex && !preloadOrder.includes(prevIdx)) {
      preloadOrder.push(prevIdx);
    }
  }

  // Preload in order
  preloadOrder.forEach((idx) => {
    const img = new Image();
    img.src = buildFullImageUrl(images[idx]);
  });
}

function showOverlay(
  state,
  container,
  images,
  currentIndex,
  isFirstOpen = false,
) {
  if (state.overlay) {
    state.overlay.remove();
    state.overlay = null;
  }

  if (state.overlayKeyHandler) {
    document.removeEventListener("keydown", state.overlayKeyHandler, true);
    state.overlayKeyHandler = null;
  }

  // Preload images on first open
  if (isFirstOpen) {
    preloadImages(images, currentIndex);
  }

  const overlay = document.createElement("div");
  overlay.className = "lc-fast-preview-overlay";

  const fullImg = document.createElement("img");
  fullImg.src = buildFullImageUrl(images[currentIndex]);
  fullImg.alt = "";

  const closeButton = document.createElement("button");
  closeButton.type = "button";
  closeButton.textContent = "×";

  const closeOverlay = () => {
    overlay.remove();
    state.overlay = null;
    if (state.overlayKeyHandler) {
      document.removeEventListener("keydown", state.overlayKeyHandler, true);
      state.overlayKeyHandler = null;
    }
  };

  closeButton.addEventListener("click", (e) => {
    e.stopPropagation();
    closeOverlay();
  });

  overlay.addEventListener("click", () => {
    closeOverlay();
  });

  state.overlayKeyHandler = (e) => {
    if (e.key === "ArrowLeft") {
      e.preventDefault();
      e.stopPropagation();
      e.stopImmediatePropagation();
      const newIndex = (currentIndex - 1 + images.length) % images.length;
      showOverlay(state, container, images, newIndex, false);
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      e.stopPropagation();
      e.stopImmediatePropagation();
      const newIndex = (currentIndex + 1) % images.length;
      showOverlay(state, container, images, newIndex, false);
    } else if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation();
      closeOverlay();
    }
  };

  document.addEventListener("keydown", state.overlayKeyHandler, true);

  overlay.appendChild(fullImg);
  overlay.appendChild(closeButton);
  container.appendChild(overlay);
  state.overlay = overlay;
}

app.registerExtension({
  name: "FastImagePreview.Render",

  async nodeCreated(node) {
    if (node.comfyClass !== "Fast Image Preview") {
      return;
    }

    ensureStyles();

    const container = document.createElement("div");
    container.className = "lc-fast-preview";
    wrapper.appendChild(container);

    const previewWidget = node.addDOMWidget(
      "fast_preview",
      "fast_preview",
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
      const height = Math.max(PREVIEW_MIN_HEIGHT, available);
      this.computedHeight = height;
      return [width, height];
    };

    const minNodeHeight = PREVIEW_MIN_HEIGHT + NODE_HEIGHT_PADDING;
    if ((node?.size?.[1] ?? 0) < minNodeHeight) {
      node.setSize([node.size?.[0] ?? 240, minNodeHeight]);
    }

    const state = {
      container,
      items: [],
      aspectRatio: 1,
      layoutRaf: null,
      resizeObserver: null,
      overlay: null,
    };
    node._fastPreviewState = state;

    if (typeof ResizeObserver !== "undefined") {
      state.resizeObserver = new ResizeObserver(() => scheduleLayout(state));
      state.resizeObserver.observe(container);
    }

    const originalOnResize = node.onResize;
    node.onResize = function () {
      if (originalOnResize) {
        originalOnResize.apply(this, arguments);
      }
      scheduleLayout(state);
    };

    resetPreview(state);

    const originalOnExecuted = node.onExecuted;
    node.onExecuted = function (message) {
      if (originalOnExecuted) {
        originalOnExecuted.apply(this, arguments);
      }

      const images = message?.fast_images ?? [];
      state.items = [];
      container.innerHTML = "";
      state.overlay = null;

      if (!images.length) {
        resetPreview(state);
        return;
      }

      images.forEach((imageData, index) => {
        const wrapperEl = document.createElement("div");
        wrapperEl.className = "lc-fast-preview-item";

        const img = document.createElement("img");
        img.loading = "lazy";
        img.src = buildImageUrl(imageData);

        const item = { wrapper: wrapperEl, img, ratio: null, imageData };
        img.addEventListener("load", () => {
          if (img.naturalWidth && img.naturalHeight) {
            item.ratio = img.naturalWidth / img.naturalHeight;
            updateAspectRatio(state);
            scheduleLayout(state);
          }
        });

        wrapperEl.addEventListener("click", (event) => {
          event.stopPropagation();
          showOverlay(state, container, images, index, true);
        });

        wrapperEl.appendChild(img);
        container.appendChild(wrapperEl);
        state.items.push(item);
      });

      scheduleLayout(state);
    };
  },
});
