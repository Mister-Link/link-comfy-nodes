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
    .lc-fast-preview-wrapper {
      pointer-events: none;
    }
    .dom-widget:has(.lc-fast-preview-wrapper) {
      pointer-events: none !important;
    }
    .lc-fast-preview {
      box-sizing: border-box;
      width: calc(100% + .6em);
      height: calc(100% + 2em);
      margin-left: -.3em;
      margin-top: -.75em;
      display: grid;
      grid-auto-flow: row;
      align-content: center;
      justify-content: center;
      gap: 0;
      overflow: hidden;
      position: relative;
      pointer-events: none;
    }
    .lc-fast-preview-info {
      color: #c0c0c0;
      font-size: 8px;
      padding: 2px 4px;
      line-height: 1.2;
      display: flex;
      justify-content: center;
      align-items: center;
      position: relative;
    }
    .lc-fast-preview-info-center {
      flex-shrink: 0;
    }
    .lc-fast-preview-info-center.vary {
      color: #ff0000;
    }
    .lc-fast-preview-info-right {
      position: absolute;
      right: 0px;
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
      pointer-events: auto;
    }
    .lc-fast-preview-item img {
      width: 100%;
      height: 100%;
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
      pointer-events: auto;
    }
    .lc-fast-preview-overlay img {
      max-width: 100%;
      max-height: 100%;
      width: auto;
      height: auto;
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

let pointerEventsEnforced = false;

function enforcePointerEventsNone(el) {
  if (!el) {
    return;
  }

  const apply = () => {
    el.style.setProperty("pointer-events", "none", "important");
  };

  apply();

  if (el._lcPointerEventsObserver) {
    return;
  }

  const observer = new MutationObserver(() => apply());
  observer.observe(el, {
    attributes: true,
    attributeFilter: ["style", "class"],
  });

  el._lcPointerEventsObserver = observer;
}

function initPointerEventsEnforcement() {
  if (pointerEventsEnforced) {
    return;
  }
  pointerEventsEnforced = true;

  const applyForWrapper = (wrapper) => {
    const parent = wrapper?.parentElement;
    if (parent && parent.classList.contains("dom-widget")) {
      enforcePointerEventsNone(parent);
    }
  };

  document
    .querySelectorAll(".lc-fast-preview-wrapper")
    .forEach(applyForWrapper);

  const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      mutation.addedNodes.forEach((node) => {
        if (!(node instanceof Element)) {
          return;
        }
        if (node.classList?.contains("lc-fast-preview-wrapper")) {
          applyForWrapper(node);
          return;
        }
        if (node.classList?.contains("dom-widget")) {
          const wrapper = node.querySelector(".lc-fast-preview-wrapper");
          if (wrapper) {
            applyForWrapper(wrapper);
          }
          return;
        }
        const wrappers = node.querySelectorAll?.(".lc-fast-preview-wrapper");
        if (wrappers?.length) {
          wrappers.forEach(applyForWrapper);
        }
      });
    });
  });

  observer.observe(document.body, { childList: true, subtree: true });
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

function resetPreview(state) {
  state.container.innerHTML = "";
}

function updateInfoDisplay(state, currentIndex = null) {
  const { infoDiv, imageDimensions } = state;

  if (!imageDimensions.length) {
    infoDiv.innerHTML = "";
    return;
  }

  if (currentIndex !== null) {
    const dim = imageDimensions[currentIndex];
    if (dim) {
      const centerSpan = document.createElement("span");
      centerSpan.className = "lc-fast-preview-info-center";
      centerSpan.textContent = `${dim.width}x${dim.height}`;

      const rightSpan = document.createElement("span");
      rightSpan.className = "lc-fast-preview-info-right";
      rightSpan.textContent = `${currentIndex + 1}/${imageDimensions.length}`;

      infoDiv.innerHTML = "";
      infoDiv.appendChild(centerSpan);
      infoDiv.appendChild(rightSpan);
    }
  } else {
    const firstDim = imageDimensions[0];
    const allSame = imageDimensions.every(
      (dim) => dim.width === firstDim.width && dim.height === firstDim.height,
    );

    const centerSpan = document.createElement("span");
    centerSpan.className = "lc-fast-preview-info-center";

    if (allSame) {
      centerSpan.textContent = `${firstDim.width}x${firstDim.height}`;
    } else {
      centerSpan.textContent = "dimensions vary";
      centerSpan.classList.add("vary");
    }

    const rightSpan = document.createElement("span");
    rightSpan.className = "lc-fast-preview-info-right";
    rightSpan.textContent = `#: ${imageDimensions.length}`;

    infoDiv.innerHTML = "";
    infoDiv.appendChild(centerSpan);
    infoDiv.appendChild(rightSpan);
  }
}

let isMiddleMouseForwarding = false;

function forwardEventToGraphCanvas(e) {
  const canvas = document.getElementById("graph-canvas");
  if (!canvas) return;

  e.preventDefault();
  e.stopPropagation();

  canvas.focus?.();

  if (e.type === "wheel") {
    canvas.dispatchEvent(
      new WheelEvent("wheel", {
        deltaX: e.deltaX,
        deltaY: e.deltaY,
        deltaZ: e.deltaZ,
        deltaMode: e.deltaMode,

        clientX: e.clientX,
        clientY: e.clientY,
        screenX: e.screenX,
        screenY: e.screenY,

        ctrlKey: e.ctrlKey,
        shiftKey: e.shiftKey,
        altKey: e.altKey,
        metaKey: e.metaKey,

        bubbles: true,
        cancelable: true,
      }),
    );
    return;
  }

  if ((e.type === "pointerdown" || e.type === "mousedown") && e.button === 1) {
    isMiddleMouseForwarding = true;
  }

  if ((e.type === "pointerup" || e.type === "mouseup") && e.button === 1) {
    isMiddleMouseForwarding = false;
  }

  canvas.dispatchEvent(
    new PointerEvent(e.type, {
      pointerId: e.pointerId,
      pointerType: e.pointerType,
      isPrimary: e.isPrimary,
      button: e.button,
      buttons: e.buttons,
      clientX: e.clientX,
      clientY: e.clientY,
      screenX: e.screenX,
      screenY: e.screenY,

      ctrlKey: e.ctrlKey,
      shiftKey: e.shiftKey,
      altKey: e.altKey,
      metaKey: e.metaKey,

      bubbles: true,
      cancelable: true,
    }),
  );
}

function attachNavigationForwarding(el) {
  el.addEventListener("wheel", forwardEventToGraphCanvas, { passive: false });

  const pointerEvents = [
    "pointerdown",
    "pointermove",
    "pointerup",
    "pointercancel",
  ];

  pointerEvents.forEach((type) => {
    el.addEventListener(type, (e) => {
      if (type === "pointerdown" && e.button === 1) {
        forwardEventToGraphCanvas(e);
      } else if (type === "pointerup" && e.button === 1) {
        forwardEventToGraphCanvas(e);
      } else if (type === "pointermove" && isMiddleMouseForwarding) {
        forwardEventToGraphCanvas(e);
      } else if (type === "pointercancel" && isMiddleMouseForwarding) {
        forwardEventToGraphCanvas(e);
      }
    });
  });
}

function preloadAllImages(images, cache, state) {
  images.forEach((imageData, idx) => {
    const img = new Image();
    img.decode = img.decode || (() => Promise.resolve());
    img.src = buildFullImageUrl(imageData);

    cache[idx] = img;

    img.onload = () => {
      state.imageDimensions[idx] = {
        width: img.naturalWidth,
        height: img.naturalHeight,
      };
    };

    img.decode().catch(() => {});
  });
}

function showOverlay(
  state,
  container,
  images,
  currentIndex,
  isFirstOpen = false,
  node = null,
) {
  if (state.overlay) {
    state.overlay.remove();
    state.overlay = null;
  }

  state.overlayKeyHandler = null;

  if (isFirstOpen && !state.imagesPreloaded) {
    preloadAllImages(images, state.imageCache, state);
    state.imagesPreloaded = true;
  }

  updateInfoDisplay(state, currentIndex);

  const overlay = document.createElement("div");
  overlay.className = "lc-fast-preview-overlay";
  overlay.tabIndex = 0;

  const fullImg = state.imageCache[currentIndex]
    ? state.imageCache[currentIndex].cloneNode()
    : document.createElement("img");

  if (!state.imageCache[currentIndex]) {
    fullImg.src = buildFullImageUrl(images[currentIndex]);
  }
  fullImg.alt = "";

  const closeOverlay = () => {
    overlay.remove();
    state.overlay = null;
    state.overlayKeyHandler = null;
    updateInfoDisplay(state);
  };

  overlay.addEventListener("click", (event) => {
    if (event.button !== 0) return;
    closeOverlay();
  });

  attachNavigationForwarding(overlay);

  state.overlayKeyHandler = (e) => {
    if (e.key === "ArrowLeft") {
      e.preventDefault();
      e.stopPropagation();
      e.stopImmediatePropagation();
      const newIndex = (currentIndex - 1 + images.length) % images.length;
      showOverlay(state, container, images, newIndex, false, node);
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      e.stopPropagation();
      e.stopImmediatePropagation();
      const newIndex = (currentIndex + 1) % images.length;
      showOverlay(state, container, images, newIndex, false, node);
    } else if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation();
      e.stopImmediatePropagation();
      closeOverlay();
    }
  };

  overlay.addEventListener("keydown", state.overlayKeyHandler);

  overlay.appendChild(fullImg);
  container.appendChild(overlay);
  requestAnimationFrame(() => {
    overlay.focus({ preventScroll: true });
  });
  state.overlay = overlay;
}

app.registerExtension({
  name: "FastImagePreview.Render",

  async nodeCreated(node) {
    if (node.comfyClass !== "Fast Image Preview") {
      return;
    }

    ensureStyles();
    initPointerEventsEnforcement();

    const wrapper = document.createElement("div");
    wrapper.className = "lc-fast-preview-wrapper";

    const container = document.createElement("div");
    container.className = "lc-fast-preview";
    wrapper.appendChild(container);

    const infoDiv = document.createElement("div");
    infoDiv.className = "lc-fast-preview-info";
    wrapper.appendChild(infoDiv);

    const previewWidget = node.addDOMWidget(
      "fast_preview",
      "fast_preview",
      wrapper,
      {
        serialize: false,
        hideOnZoom: false,
      },
    );

    if (previewWidget.element && previewWidget.element.parentElement) {
      enforcePointerEventsNone(previewWidget.element.parentElement);
    }
    requestAnimationFrame(() => {
      const parent = wrapper.closest?.(".dom-widget");
      if (parent) {
        enforcePointerEventsNone(parent);
      }
    });

    console.log(previewWidget);
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
      infoDiv,
      items: [],
      aspectRatio: 1,
      layoutRaf: null,
      resizeObserver: null,
      overlay: null,
      imagesPreloaded: false,
      imageCache: {},
      imageDimensions: [],
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
      state.imagesPreloaded = false;
      state.imageCache = {};
      state.imageDimensions = [];

      if (!images.length) {
        resetPreview(state);
        updateInfoDisplay(state);
        return;
      }

      const isSingleFrame = images.length === 1;

      images.forEach((imageData, index) => {
        const wrapperEl = document.createElement("div");
        wrapperEl.className = "lc-fast-preview-item";

        const img = document.createElement("img");
        img.loading = "lazy";
        img.src = isSingleFrame
          ? buildFullImageUrl(imageData)
          : buildImageUrl(imageData);

        if (imageData.width && imageData.height) {
          state.imageDimensions[index] = {
            width: imageData.width,
            height: imageData.height,
          };
        }

        const item = { wrapper: wrapperEl, img, ratio: null, imageData };
        img.addEventListener("load", () => {
          if (img.naturalWidth && img.naturalHeight) {
            item.ratio = img.naturalWidth / img.naturalHeight;
            updateAspectRatio(state);
            scheduleLayout(state);
          }
        });

        if (!isSingleFrame) {
          wrapperEl.addEventListener("click", (event) => {
            if (event.button !== 0) return;
            event.stopPropagation();
            showOverlay(state, container, images, index, true, node);
          });

          attachNavigationForwarding(wrapperEl);
        } else {
          wrapperEl.style.pointerEvents = "none";
          wrapperEl.style.cursor = "default";
        }

        wrapperEl.appendChild(img);
        container.appendChild(wrapperEl);
        state.items.push(item);
      });

      if (state.imageDimensions.filter((d) => d).length === images.length) {
        updateInfoDisplay(state);
      }

      scheduleLayout(state);
    };
  },
});
