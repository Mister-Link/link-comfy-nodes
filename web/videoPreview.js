import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function openMaskEditor(node, previewWidget) {
  // Create dialog overlay
  const overlay = document.createElement("div");
  overlay.style.cssText = `
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.8);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 10000;
  `;

  // Create dialog container
  const dialog = document.createElement("div");
  dialog.style.cssText = `
    background: rgb(43, 43, 43);
    border-radius: 8px;
    padding: 20px;
    max-width: 90vw;
    max-height: 90vh;
    display: flex;
    flex-direction: column;
    gap: 15px;
    color: #fff;
  `;

  // Title
  const title = document.createElement("h3");
  title.textContent = "Set Mask Region";
  title.style.cssText = "margin: 0; font-size: 18px;";
  dialog.appendChild(title);

  // Instructions
  const instructions = document.createElement("p");
  instructions.textContent = "Click and drag to select the mask region.";
  instructions.style.cssText = "margin: 0; font-size: 14px; color: #aaa;";
  dialog.appendChild(instructions);

  // Canvas container
  const canvasContainer = document.createElement("div");
  canvasContainer.style.cssText = `
    position: relative;
    max-width: 100%;
    max-height: 60vh;
    overflow: hidden;
    background: rgb(43, 43, 43);
    border: 2px solid #444;
    display: flex;
    align-items: center;
    justify-content: center;
  `;

  // Create canvas for video + mask overlay
  // Note: Don't add ComfyUI event forwarding to this canvas - it's for drawing
  const canvas = document.createElement("canvas");
  canvas.style.cssText = `
    display: block;
    cursor: crosshair;
    max-width: 100%;
    max-height: 60vh;
    object-fit: contain;
  `;

  // Note: Do not stop propagation here - let the drawing handlers below handle it

  canvasContainer.appendChild(canvas);

  // Add canvas container to dialog
  dialog.appendChild(canvasContainer);

  // Mask selection state
  let maskRect = node.maskRegion || null;
  let isDrawing = false;
  let isDragging = false;
  let isResizing = false;
  let resizeHandle = null; // 'nw', 'ne', 'sw', 'se', 'n', 's', 'e', 'w'
  let startX = 0;
  let startY = 0;
  let dragStartRect = null;
  let animationInterval = null;
  let dialogFrameIndex = 0; // Separate frame index for dialog to avoid interference
  const HANDLE_SIZE = 12; // Size of corner/edge handles for resizing

  const toCanvasCoords = (event) => {
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((event.clientX - rect.left) * canvas.width) / rect.width,
      y: ((event.clientY - rect.top) * canvas.height) / rect.height,
    };
  };

  const getCursorForHandle = (handle) => {
    const cursors = {
      nw: "nw-resize",
      ne: "ne-resize",
      sw: "sw-resize",
      se: "se-resize",
      n: "n-resize",
      s: "s-resize",
      w: "w-resize",
      e: "e-resize",
    };
    return cursors[handle] || "crosshair";
  };

  // Helper function to check if point is in a handle
  const getHandleAt = (x, y) => {
    if (!maskRect || maskRect.width === 0 || maskRect.height === 0) return null;

    const handles = [
      { name: "nw", x: maskRect.x, y: maskRect.y },
      { name: "ne", x: maskRect.x + maskRect.width, y: maskRect.y },
      { name: "sw", x: maskRect.x, y: maskRect.y + maskRect.height },
      {
        name: "se",
        x: maskRect.x + maskRect.width,
        y: maskRect.y + maskRect.height,
      },
      { name: "n", x: maskRect.x + maskRect.width / 2, y: maskRect.y },
      {
        name: "s",
        x: maskRect.x + maskRect.width / 2,
        y: maskRect.y + maskRect.height,
      },
      { name: "w", x: maskRect.x, y: maskRect.y + maskRect.height / 2 },
      {
        name: "e",
        x: maskRect.x + maskRect.width,
        y: maskRect.y + maskRect.height / 2,
      },
    ];

    for (const handle of handles) {
      if (
        Math.abs(x - handle.x) <= HANDLE_SIZE &&
        Math.abs(y - handle.y) <= HANDLE_SIZE
      ) {
        return handle.name;
      }
    }
    return null;
  };

  // Helper function to check if point is inside mask rect
  const isInsideMask = (x, y) => {
    if (!maskRect) return false;
    return (
      x >= maskRect.x &&
      x <= maskRect.x + maskRect.width &&
      y >= maskRect.y &&
      y <= maskRect.y + maskRect.height
    );
  };

  // Draw current frame with mask overlay
  const drawCanvas = () => {
    if (!previewWidget.frames || previewWidget.frames.length === 0) return;

    const frameData =
      previewWidget.frames[dialogFrameIndex % previewWidget.frames.length];
    if (!frameData) return;

    canvas.width = frameData.width;
    canvas.height = frameData.height;

    const ctx = canvas.getContext("2d");
    ctx.putImageData(frameData.imageData, 0, 0);

    // Draw mask overlay only if we have a valid mask
    if (maskRect && maskRect.width > 0 && maskRect.height > 0) {
      if (!isDrawing && !isDragging && !isResizing) {
        // Only show the overlay when NOT actively editing
        // Draw soft dark overlay on the entire canvas (outside the selection)
        ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Clear the selected area to show the original video with no tint
        ctx.clearRect(maskRect.x, maskRect.y, maskRect.width, maskRect.height);

        // Redraw the original video content in the cleared area
        const frameData =
          previewWidget.frames[dialogFrameIndex % previewWidget.frames.length];
        if (frameData) {
          ctx.putImageData(
            frameData.imageData,
            0,
            0,
            maskRect.x,
            maskRect.y,
            maskRect.width,
            maskRect.height,
          );
        }
      }

      // Always draw red border around selection
      ctx.strokeStyle = "#ff4444";
      ctx.lineWidth = 2;
      ctx.strokeRect(maskRect.x, maskRect.y, maskRect.width, maskRect.height);
      ctx.shadowBlur = 0;

      // Draw resize handles (corners and edges)
      if (!isDrawing) {
        ctx.fillStyle = "#ffffff";
        ctx.strokeStyle = "#ff4444";
        ctx.lineWidth = 2;

        const handles = [
          { x: maskRect.x, y: maskRect.y }, // nw
          { x: maskRect.x + maskRect.width, y: maskRect.y }, // ne
          { x: maskRect.x, y: maskRect.y + maskRect.height }, // sw
          { x: maskRect.x + maskRect.width, y: maskRect.y + maskRect.height }, // se
          { x: maskRect.x + maskRect.width / 2, y: maskRect.y }, // n
          {
            x: maskRect.x + maskRect.width / 2,
            y: maskRect.y + maskRect.height,
          }, // s
          { x: maskRect.x, y: maskRect.y + maskRect.height / 2 }, // w
          {
            x: maskRect.x + maskRect.width,
            y: maskRect.y + maskRect.height / 2,
          }, // e
        ];

        for (const handle of handles) {
          ctx.fillRect(handle.x - 4, handle.y - 4, 8, 8);
          ctx.strokeRect(handle.x - 4, handle.y - 4, 8, 8);
        }
      }
    }
  };

  // Start animation
  const startAnimation = () => {
    if (animationInterval) return;
    const frameDuration = previewWidget.frameDuration || 100;
    animationInterval = setInterval(() => {
      dialogFrameIndex = (dialogFrameIndex + 1) % previewWidget.frames.length;
      drawCanvas();
    }, frameDuration);
  };

  // Stop animation
  const stopAnimation = () => {
    if (animationInterval) {
      clearInterval(animationInterval);
      animationInterval = null;
    }
  };

  // Mouse event handlers
  const handlePointerDown = (e) => {
    e.preventDefault();
    e.stopPropagation();
    const coords = toCanvasCoords(e);
    startX = coords.x;
    startY = coords.y;

    // Check if clicking on a handle (for resizing)
    const handle = getHandleAt(startX, startY);
    if (handle) {
      isResizing = true;
      resizeHandle = handle;
      dragStartRect = { ...maskRect };
      canvas.style.cursor = getCursorForHandle(handle);
    }
    // Check if clicking inside mask (for dragging)
    else if (isInsideMask(startX, startY)) {
      isDragging = true;
      dragStartRect = { ...maskRect };
      canvas.style.cursor = "move";
    }
    // Otherwise start drawing a new mask
    else {
      isDrawing = true;
      maskRect = { x: startX, y: startY, width: 0, height: 0 };
    }

    // Attach document-level listeners so dragging works even outside the canvas
    document.addEventListener("pointermove", handlePointerMove);
    document.addEventListener("pointerup", handlePointerUp);

    drawCanvas();
  };

  canvas.addEventListener("pointerdown", handlePointerDown);

  const handlePointerMove = (e) => {
    const coords = toCanvasCoords(e);
    let currentX = coords.x;
    let currentY = coords.y;

    // Don't clamp coordinates during active operations - let them extend beyond canvas
    // This allows smooth dragging when cursor goes outside the video area

    // Handle drawing new mask
    if (isDrawing) {
      e.preventDefault();
      e.stopPropagation();
      // Clamp for drawing to keep rectangle within bounds
      const clampedX = Math.max(0, Math.min(canvas.width, currentX));
      const clampedY = Math.max(0, Math.min(canvas.height, currentY));
      maskRect = {
        x: Math.min(startX, clampedX),
        y: Math.min(startY, clampedY),
        width: Math.abs(clampedX - startX),
        height: Math.abs(clampedY - startY),
      };
      drawCanvas();
    }
    // Handle dragging mask to translate
    else if (isDragging) {
      e.preventDefault();
      e.stopPropagation();
      // Use unclamped coordinates to calculate delta for smooth dragging
      const dx = currentX - startX;
      const dy = currentY - startY;

      maskRect = {
        x: Math.max(
          0,
          Math.min(canvas.width - dragStartRect.width, dragStartRect.x + dx),
        ),
        y: Math.max(
          0,
          Math.min(canvas.height - dragStartRect.height, dragStartRect.y + dy),
        ),
        width: dragStartRect.width,
        height: dragStartRect.height,
      };
      drawCanvas();
    }
    // Handle resizing mask
    else if (isResizing) {
      e.preventDefault();
      e.stopPropagation();
      // Use unclamped coordinates to calculate delta for smooth resizing
      const dx = currentX - startX;
      const dy = currentY - startY;

      let newRect = { ...dragStartRect };

      // Apply changes based on which handle is being dragged
      if (resizeHandle.includes("n")) {
        newRect.y = dragStartRect.y + dy;
        newRect.height = dragStartRect.height - dy;
      }
      if (resizeHandle.includes("s")) {
        newRect.height = dragStartRect.height + dy;
      }
      if (resizeHandle.includes("w")) {
        newRect.x = dragStartRect.x + dx;
        newRect.width = dragStartRect.width - dx;
      }
      if (resizeHandle.includes("e")) {
        newRect.width = dragStartRect.width + dx;
      }

      // Ensure minimum size and bounds
      if (newRect.width < 10) newRect.width = 10;
      if (newRect.height < 10) newRect.height = 10;
      newRect.x = Math.max(
        0,
        Math.min(canvas.width - newRect.width, newRect.x),
      );
      newRect.y = Math.max(
        0,
        Math.min(canvas.height - newRect.height, newRect.y),
      );

      maskRect = newRect;
      drawCanvas();
    }
    // Update cursor when hovering over handles or mask
    else if (maskRect) {
      e.preventDefault();
      e.stopPropagation();
      // Clamp for cursor detection since handles are within canvas
      const clampedX = Math.max(0, Math.min(canvas.width, currentX));
      const clampedY = Math.max(0, Math.min(canvas.height, currentY));
      const handle = getHandleAt(clampedX, clampedY);
      if (handle) {
        canvas.style.cursor = getCursorForHandle(handle);
      } else if (isInsideMask(clampedX, clampedY)) {
        canvas.style.cursor = "move";
      } else {
        canvas.style.cursor = "crosshair";
      }
    }
  };

  const handlePointerUp = (e) => {
    if (isDrawing || isDragging || isResizing) {
      e.preventDefault();
      e.stopPropagation();
    }
    isDrawing = false;
    isDragging = false;
    isResizing = false;
    resizeHandle = null;
    dragStartRect = null;
    canvas.style.cursor = "crosshair";

    // Remove document listeners
    document.removeEventListener("pointermove", handlePointerMove);
    document.removeEventListener("pointerup", handlePointerUp);
  };

  canvas.addEventListener("pointermove", handlePointerMove);
  canvas.addEventListener("pointerup", handlePointerUp);

  // Prevent context menu
  canvas.addEventListener("contextmenu", (e) => {
    e.preventDefault();
    e.stopPropagation();
  });

  // Buttons container
  const buttonsContainer = document.createElement("div");
  buttonsContainer.style.cssText =
    "display: flex; gap: 10px; justify-content: flex-end;";

  const createActionButton = (label, background, onClick) => {
    const button = document.createElement("button");
    button.textContent = label;
    button.style.cssText = `
      padding: 8px 16px;
      background: ${background};
      color: #fff;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-size: 14px;
    `;
    if (onClick) {
      button.addEventListener("click", onClick);
    }
    return button;
  };

  // Clear button
  const clearButton = createActionButton("Clear Mask", "#555", () => {
    maskRect = null;
    drawCanvas();
  });
  buttonsContainer.appendChild(clearButton);

  // Cancel button
  const cancelButton = createActionButton("Cancel", "#555", null);
  buttonsContainer.appendChild(cancelButton);

  // Apply button
  const applyButton = createActionButton("Apply", "#0066cc", null);
  buttonsContainer.appendChild(applyButton);

  dialog.appendChild(buttonsContainer);
  overlay.appendChild(dialog);
  document.body.appendChild(overlay);

  // Initial draw and start animation
  drawCanvas();
  startAnimation();

  // Helper function to close dialog
  const closeDialog = () => {
    stopAnimation();
    document.body.removeChild(overlay);
  };

  cancelButton.addEventListener("click", closeDialog);

  applyButton.addEventListener("click", async () => {
    // Store mask region on node
    node.maskRegion = maskRect;

    // Send mask data to backend
    if (maskRect) {
      const videoWidget = node.widgets?.find((w) => w.name === "video");
      if (videoWidget && videoWidget.value) {
        try {
          const response = await api.fetchApi("/videomaskeditor/setmask", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              node_id: node.id,
              video: videoWidget.value,
              mask_region: maskRect,
            }),
          });

          if (!response.ok) {
            console.error(
              "[VideoMaskEditor] Failed to set mask:",
              await response.text(),
            );
          } else {
            console.log("[VideoMaskEditor] Mask set successfully");
          }
        } catch (error) {
          console.error("[VideoMaskEditor] Error setting mask:", error);
        }
      }
    }

    closeDialog();
  });

  // Don't allow closing by clicking outside - user must click Cancel or Apply
}

function chainCallback(obj, eventName, callback) {
  const orig = obj[eventName];
  obj[eventName] = function (...args) {
    const r = orig ? orig.apply(this, args) : undefined;
    return callback.apply(this, [r, ...args]);
  };
}

app.registerExtension({
  name: "VideoMaskEditor.Preview",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name !== "VideoMaskEditor") {
      return;
    }

    // Add preview widget
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
      const previewNode = this;

      // Clear any stale mask data for this node
      api
        .fetchApi("/videomaskeditor/clearmask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ node_id: previewNode.id }),
        })
        .catch((err) =>
          console.error("[VideoMaskEditor] Failed to clear mask:", err),
        );

      // Create preview container
      const element = document.createElement("div");
      const previewWidget = this.addDOMWidget(
        "videopreview",
        "preview",
        element,
        {
          serialize: false,
          hideOnZoom: false,
          getValue() {
            return element.value;
          },
          setValue(v) {
            element.value = v;
          },
        },
      );

      previewWidget.computeSize = function (width) {
        if (this.aspectRatio && this.parentEl.style.display !== "none") {
          let height = (previewNode.size[0] - 20) / this.aspectRatio + 10;
          if (!(height > 0)) {
            height = 0;
          }
          this.computedHeight = height + 10;
          return [width, height];
        }
        return [width, -4];
      };

      previewWidget.value = { hidden: false, paused: false, params: {} };

      previewWidget.parentEl = document.createElement("div");
      previewWidget.parentEl.className = "vhs_preview";
      previewWidget.parentEl.style.cssText = `
        width: 100%;
        background-color: #000;
        margin-bottom: 8px;
        display: none;
      `;
      element.appendChild(previewWidget.parentEl);

      previewWidget.canvasEl = document.createElement("canvas");
      previewWidget.canvasEl.style.width = "100%";
      previewWidget.canvasEl.style.height = "auto";
      previewWidget.canvasEl.style.display = "block";

      // Forward all mouse events to ComfyUI canvas to enable proper context menu and interactions
      previewWidget.canvasEl.addEventListener(
        "contextmenu",
        (e) => {
          e.preventDefault();
          return app.canvas._mousedown_callback(e);
        },
        true,
      );

      previewWidget.canvasEl.addEventListener(
        "pointerdown",
        (e) => {
          e.preventDefault();
          return app.canvas._mousedown_callback(e);
        },
        true,
      );

      previewWidget.canvasEl.addEventListener(
        "mousewheel",
        (e) => {
          e.preventDefault();
          return app.canvas._mousewheel_callback(e);
        },
        true,
      );

      previewWidget.canvasEl.addEventListener(
        "pointermove",
        (e) => {
          e.preventDefault();
          return app.canvas._mousemove_callback(e);
        },
        true,
      );

      previewWidget.canvasEl.addEventListener(
        "pointerup",
        (e) => {
          e.preventDefault();
          return app.canvas._mouseup_callback(e);
        },
        true,
      );

      previewWidget.parentEl.appendChild(previewWidget.canvasEl);

      previewWidget.frames = [];
      previewWidget.frameIndex = 0;
      previewWidget.playInterval = null;
      previewWidget.frameDuration = 100;

      const drawFrame = (frameIndex) => {
        if (!previewWidget.frames || previewWidget.frames.length === 0) return;

        const idx = frameIndex % previewWidget.frames.length;
        const frameData = previewWidget.frames[idx];
        if (!frameData) return;

        previewWidget.canvasEl.width = frameData.width;
        previewWidget.canvasEl.height = frameData.height;
        const ctx = previewWidget.canvasEl.getContext("2d");
        ctx.putImageData(frameData.imageData, 0, 0);

        // Draw mask region overlay if it exists
        if (
          previewNode.maskRegion &&
          previewNode.maskRegion.width > 0 &&
          previewNode.maskRegion.height > 0
        ) {
          const maskRect = previewNode.maskRegion;

          // Draw soft dark overlay on entire canvas
          ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
          ctx.fillRect(
            0,
            0,
            previewWidget.canvasEl.width,
            previewWidget.canvasEl.height,
          );

          // Clear the selected area to show original video
          ctx.clearRect(
            maskRect.x,
            maskRect.y,
            maskRect.width,
            maskRect.height,
          );

          // Redraw the original video content in the cleared area
          ctx.putImageData(
            frameData.imageData,
            0,
            0,
            maskRect.x,
            maskRect.y,
            maskRect.width,
            maskRect.height,
          );

          // Draw red border around selection
          ctx.strokeStyle = "#ff4444";
          ctx.lineWidth = 2;
          ctx.strokeRect(
            maskRect.x,
            maskRect.y,
            maskRect.width,
            maskRect.height,
          );
        }

        // Update aspect ratio if it changed (only on first frame or size change)
        if (
          !previewWidget.aspectRatio ||
          Math.abs(
            previewWidget.aspectRatio - frameData.width / frameData.height,
          ) > 0.01
        ) {
          previewWidget.aspectRatio = frameData.width / frameData.height;
          previewNode.size = [Math.max(previewNode.size[0], 300), 0];

          // Only redraw the graph when aspect ratio changes (resize), not on every frame
          if (previewNode.graph && previewNode.graph._canvas) {
            previewNode.graph._canvas.draw(true);
          } else if (previewNode.graph && previewNode.graph.canvas) {
            previewNode.graph.canvas.draw(true);
          }
        }
      };

      const stopAnimation = () => {
        if (previewWidget.playInterval) {
          clearInterval(previewWidget.playInterval);
          previewWidget.playInterval = null;
        }
      };

      const startAnimation = () => {
        stopAnimation();
        if (!previewWidget.frames || previewWidget.frames.length === 0) return;

        console.log(
          `[VideoPreview] Starting animation with ${previewWidget.frames.length} frames`,
        );

        const frameDuration = Math.max(previewWidget.frameDuration || 100, 16);

        const advanceFrame = () => {
          drawFrame(previewWidget.frameIndex);
          previewWidget.frameIndex =
            (previewWidget.frameIndex + 1) % previewWidget.frames.length;
        };

        advanceFrame();
        previewWidget.playInterval = setInterval(advanceFrame, frameDuration);
      };

      const getWidgetValue = (name, fallback) => {
        const widget = previewNode.widgets?.find((w) => w.name === name);
        return widget && widget.value !== undefined ? widget.value : fallback;
      };

      const previewState = {
        pendingTimeout: null,
        requestId: 0,
      };

      const loadInputPreview = async () => {
        previewState.pendingTimeout = null;
        const videoValue = getWidgetValue("video", null);
        if (!videoValue) {
          stopAnimation();
          previewWidget.parentEl.hidden = true;
          previewWidget.frames = [];
          return;
        }

        const params = new URLSearchParams({
          video: videoValue,
          framerate: getWidgetValue("framerate", 0) || 0,
          custom_width: getWidgetValue("custom_width", 0) || 0,
          custom_height: getWidgetValue("custom_height", 0) || 0,
          frame_load_cap: getWidgetValue("frame_load_cap", 0) || 0,
          skip_first_frames: getWidgetValue("skip_first_frames", 0) || 0,
          select_every_nth: getWidgetValue("select_every_nth", 1) || 1,
          format: getWidgetValue("format", "None") || "None",
          max_preview_frames: 48,
        });

        const requestId = ++previewState.requestId;
        stopAnimation();
        previewWidget.frames = [];
        previewWidget.frameIndex = 0;
        previewWidget.parentEl.style.display = "none";

        try {
          const response = await api.fetchApi(
            `/videomaskeditor/preview?${params.toString()}`,
          );
          if (!response.ok) {
            const message = await response.text();
            throw new Error(message || "Failed to load preview");
          }

          const data = await response.json();
          const frames = data.frames || [];
          if (!frames.length || requestId !== previewState.requestId) {
            return;
          }

          // Calculate frame duration based on the effective fps (the target framerate)
          // This represents how fast the selected frames should play back
          previewWidget.frameDuration = data.fps > 0 ? 1000 / data.fps : 100;
          previewWidget.frames = new Array(frames.length);

          console.log(
            `[VideoPreview] Loaded ${frames.length} frames at ${data.fps} fps (frame duration: ${previewWidget.frameDuration}ms)`,
          );

          let loadedCount = 0;
          frames.forEach((frameInfo, idx) => {
            const img = new Image();
            img.onload = () => {
              if (requestId !== previewState.requestId) {
                return;
              }
              const canvas = document.createElement("canvas");
              canvas.width = img.width;
              canvas.height = img.height;
              const ctx = canvas.getContext("2d");
              ctx.drawImage(img, 0, 0);
              const imageData = ctx.getImageData(0, 0, img.width, img.height);
              previewWidget.frames[idx] = {
                imageData,
                width: img.width,
                height: img.height,
              };

              loadedCount++;
              if (
                loadedCount === frames.length &&
                requestId === previewState.requestId
              ) {
                previewWidget.parentEl.style.display = "block";
                startAnimation();
              }
            };

            img.onerror = (e) => {
              console.error(
                `[VideoPreview] Failed to decode preview frame ${idx}:`,
                e,
              );
            };

            img.src = `data:image/png;base64,${frameInfo.data}`;
          });
        } catch (err) {
          if (requestId === previewState.requestId) {
            console.error("[VideoPreview] Preview load failed", err);
            previewWidget.parentEl.style.display = "none";
          }
        }
      };

      const scheduleInputPreview = () => {
        if (previewState.pendingTimeout) {
          clearTimeout(previewState.pendingTimeout);
        }
        previewState.pendingTimeout = setTimeout(loadInputPreview, 250);
      };

      previewNode._vmeScheduleInputPreview = scheduleInputPreview;
      previewNode._vmeStopPreviewAnimation = stopAnimation;
      previewNode._vmePreviewWidget = previewWidget;

      // Load execution frames after running
      chainCallback(nodeType.prototype, "onExecuted", function () {
        console.log("[VideoPreview] onExecuted - loading execution frames");
        if (previewNode._vmeStopPreviewAnimation) {
          previewNode._vmeStopPreviewAnimation();
        }
        const nodeId = previewNode.id;

        const loadExecutionFrames = async () => {
          try {
            const historyResp = await fetch(api.apiURL("/history"));
            const history = await historyResp.json();
            const lastPromptId = Object.keys(history).pop();

            console.log("[VideoPreview] Last prompt ID:", lastPromptId);

            if (
              !lastPromptId ||
              !history[lastPromptId] ||
              !history[lastPromptId].outputs
            ) {
              console.log("[VideoPreview] No outputs found");
              return;
            }

            const output = history[lastPromptId].outputs[nodeId];
            console.log("[VideoPreview] Node output:", output);

            if (!output || !output.images || !Array.isArray(output.images)) {
              console.log("[VideoPreview] No images in output");
              return;
            }

            console.log(`[VideoPreview] Found ${output.images.length} images`);

            previewWidget.frames = [];
            previewWidget.frameIndex = 0;

            // Get the video_fps from the node's output (4th return value - index 3)
            // This is the effective FPS that the frames should be played back at
            const videoFps =
              previewNode.widgets?.find((w) => w.name === "framerate")?.value ||
              0;
            const defaultFps = videoFps > 0 ? videoFps : 24; // fallback to 24 if not specified
            previewWidget.frameDuration = 1000 / defaultFps;

            console.log(
              `[VideoPreview] Playing back at ${defaultFps} fps (frame duration: ${previewWidget.frameDuration}ms)`,
            );

            let loadedCount = 0;

            output.images.forEach((imgInfo, idx) => {
              const imgUrl = api.apiURL(
                "/view?" +
                  new URLSearchParams({
                    filename: imgInfo.name,
                    type: imgInfo.type,
                    subfolder: imgInfo.subfolder || "",
                  }).toString(),
              );

              console.log(`[VideoPreview] Loading image ${idx}: ${imgUrl}`);

              const img = new Image();
              img.onload = () => {
                const canvas = document.createElement("canvas");
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext("2d");
                ctx.drawImage(img, 0, 0);

                const imageData = ctx.getImageData(0, 0, img.width, img.height);
                previewWidget.frames[idx] = {
                  imageData,
                  width: img.width,
                  height: img.height,
                };

                loadedCount++;
                console.log(
                  `[VideoPreview] Loaded ${loadedCount}/${output.images.length} images`,
                );

                if (loadedCount === output.images.length) {
                  previewWidget.parentEl.style.display = "block";
                  startAnimation();
                }
              };

              img.onerror = (e) => {
                console.error(
                  `[VideoPreview] Failed to load image ${idx}:`,
                  imgUrl,
                  e,
                );
              };

              img.src = imgUrl;
            });
          } catch (e) {
            console.error("[VideoPreview] Error loading execution frames:", e);
          }
        };

        setTimeout(loadExecutionFrames, 100);
      });

      // Store mask region data on the node
      previewNode.maskRegion = null;

      // Add context menu option for mask editing
      previewNode.getExtraMenuOptions = function (_, options) {
        options.unshift({
          content: "Set Mask",
          callback: () => {
            openMaskEditor(previewNode, previewWidget);
          },
        });
      };
    });

    // Add upload widget
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
      const videoWidget = this.widgets?.find((w) => w.name === "video");
      const triggerPreviewUpdate = () => {
        if (this._vmeScheduleInputPreview) {
          this._vmeScheduleInputPreview();
        }
      };

      if (videoWidget) {
        const originalVideoCallback = videoWidget.callback;
        videoWidget.callback = function (value) {
          if (originalVideoCallback) originalVideoCallback.call(this, value);
          triggerPreviewUpdate();
          return value;
        };
      }

      const previewDependentWidgets = [
        "custom_width",
        "custom_height",
        "frame_load_cap",
        "skip_first_frames",
        "select_every_nth",
        "format",
      ];

      previewDependentWidgets.forEach((widgetName) => {
        const widget = this.widgets?.find((w) => w.name === widgetName);
        if (!widget) return;
        const original = widget.callback;
        widget.callback = function (value) {
          if (original) original.call(this, value);
          triggerPreviewUpdate();
          return value;
        };
      });

      const fileInput = document.createElement("input");
      fileInput.type = "file";
      fileInput.accept = "video/*,.webm,.mp4,.mkv,.gif,.mov";
      fileInput.style.display = "none";

      fileInput.addEventListener("change", async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        try {
          const formData = new FormData();
          formData.append("image", file);
          formData.append("type", "input");

          const response = await api.fetchApi("/upload/image", {
            method: "POST",
            body: formData,
          });

          if (response.ok) {
            const data = await response.json();
            if (videoWidget) {
              videoWidget.value = data.name;
              if (videoWidget.callback) {
                videoWidget.callback(data.name);
              }
            }
          }
        } catch (error) {
          console.error("Upload failed:", error);
        }
      });
      document.body.appendChild(fileInput);

      this.addWidget("button", "choose video to upload", "upload", () => {
        app.canvas.node_widget = null;
        fileInput.click();
      }).options.serialize = false;

      if (this._vmePreviewWidget) {
        const idx = this.widgets.indexOf(this._vmePreviewWidget);
        if (idx !== -1) {
          const [widget] = this.widgets.splice(idx, 1);
          this.widgets.push(widget);
        }
      }

      // Broadcast framerate
      const framerateWidget = this.widgets?.find((w) => w.name === "framerate");
      if (framerateWidget) {
        const originalCallback = framerateWidget.callback;
        framerateWidget.callback = function (value) {
          if (originalCallback) originalCallback.call(this, value);
          const allDialogs = document.querySelectorAll("[data-type='dialog']");
          allDialogs.forEach((dialog) => {
            if (dialog.messageBroker) {
              dialog.messageBroker.publish("setVideoFpsRequest", value);
            }
          });
          triggerPreviewUpdate();
        };
      }

      triggerPreviewUpdate();
    });
  },
  async setup() {
    // Listen for mask update events from the backend
    api.addEventListener("videomaskeditor.mask_updated", ({ detail }) => {
      const nodeId = detail?.node_id;
      if (nodeId) {
        const node = app.graph._nodes_by_id[nodeId];
        if (node) {
          // Mark the node as needing to be re-executed by changing a widget value
          // This forces ComfyUI to invalidate the cache
          const maskCropsWidget = node.widgets?.find(
            (w) => w.name === "mask_crops_frames",
          );
          if (maskCropsWidget) {
            const currentValue = maskCropsWidget.value;
            // Toggle and restore to force cache invalidation
            maskCropsWidget.value = !currentValue;
            maskCropsWidget.value = currentValue;
          }

          // Mark the node and graph as dirty
          node.setDirtyCanvas(true, true);
          app.graph.setDirtyCanvas(true, true);
        }
      }
    });
  },
});
