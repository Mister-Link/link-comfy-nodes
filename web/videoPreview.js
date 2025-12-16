import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function openMaskEditor(node, previewWidget) {
  // Store for unmasked frames that the dialog will use
  const dialogFrames = {
    frames: [],
    frameDuration: 100,
    loading: true,
  };

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
    position: relative;
  `;

  // Close button
  const closeButton = document.createElement("button");
  closeButton.textContent = "×";
  closeButton.style.cssText = `
    position: absolute;
    top: 10px;
    right: 10px;
    background: transparent;
    color: #aaa;
    border: none;
    font-size: 28px;
    line-height: 20px;
    cursor: pointer;
    padding: 0;
    width: 30px;
    height: 30px;
    border-radius: 4px;
    transition: background 0.2s, color 0.2s;
  `;
  closeButton.addEventListener("mouseenter", () => {
    closeButton.style.background = "#555";
    closeButton.style.color = "#fff";
  });
  closeButton.addEventListener("mouseleave", () => {
    closeButton.style.background = "transparent";
    closeButton.style.color = "#aaa";
  });
  dialog.appendChild(closeButton);

  // Instructions
  const instructions = document.createElement("p");
  instructions.textContent =
    "Click-drag to set mask region. Right-click to erase. Scroll to set brush size.";
  instructions.style.cssText =
    "margin: 0; font-size: 13px; color: #aaa; padding-right: 40px;";
  dialog.appendChild(instructions);

  // Mode selector and controls container
  const controlsRow = document.createElement("div");
  controlsRow.style.cssText = "display: flex; gap: 10px; align-items: center;";

  // Mask mode selector
  const modeLabel = document.createElement("label");
  modeLabel.textContent = "Mode: ";
  modeLabel.style.cssText = "font-size: 14px;";

  const modeSelect = document.createElement("select");
  modeSelect.style.cssText = `
    padding: 4px 8px;
    background: #333;
    color: #fff;
    border: 1px solid #555;
    border-radius: 4px;
    cursor: pointer;
  `;
  modeSelect.innerHTML = `
    <option value="bbox">BBox Selection</option>
    <option value="paint">Paint Brush</option>
  `;

  let maskMode = "bbox";
  let interpolateEnabled = true; // Interpolation enabled by default

  modeSelect.addEventListener("change", (e) => {
    const previousMode = maskMode;
    maskMode = e.target.value;
    canvas.style.cursor = maskMode === "paint" ? "crosshair" : "crosshair";

    // Show/hide interpolate checkbox based on mode
    if (maskMode === "bbox") {
      interpolateCheckboxContainer.style.display = "flex";
    } else {
      interpolateCheckboxContainer.style.display = "none";
    }

    // Don't clear anything when switching modes - allow coexistence
    // maskRect = null;
    // paintMaskData = null;
    // Don't reset lastRenderedKeyframeIndex to preserve current mask state
    // lastRenderedKeyframeIndex = -1;

    drawCanvas();
  });

  // Interpolate checkbox (only visible in bbox mode)
  const interpolateCheckboxContainer = document.createElement("div");
  interpolateCheckboxContainer.style.cssText =
    "display: flex; gap: 5px; align-items: center; margin-left: 10px;";

  const interpolateCheckbox = document.createElement("input");
  interpolateCheckbox.type = "checkbox";
  interpolateCheckbox.checked = true;
  interpolateCheckbox.style.cssText = "cursor: pointer;";
  interpolateCheckbox.addEventListener("change", (e) => {
    interpolateEnabled = e.target.checked;
    // Trigger re-execution to update masks with/without interpolation
    drawCanvas();
  });

  const interpolateLabel = document.createElement("label");
  interpolateLabel.textContent = "Interpolate";
  interpolateLabel.style.cssText = "font-size: 13px; cursor: pointer;";
  interpolateLabel.addEventListener("click", () => {
    interpolateCheckbox.checked = !interpolateCheckbox.checked;
    interpolateEnabled = interpolateCheckbox.checked;
    drawCanvas();
  });

  interpolateCheckboxContainer.appendChild(interpolateCheckbox);
  interpolateCheckboxContainer.appendChild(interpolateLabel);

  // Brush size (controlled by scroll wheel)
  let brushSize = 20;

  let isPlaying = true;

  controlsRow.appendChild(modeLabel);
  controlsRow.appendChild(modeSelect);
  controlsRow.appendChild(interpolateCheckboxContainer);

  dialog.appendChild(controlsRow);

  // Canvas container with transparency checkerboard background
  const canvasContainer = document.createElement("div");
  canvasContainer.style.cssText = `
    position: relative;
    max-width: 100%;
    max-height: 60vh;
    overflow: hidden;
    background-image:
      linear-gradient(45deg, #383838 25%, transparent 25%),
      linear-gradient(-45deg, #383838 25%, transparent 25%),
      linear-gradient(45deg, transparent 75%, #383838 75%),
      linear-gradient(-45deg, transparent 75%, #383838 75%);
    background-size: 20px 20px;
    background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
    background-color: #2b2b2b;
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

  // Brush preview cursor (only visible in paint mode)
  const brushPreview = document.createElement("div");
  brushPreview.style.cssText = `
    position: absolute;
    border: 2px solid rgba(255, 255, 255, 0.8);
    border-radius: 50%;
    pointer-events: none;
    display: none;
    z-index: 1000;
  `;
  canvasContainer.appendChild(brushPreview);

  // Note: Do not stop propagation here - let the drawing handlers below handle it

  canvasContainer.appendChild(canvas);

  // Add canvas container to dialog
  dialog.appendChild(canvasContainer);

  // Timeline scrubber
  const scrubberContainer = document.createElement("div");
  scrubberContainer.style.cssText = `
    width: 100%;
    padding: 10px;
    background: #2a2a2a;
    border-radius: 4px;
  `;

  const scrubberBar = document.createElement("div");
  scrubberBar.style.cssText = `
    position: relative;
    width: 100%;
    height: 40px;
    background: #888;
    border-radius: 4px;
    cursor: pointer;
    overflow: hidden;
  `;

  const selectionMarker = document.createElement("div");
  selectionMarker.style.cssText = `
    position: absolute;
    top: 0;
    left: 0; /* Updated dynamically */
    width: 0; /* Updated dynamically */
    height: 100%;
    background: rgba(80, 120, 255, 0.25);
    border: 1px solid white;
    z-index: 20;
    pointer-events: none;
    box-sizing: border-box;
    display: none;
  `;
  scrubberBar.appendChild(selectionMarker);

  // Add scrubber drag state
  let isScrubberDragging = false;

  const handleScrubberDrag = (e) => {
    if (!dialogFrames.frames || dialogFrames.frames.length === 0) return;

    const rect = scrubberBar.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percent = Math.max(0, Math.min(1, x / rect.width));
    const frameIdx = Math.floor(percent * dialogFrames.frames.length);

    dialogFrameIndex = Math.max(
      0,
      Math.min(dialogFrames.frames.length - 1, frameIdx),
    );

    if (isPlaying) {
      isPlaying = false;
      if (playPauseBtnTransport) {
        playPauseBtnTransport.textContent = "|>";
      }
      stopAnimation();
    }

    updateScrubber();
    drawCanvas();
  };

  scrubberBar.addEventListener("pointerdown", (e) => {
    isScrubberDragging = true;
    handleScrubberDrag(e);

    const onPointerMove = (e) => {
      if (isScrubberDragging) {
        handleScrubberDrag(e);
      }
    };

    const onPointerUp = () => {
      isScrubberDragging = false;
      document.removeEventListener("pointermove", onPointerMove);
      document.removeEventListener("pointerup", onPointerUp);
    };

    document.addEventListener("pointermove", onPointerMove);
    document.addEventListener("pointerup", onPointerUp);
  });

  scrubberContainer.appendChild(scrubberBar);

  // Frame info display (bottom left)
  const frameInfo = document.createElement("div");
  frameInfo.style.cssText = "font-size: 12px; color: #aaa; margin-top: 5px;";
  frameInfo.textContent = "Frame: 0 / 0";
  scrubberContainer.appendChild(frameInfo);

  // Transport controls container (below scrubber bar)
  const transportControls = document.createElement("div");
  transportControls.style.cssText = `
    display: flex;
    gap: 8px;
    align-items: center;
    justify-content: center;
    margin-top: 10px;
  `;

  const createTransportButton = (label, onClick) => {
    const button = document.createElement("button");
    button.textContent = label;
    button.style.cssText = `
      padding: 8px 14px;
      background: #444;
      color: #fff;
      border: none;
      border-radius: 4px;
      font-size: 16px;
      min-width: 44px;
      font-family: monospace;
      transition: background 0.2s;
    `;
    button.style.cursor = "pointer";
    button.addEventListener("click", onClick);
    button.addEventListener("mouseenter", () => {
      if (!button.disabled) {
        button.style.background = "#555";
      }
    });
    button.addEventListener("mouseleave", () => {
      if (!button.disabled) {
        button.style.background = "#444";
      }
    });
    return button;
  };

  scrubberContainer.appendChild(transportControls);
  dialog.appendChild(scrubberContainer);

  // Mask selection state
  let maskRect = node.maskRegion || null;
  const initialMaskRect = maskRect ? { ...maskRect } : null; // Store initial state for cancel
  let isDrawing = false;
  let isDragging = false;
  let isResizing = false;
  let resizeHandle = null; // 'nw', 'ne', 'sw', 'se', 'n', 's', 'e', 'w'
  let startX = 0;
  let startY = 0;
  let dragStartRect = null;
  let animationInterval = null;
  let dialogFrameIndex = 0; // Separate frame index for dialog to avoid interference
  let lastRenderedKeyframeIndex = -1; // Track which keyframe we last loaded maskRect from
  const HANDLE_SIZE = 12; // Size of corner/edge handles for resizing

  // Keyframe state
  let keyframes = {}; // frame_index -> {type: "bbox"|"painted", data: ...}
  let initialKeyframes = {}; // Store initial keyframes for cancel
  let paintMaskData = null; // Raw ImageData buffer for painting (width * height array)
  let paintMaskFrameIndex = -1; // Which frame the current paint data belongs to
  let isPainting = false;
  let lastPaintX = 0;
  let lastPaintY = 0;
  let currentPaintColor = 0; // 0 = black (painted), 255 = white (erased)
  let paintRenderScheduled = false;

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

  // Forward declare updateTransportButtons so it can be used in updateScrubber
  let updateTransportButtons = null;

  // Update scrubber position and keyframe markers
  const updateScrubber = () => {
    if (!dialogFrames.frames || dialogFrames.frames.length === 0) return;

    const totalFrames = dialogFrames.frames.length;
    frameInfo.textContent = `Frame: ${dialogFrameIndex + 1} / ${totalFrames}${
      keyframes[dialogFrameIndex] ? " (Keyframe)" : ""
    }`;

    // Update transport button states if available
    if (updateTransportButtons) {
      updateTransportButtons();
    }

    // Update the selection marker's position
    if (totalFrames > 0) {
      const frameWidthPercent = 100 / totalFrames;
      const leftPosPercent = (dialogFrameIndex / totalFrames) * 100;

      selectionMarker.style.display = "block";
      selectionMarker.style.left = `${leftPosPercent}%`;
      selectionMarker.style.width = `${frameWidthPercent}%`;
    } else {
      selectionMarker.style.display = "none";
    }

    // Remove existing frame boxes
    const existingBoxes = scrubberBar.querySelectorAll(".frame-box");
    existingBoxes.forEach((b) => b.remove());

    // Draw frame boxes
    const frameWidth = 100 / totalFrames;

    for (let i = 0; i < totalFrames; i++) {
      const frameBox = document.createElement("div");
      frameBox.className = "frame-box";
      frameBox.dataset.frameIndex = i;

      const leftPos = (i / totalFrames) * 100;
      const isCurrentFrame = i === dialogFrameIndex;
      const isKeyframe = keyframes[i];

      // Determine color and styling
      let bgColor = "#666"; // default frame
      let zIndex = 0;

      if (isKeyframe) {
        bgColor = "#00cc00"; // green for keyframes
      }

      if (isCurrentFrame) {
        zIndex = 10; // bring to front so keyframe color is visible under marker
      }

      frameBox.style.cssText = `
          position: absolute;
          top: 0;
          left: ${leftPos}%;
          width: ${frameWidth}%;
          height: 100%;
          background: ${bgColor};
          border-right: 1px solid #444;
          pointer-events: auto;
          z-index: ${zIndex};
          box-sizing: border-box;
          transition: background-color 0.15s ease;
        `;

      // Add hover effect
      frameBox.addEventListener("mouseenter", () => {
        if (i !== dialogFrameIndex) {
          frameBox.style.filter = "brightness(1.2)";
        }
      });

      frameBox.addEventListener("mouseleave", () => {
        frameBox.style.filter = "none";
      });

      // Click to select frame
      frameBox.addEventListener("click", (e) => {
        e.stopPropagation();
        dialogFrameIndex = i;
        if (isPlaying) {
          isPlaying = false;
          if (playPauseBtnTransport) {
            playPauseBtnTransport.textContent = "|>";
          }
          stopAnimation();
        }
        updateScrubber();
        drawCanvas();
      });

      // Right-click to delete keyframe
      frameBox.addEventListener("contextmenu", async (e) => {
        e.preventDefault();
        e.stopPropagation();

        if (keyframes[i]) {
          delete keyframes[i];

          try {
            await api.fetchApi("/videomaskeditor/deletekeyframe", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                node_id: node.id,
                frame_index: i,
              }),
            });
          } catch (error) {
            console.error(
              "[VideoMaskEditor] Failed to delete keyframe:",
              error,
            );
          }

          maskRect = null;
          paintedMaskCanvas = null;
          lastRenderedKeyframeIndex = -1;
          updateScrubber();
          drawCanvas();
        }
      });

      scrubberBar.appendChild(frameBox);
    }
  };

  // Load keyframes from backend
  const loadKeyframes = async () => {
    try {
      const response = await api.fetchApi(
        `/videomaskeditor/getkeyframes?node_id=${node.id}`,
      );
      if (response.ok) {
        const data = await response.json();
        keyframes = data.keyframes || {};
        // Store a deep copy of initial keyframes for cancel functionality
        initialKeyframes = JSON.parse(JSON.stringify(keyframes));
        updateScrubber();
      }
    } catch (error) {
      console.error("[VideoMaskEditor] Failed to load keyframes:", error);
    }
  };

  // Draw current frame with mask overlay
  const drawCanvas = () => {
    if (!dialogFrames.frames || dialogFrames.frames.length === 0) return;

    const frameData =
      dialogFrames.frames[dialogFrameIndex % dialogFrames.frames.length];
    if (!frameData) return;

    canvas.width = frameData.width;
    canvas.height = frameData.height;

    const ctx = canvas.getContext("2d");
    ctx.putImageData(frameData.imageData, 0, 0);

    let activeKeyframe = null;

    // Find the most recent keyframe for the current frame
    const sortedKeyframeIndices = Object.keys(keyframes)
      .map(Number)
      .sort((a, b) => a - b);

    for (let i = sortedKeyframeIndices.length - 1; i >= 0; i--) {
      const keyframeIndex = sortedKeyframeIndices[i];
      if (keyframeIndex <= dialogFrameIndex) {
        activeKeyframe = keyframes[keyframeIndex];
        break;
      }
    }

    // Determine what to show
    let showPaint = false;
    let showBbox = false;
    let paintDataToShow = null;

    // By default, clear the mask unless we find a keyframe or are editing
    // Don't clear if we're actively painting either, or if we have paint data for this frame
    if (
      !isDrawing &&
      !isDragging &&
      !isResizing &&
      !isPainting &&
      !(paintMaskData && paintMaskFrameIndex === dialogFrameIndex)
    ) {
      maskRect = null;
    }

    if (activeKeyframe) {
      // Support hybrid keyframes with both bbox and painted data
      if (activeKeyframe.type === "bbox" || activeKeyframe.bbox) {
        showBbox = true;
        if (!isDrawing && !isDragging && !isResizing) {
          // Apply interpolation if enabled and there's a next bbox keyframe
          if (interpolateEnabled) {
            // Find next bbox keyframe for interpolation
            let nextKeyframeIndex = null;
            let nextKeyframe = null;

            for (let i = 0; i < sortedKeyframeIndices.length; i++) {
              const kfIdx = sortedKeyframeIndices[i];
              if (kfIdx > dialogFrameIndex) {
                nextKeyframeIndex = kfIdx;
                nextKeyframe = keyframes[kfIdx];
                break;
              }
            }

            // If we have a next bbox keyframe, interpolate
            if (
              nextKeyframe &&
              nextKeyframe.type === "bbox" &&
              nextKeyframeIndex !== null
            ) {
              const prevKeyframeIndex = sortedKeyframeIndices.find(
                (idx) =>
                  idx <= dialogFrameIndex && keyframes[idx] === activeKeyframe,
              );
              const prevBbox = activeKeyframe.bbox;
              const nextBbox = nextKeyframe.bbox;

              if (prevBbox && nextBbox && prevKeyframeIndex !== undefined) {
                const totalFrames = nextKeyframeIndex - prevKeyframeIndex;
                const t = (dialogFrameIndex - prevKeyframeIndex) / totalFrames;

                // Linear interpolation
                maskRect = {
                  x: Math.round(prevBbox.x * (1 - t) + nextBbox.x * t),
                  y: Math.round(prevBbox.y * (1 - t) + nextBbox.y * t),
                  width: Math.round(
                    prevBbox.width * (1 - t) + nextBbox.width * t,
                  ),
                  height: Math.round(
                    prevBbox.height * (1 - t) + nextBbox.height * t,
                  ),
                };
              } else {
                maskRect = activeKeyframe.bbox
                  ? { ...activeKeyframe.bbox }
                  : null;
              }
            } else {
              maskRect = activeKeyframe.bbox
                ? { ...activeKeyframe.bbox }
                : null;
            }
          } else {
            // No interpolation - just use the keyframe bbox
            maskRect = activeKeyframe.bbox ? { ...activeKeyframe.bbox } : null;
          }
        }
      }

      // Also check for painted data (can coexist with bbox)
      if (activeKeyframe.type === "painted" || activeKeyframe.mask_data) {
        showPaint = true;
        paintDataToShow = activeKeyframe.mask_data;
      }
    }

    // Active bbox editing takes precedence but doesn't hide paint
    if (isDrawing || isDragging || isResizing) {
      showBbox = true;
      // showPaint remains as-is, allowing both to show
    }

    // If actively painting, don't hide bbox
    if (isPainting) {
      // showBbox remains as-is
      // maskRect remains as-is
    }

    // Both bbox and paint can be visible simultaneously
    // if (maskRect) {
    //   showPaint = false;
    // }

    // --- RENDER ---

    // 1. Render Paint, if applicable
    if (showPaint && paintDataToShow) {
      try {
        const binary = atob(paintDataToShow);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
          bytes[i] = binary.charCodeAt(i);
        }
        const maskArray = new Float32Array(bytes.buffer);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        for (let i = 0; i < maskArray.length; i++) {
          if (maskArray[i] > 0.5) {
            const idx = i * 4;
            // Set to semi-transparent red, ignoring original color and alpha
            imageData.data[idx] = 255; // R
            imageData.data[idx + 1] = 0; // G
            imageData.data[idx + 2] = 0; // B
            imageData.data[idx + 3] = 180; // Alpha (70% opacity)
          }
        }
        ctx.putImageData(imageData, 0, 0);
      } catch (error) {
        console.error(
          "[VideoMaskEditor] Failed to display painted mask:",
          error,
        );
      }
    }

    // 3. Render Bbox, if applicable
    if (showBbox && maskRect && maskRect.width > 0 && maskRect.height > 0) {
      // Get the current image data to modify pixels in the masked region
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;

      // Add red tint to the masked region while preserving transparency
      const x1 = Math.floor(Math.max(0, maskRect.x));
      const y1 = Math.floor(Math.max(0, maskRect.y));
      const x2 = Math.ceil(Math.min(canvas.width, maskRect.x + maskRect.width));
      const y2 = Math.ceil(
        Math.min(canvas.height, maskRect.y + maskRect.height),
      );

      for (let y = y1; y < y2; y++) {
        for (let x = x1; x < x2; x++) {
          const idx = (y * canvas.width + x) * 4;
          const originalAlpha = data[idx + 3];

          // Add red tint: blend towards red while preserving original alpha
          const redMix = 0.5; // How much red to blend in

          data[idx] = Math.min(
            255,
            Math.round(data[idx] + (255 - data[idx]) * redMix),
          ); // R
          data[idx + 1] = Math.round(data[idx + 1] * (1 - redMix * 0.6)); // G
          data[idx + 2] = Math.round(data[idx + 2] * (1 - redMix * 0.6)); // B

          // For transparent areas, add semi-transparent red so it's visible
          if (originalAlpha < 128) {
            data[idx + 3] = Math.max(originalAlpha, 128); // Ensure minimum visibility
          }
        }
      }

      // Put the modified image data back
      ctx.putImageData(imageData, 0, 0);

      // Draw red border
      ctx.strokeStyle = "#ff4444";
      ctx.lineWidth = 2;
      ctx.strokeRect(maskRect.x, maskRect.y, maskRect.width, maskRect.height);

      if (!isDrawing) {
        ctx.fillStyle = "#ffffff";
        const handles = [
          { x: maskRect.x, y: maskRect.y },
          { x: maskRect.x + maskRect.width, y: maskRect.y },
          { x: maskRect.x, y: maskRect.y + maskRect.height },
          { x: maskRect.x + maskRect.width, y: maskRect.y + maskRect.height },
          { x: maskRect.x + maskRect.width / 2, y: maskRect.y },
          {
            x: maskRect.x + maskRect.width / 2,
            y: maskRect.y + maskRect.height,
          },
          { x: maskRect.x, y: maskRect.y + maskRect.height / 2 },
          {
            x: maskRect.x + maskRect.width,
            y: maskRect.y + maskRect.height / 2,
          },
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
    const frameDuration = dialogFrames.frameDuration || 100;
    animationInterval = setInterval(() => {
      dialogFrameIndex = (dialogFrameIndex + 1) % dialogFrames.frames.length;
      drawCanvas();
      updateScrubber();
    }, frameDuration);
  };

  // Stop animation
  const stopAnimation = () => {
    if (animationInterval) {
      clearInterval(animationInterval);
      animationInterval = null;
    }
  };

  // Play/Pause button handler (forward declare so we can reference it later)
  let playPauseBtnTransport = null;

  // Scrubber no longer needs click handlers since frame boxes handle selection

  // Helper function to get sorted keyframe indices
  const getSortedKeyframeIndices = () => {
    return Object.keys(keyframes)
      .map((k) => parseInt(k))
      .sort((a, b) => a - b);
  };

  // Helper function to check if previous keyframe exists
  const hasPreviousKeyframe = () => {
    const sorted = getSortedKeyframeIndices();
    return sorted.some((kf) => kf < dialogFrameIndex);
  };

  // Helper function to check if next keyframe exists
  const hasNextKeyframe = () => {
    const sorted = getSortedKeyframeIndices();
    return sorted.some((kf) => kf > dialogFrameIndex);
  };

  // Transport control functions
  const goToBeginning = () => {
    dialogFrameIndex = 0;
    if (isPlaying) {
      isPlaying = false;
      if (playPauseBtnTransport) {
        playPauseBtnTransport.textContent = "|>";
      }
      stopAnimation();
    }
    updateTransportButtons();
    drawCanvas();
    updateScrubber();
  };

  const goToEnd = () => {
    dialogFrameIndex = dialogFrames.frames.length - 1;
    if (isPlaying) {
      isPlaying = false;
      if (playPauseBtnTransport) {
        playPauseBtnTransport.textContent = "|>";
      }
      stopAnimation();
    }
    updateTransportButtons();
    drawCanvas();
    updateScrubber();
  };

  const goToPreviousKeyframe = () => {
    const sortedKeyframeIndices = getSortedKeyframeIndices();

    // Find the previous keyframe (strictly before current frame)
    let targetFrame = null;
    for (let i = sortedKeyframeIndices.length - 1; i >= 0; i--) {
      if (sortedKeyframeIndices[i] < dialogFrameIndex) {
        targetFrame = sortedKeyframeIndices[i];
        break;
      }
    }

    if (targetFrame !== null) {
      dialogFrameIndex = targetFrame;
      if (isPlaying) {
        isPlaying = false;
        if (playPauseBtnTransport) {
          playPauseBtnTransport.textContent = "|>";
        }
        stopAnimation();
      }
      updateTransportButtons();
      drawCanvas();
      updateScrubber();
    }
  };

  const goToPreviousFrame = () => {
    if (dialogFrameIndex > 0) {
      dialogFrameIndex--;
      if (isPlaying) {
        isPlaying = false;
        if (playPauseBtnTransport) {
          playPauseBtnTransport.textContent = "|>";
        }
        stopAnimation();
      }
      updateTransportButtons();
      drawCanvas();
      updateScrubber();
    }
  };

  const togglePlayPause = () => {
    isPlaying = !isPlaying;
    if (isPlaying) {
      if (playPauseBtnTransport) {
        playPauseBtnTransport.textContent = "||";
      }
      startAnimation();
    } else {
      if (playPauseBtnTransport) {
        playPauseBtnTransport.textContent = "|>";
      }
      stopAnimation();
    }
  };

  const goToNextFrame = () => {
    if (dialogFrameIndex < dialogFrames.frames.length - 1) {
      dialogFrameIndex++;
      if (isPlaying) {
        isPlaying = false;
        if (playPauseBtnTransport) {
          playPauseBtnTransport.textContent = "|>";
        }
        stopAnimation();
      }
      updateTransportButtons();
      drawCanvas();
      updateScrubber();
    }
  };

  const goToNextKeyframe = () => {
    const sortedKeyframeIndices = getSortedKeyframeIndices();

    // Find the next keyframe (strictly after current frame)
    let targetFrame = null;
    for (const kfIdx of sortedKeyframeIndices) {
      if (kfIdx > dialogFrameIndex) {
        targetFrame = kfIdx;
        break;
      }
    }

    if (targetFrame !== null) {
      dialogFrameIndex = targetFrame;
      if (isPlaying) {
        isPlaying = false;
        if (playPauseBtnTransport) {
          playPauseBtnTransport.textContent = "|>";
        }
        stopAnimation();
      }
      updateTransportButtons();
      drawCanvas();
      updateScrubber();
    }
  };

  // Create transport control buttons
  const beginningBtn = createTransportButton("|<<", goToBeginning);
  const prevKeyframeBtn = createTransportButton("<<", goToPreviousKeyframe);
  const prevFrameBtn = createTransportButton("<", goToPreviousFrame);
  playPauseBtnTransport = createTransportButton("||", togglePlayPause);
  const nextFrameBtn = createTransportButton(">", goToNextFrame);
  const nextKeyframeBtn = createTransportButton(">>", goToNextKeyframe);
  const endBtn = createTransportButton(">>|", goToEnd);

  // Update button states based on keyframe availability
  updateTransportButtons = () => {
    const hasPrev = hasPreviousKeyframe();
    const hasNext = hasNextKeyframe();

    prevKeyframeBtn.disabled = !hasPrev;
    prevKeyframeBtn.style.opacity = hasPrev ? "1" : "0.4";
    prevKeyframeBtn.style.cursor = hasPrev ? "pointer" : "not-allowed";
    prevKeyframeBtn.style.background = hasPrev ? "#444" : "#333";

    nextKeyframeBtn.disabled = !hasNext;
    nextKeyframeBtn.style.opacity = hasNext ? "1" : "0.4";
    nextKeyframeBtn.style.cursor = hasNext ? "pointer" : "not-allowed";
    nextKeyframeBtn.style.background = hasNext ? "#444" : "#333";
  };

  transportControls.appendChild(beginningBtn);
  transportControls.appendChild(prevKeyframeBtn);
  transportControls.appendChild(prevFrameBtn);
  transportControls.appendChild(playPauseBtnTransport);
  transportControls.appendChild(nextFrameBtn);
  transportControls.appendChild(nextKeyframeBtn);
  transportControls.appendChild(endBtn);

  // Initial button state update
  updateTransportButtons();

  // Initialize paint mask data buffer
  const initPaintMask = () => {
    // Return if we're already painting on this frame
    if (paintMaskData && paintMaskFrameIndex === dialogFrameIndex) {
      return;
    }

    const size = canvas.width * canvas.height;
    paintMaskData = new Uint8ClampedArray(size);
    paintMaskFrameIndex = dialogFrameIndex;

    // Find the most recent keyframe to see if we can inherit its paint
    let activeKeyframe = null;
    const sortedKeyframeIndices = Object.keys(keyframes)
      .map(Number)
      .sort((a, b) => a - b);
    for (let i = sortedKeyframeIndices.length - 1; i >= 0; i--) {
      const keyframeIndex = sortedKeyframeIndices[i];
      if (keyframeIndex <= dialogFrameIndex) {
        activeKeyframe = keyframes[keyframeIndex];
        break;
      }
    }

    // If the active keyframe is painted, start with its data
    if (activeKeyframe && activeKeyframe.type === "painted") {
      try {
        const binary = atob(activeKeyframe.mask_data);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
          bytes[i] = binary.charCodeAt(i);
        }
        const maskArray = new Float32Array(bytes.buffer);
        // Convert from Float32Array (0.0 to 1.0) to Uint8ClampedArray (255 to 0)
        for (let i = 0; i < maskArray.length; i++) {
          paintMaskData[i] = maskArray[i] > 0.5 ? 0 : 255;
        }
      } catch (e) {
        console.error(
          "Failed to decode previous paint mask, starting fresh.",
          e,
        );
        paintMaskData.fill(255); // Fallback to a blank mask on error
      }
    } else {
      // Otherwise, start with a blank mask
      paintMaskData.fill(255);
    }
  };

  // Draw a circle into the paint mask buffer
  const paintCircle = (centerX, centerY, radius, color) => {
    if (!paintMaskData) return;

    const width = canvas.width;
    const height = canvas.height;
    const radiusSq = radius * radius;

    const y_min = Math.max(0, Math.floor(centerY - radius));
    const y_max = Math.min(height - 1, Math.ceil(centerY + radius));

    for (let j = y_min; j <= y_max; j++) {
      const dy = j - centerY;
      const dx_sq = radiusSq - dy * dy;
      if (dx_sq < 0) continue;
      const dx = Math.sqrt(dx_sq);

      const x_min = Math.max(0, Math.ceil(centerX - dx));
      const x_max = Math.min(width - 1, Math.floor(centerX + dx));

      let idx = j * width + x_min;
      for (let i = x_min; i <= x_max; i++) {
        paintMaskData[idx++] = color;
      }
    }
  };

  // Draw a line of circles between two points
  const paintLine = (x0, y0, x1, y1, radius, color) => {
    const dx = x1 - x0;
    const dy = y1 - y0;
    const distance = Math.sqrt(dx * dx + dy * dy);

    if (distance === 0) {
      paintCircle(x0, y0, radius, color);
      return;
    }

    // Calculate number of steps based on radius for smooth line
    const steps = Math.max(1, Math.ceil(distance / (radius * 0.5)));

    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      const x = x0 + dx * t;
      const y = y0 + dy * t;
      paintCircle(x, y, radius, color);
    }
  };

  // Mouse event handlers
  const handlePointerDown = (e) => {
    e.preventDefault();
    e.stopPropagation();

    // Pause the player when clicking on canvas
    if (isPlaying) {
      isPlaying = false;
      if (playPauseBtnTransport) {
        playPauseBtnTransport.textContent = "|>";
      }
      stopAnimation();
    }

    const coords = toCanvasCoords(e);
    startX = coords.x;
    startY = coords.y;

    const isRightClick = e.button === 2;

    if (maskMode === "paint") {
      // In paint mode - don't clear bbox, allow coexistence
      // if (maskRect) {
      //   maskRect = null;
      //   lastRenderedKeyframeIndex = -1;
      // }

      // Don't delete bbox keyframe - allow both to exist
      // if (
      //   keyframes[dialogFrameIndex] &&
      //   keyframes[dialogFrameIndex].type === "bbox"
      // ) {
      //   delete keyframes[dialogFrameIndex];
      //   // Fire-and-forget deletion on the backend
      //   api
      //     .fetchApi("/videomaskeditor/deletekeyframe", {
      //       method: "POST",
      //       headers: { "Content-Type": "application/json" },
      //       body: JSON.stringify({
      //         node_id: node.id,
      //         frame_index: dialogFrameIndex,
      //       }),
      //     })
      //     .catch((err) => console.error("Failed to delete bbox keyframe", err));
      // }

      // Paint mode - start painting or erasing
      isPainting = true;
      initPaintMask();

      currentPaintColor = isRightClick ? 255 : 0; // 255 = erase (white), 0 = paint (black)
      lastPaintX = startX;
      lastPaintY = startY;

      paintCircle(startX, startY, brushSize, currentPaintColor);
      drawCanvas();
    } else {
      // BBox mode
      if (isRightClick) {
        // Right-click: delete the bbox if clicking inside it
        if (isInsideMask(startX, startY)) {
          maskRect = null;
          drawCanvas();
          return;
        }
      } else {
        // Left-click: normal bbox operations
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
          // Don't delete painted keyframe - allow both to exist
          // if (
          //   keyframes[dialogFrameIndex] &&
          //   keyframes[dialogFrameIndex].type === "painted"
          // ) {
          //   delete keyframes[dialogFrameIndex];
          //   api
          //     .fetchApi("/videomaskeditor/deletekeyframe", {
          //       method: "POST",
          //       headers: { "Content-Type": "application/json" },
          //       body: JSON.stringify({
          //         node_id: node.id,
          //         frame_index: dialogFrameIndex,
          //       }),
          //     })
          //     .catch((err) =>
          //       console.error("Failed to delete painted keyframe", err),
          //     );
          //   // Redraw to remove the old painted mask before starting to draw the bbox
          //   drawCanvas();
          // }

          isDrawing = true;
          maskRect = { x: startX, y: startY, width: 0, height: 0 };
        }
      }
    }

    // Attach document-level listeners so dragging works even outside the canvas
    document.addEventListener("pointermove", handlePointerMove);
    document.addEventListener("pointerup", handlePointerUp);

    drawCanvas();
  };

  canvas.addEventListener("pointerdown", handlePointerDown);

  const updateBrushPreview = (e) => {
    if (maskMode === "paint") {
      const canvasRect = canvas.getBoundingClientRect();
      const containerRect = canvasContainer.getBoundingClientRect();
      const scaleFactor = canvasRect.width / canvas.width;
      const visualBrushSize = brushSize * 2 * scaleFactor;

      brushPreview.style.display = "block";
      brushPreview.style.left = `${e.clientX - containerRect.left - visualBrushSize / 2}px`;
      brushPreview.style.top = `${e.clientY - containerRect.top - visualBrushSize / 2}px`;
      brushPreview.style.width = `${visualBrushSize}px`;
      brushPreview.style.height = `${visualBrushSize}px`;
    } else {
      brushPreview.style.display = "none";
    }
  };

  // Update brush preview position and visibility
  canvasContainer.addEventListener("pointermove", updateBrushPreview);

  // Hide brush preview when mouse leaves canvas container
  canvasContainer.addEventListener("pointerleave", () => {
    brushPreview.style.display = "none";
  });

  // Show brush preview when mouse enters canvas container (if in paint mode)
  canvasContainer.addEventListener("pointerenter", () => {
    if (maskMode === "paint") {
      brushPreview.style.display = "block";
    }
  });

  const handlePointerMove = (e) => {
    // Always update brush preview in paint mode
    if (maskMode === "paint") {
      updateBrushPreview(e);
    }
    const coords = toCanvasCoords(e);
    let currentX = coords.x;
    let currentY = coords.y;

    // Don't clamp coordinates during active operations - let them extend beyond canvas
    // This allows smooth dragging when cursor goes outside the video area

    // Handle painting - draw directly to mask buffer
    if (isPainting && paintMaskData) {
      e.preventDefault();
      e.stopPropagation();

      // Draw line from last position to current
      paintLine(
        lastPaintX,
        lastPaintY,
        currentX,
        currentY,
        brushSize,
        currentPaintColor,
      );

      lastPaintX = currentX;
      lastPaintY = currentY;

      // Schedule render only once per frame
      if (!paintRenderScheduled) {
        paintRenderScheduled = true;
        requestAnimationFrame(() => {
          drawCanvas();
          paintRenderScheduled = false;
        });
      }
    }
    // Handle drawing new mask
    else if (isDrawing) {
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

  const handlePointerUp = async (e) => {
    const wasPainting = isPainting;
    const wasDrawing = isDrawing;
    const wasDragging = isDragging;
    const wasResizing = isResizing;

    // Stop all actions immediately to prevent stray events
    isDrawing = false;
    isDragging = false;
    isResizing = false;
    isPainting = false;

    if (wasDrawing || wasDragging || wasResizing || wasPainting) {
      e.preventDefault();
      e.stopPropagation();

      // Save keyframe when user finishes drawing/painting
      if (
        maskMode === "bbox" &&
        maskRect &&
        (wasDrawing || wasDragging || wasResizing)
      ) {
        // Get or create keyframe, preserving any existing painted data
        const existingKeyframe = keyframes[dialogFrameIndex] || {};

        keyframes[dialogFrameIndex] = {
          type: existingKeyframe.mask_data ? "hybrid" : "bbox",
          bbox: { ...maskRect },
          mask_data: existingKeyframe.mask_data || null,
        };

        // Send to backend
        try {
          await api.fetchApi("/videomaskeditor/setkeyframe", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              node_id: node.id,
              frame_index: dialogFrameIndex,
              type: keyframes[dialogFrameIndex].type,
              bbox: keyframes[dialogFrameIndex].bbox,
              mask_data: keyframes[dialogFrameIndex].mask_data,
            }),
          });
        } catch (error) {
          console.error(
            "[VideoMaskEditor] Failed to save bbox keyframe:",
            error,
          );
        }

        // Reset tracking so mask will reload properly when navigating
        lastRenderedKeyframeIndex = dialogFrameIndex;
        updateScrubber();
        drawCanvas(); // Redraw to finalize bbox
      } else if (maskMode === "paint" && wasPainting && paintMaskData) {
        // Save painted keyframe
        // Convert to float array (0 = painted/selected = 1.0, 255 = not painted = 0.0)
        const maskData = new Float32Array(paintMaskData.length);
        for (let i = 0; i < paintMaskData.length; i++) {
          // Invert: 0 (black/painted) -> 1.0, 255 (white/not painted) -> 0.0
          maskData[i] = 1.0 - paintMaskData[i] / 255.0;
        }

        // Encode to base64
        const buffer = maskData.buffer;
        const bytes = new Uint8Array(buffer);
        let binary = "";
        for (let i = 0; i < bytes.length; i++) {
          binary += String.fromCharCode(bytes[i]);
        }
        const base64 = btoa(binary);

        // Get or create keyframe, preserving any existing bbox data
        const existingKeyframe = keyframes[dialogFrameIndex] || {};

        keyframes[dialogFrameIndex] = {
          type: existingKeyframe.bbox ? "hybrid" : "painted",
          bbox: existingKeyframe.bbox || null,
          mask_data: base64,
        };

        // Send to backend
        try {
          await api.fetchApi("/videomaskeditor/setkeyframe", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              node_id: node.id,
              frame_index: dialogFrameIndex,
              type: keyframes[dialogFrameIndex].type,
              bbox: keyframes[dialogFrameIndex].bbox,
              mask_data: keyframes[dialogFrameIndex].mask_data,
            }),
          });
        } catch (error) {
          console.error(
            "[VideoMaskEditor] Failed to save painted keyframe:",
            error,
          );
        }

        // Reset tracking so mask will reload properly when navigating
        lastRenderedKeyframeIndex = dialogFrameIndex;
        updateScrubber();
        // Redraw to clear live painting buffer and show final saved mask
        drawCanvas();
      }
    }

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

  // Scroll wheel to adjust brush size in paint mode
  canvas.addEventListener("wheel", (e) => {
    if (maskMode === "paint") {
      e.preventDefault();
      e.stopPropagation();

      // Scroll up = larger brush, scroll down = smaller brush
      if (e.deltaY < 0) {
        brushSize = Math.min(100, brushSize + 2);
      } else {
        brushSize = Math.max(5, brushSize - 2);
      }

      // Update brush preview size with proper scaling
      const canvasRect = canvas.getBoundingClientRect();
      const scaleFactor = canvasRect.width / canvas.width;
      const visualBrushSize = brushSize * 2 * scaleFactor;
      brushPreview.style.width = `${visualBrushSize}px`;
      brushPreview.style.height = `${visualBrushSize}px`;

      // Show brush size indicator temporarily
      console.log(`[VideoMaskEditor] Brush size: ${brushSize}px`);
    }
  });

  // Keyboard shortcuts - use capture phase to intercept before other handlers
  const handleKeyDown = (e) => {
    // Space: play/pause
    if (e.code === "Space") {
      e.preventDefault();
      e.stopPropagation();
      e.stopImmediatePropagation();
      togglePlayPause();
    }
    // Arrow left: previous frame
    else if (e.code === "ArrowLeft") {
      e.preventDefault();
      e.stopPropagation();
      e.stopImmediatePropagation();
      goToPreviousFrame();
    }
    // Arrow right: next frame
    else if (e.code === "ArrowRight") {
      e.preventDefault();
      e.stopPropagation();
      e.stopImmediatePropagation();
      goToNextFrame();
    }
  };

  // Use capture phase (true) to intercept keys before other listeners
  document.addEventListener("keydown", handleKeyDown, true);

  // Buttons container
  const buttonsContainer = document.createElement("div");
  buttonsContainer.style.cssText = `
    display: flex;
    gap: 10px;
    justify-content: space-between;
    position: relative;
    z-index: 10;
  `;

  const createActionButton = (label, background, onClick) => {
    const button = document.createElement("button");
    button.textContent = label;
    button.style.cssText = `
      padding: 8px 16px;
      background: ${background};
      color: #fff;
      border: none;
      border-radius: 4px;
      font-size: 14px;
    `;
    button.style.cursor = "pointer";
    if (onClick) {
      button.addEventListener("click", onClick);
    }
    return button;
  };

  // Left-side button group
  const leftButtonGroup = document.createElement("div");
  leftButtonGroup.style.cssText = "display: flex; gap: 10px;";

  // Clear button - clears mask for current frame
  const clearButton = createActionButton("Clear", "#555", async () => {
    // If there's a keyframe at the current frame, delete it
    if (keyframes[dialogFrameIndex]) {
      delete keyframes[dialogFrameIndex];

      try {
        await api.fetchApi("/videomaskeditor/deletekeyframe", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            node_id: node.id,
            frame_index: dialogFrameIndex,
          }),
        });
      } catch (error) {
        console.error("[VideoMaskEditor] Failed to delete keyframe:", error);
      }

      // Clear current state completely
      maskRect = null;
      paintMaskData = null;
      paintMaskFrameIndex = -1;
      lastRenderedKeyframeIndex = -1;
      isDrawing = false;
      isDragging = false;
      isResizing = false;
      isPainting = false;
      updateScrubber();
      // Force complete redraw
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      drawCanvas();
    } else {
      // If no keyframe at current frame, create an empty keyframe to override any previous masks
      // This ensures the frame shows "no mask" instead of inheriting from a previous keyframe
      keyframes[dialogFrameIndex] = {
        type: "empty",
        bbox: null,
      };

      try {
        await api.fetchApi("/videomaskeditor/setkeyframe", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            node_id: node.id,
            frame_index: dialogFrameIndex,
            type: "empty",
            mask_data: null,
          }),
        });
      } catch (error) {
        console.error("[VideoMaskEditor] Failed to set empty keyframe:", error);
      }

      maskRect = null;
      paintMaskData = null;
      lastRenderedKeyframeIndex = dialogFrameIndex;
      updateScrubber();
      drawCanvas();
    }
  });
  leftButtonGroup.appendChild(clearButton);

  // Clear All button - clears all keyframes
  const clearAllButton = createActionButton("Clear All", "#555", async () => {
    const allKeyframeIndices = Object.keys(keyframes).map((k) => parseInt(k));

    // Clear local keyframes immediately (even if empty)
    keyframes = {};

    // Delete all keyframes from backend
    for (const kfIdx of allKeyframeIndices) {
      try {
        await api.fetchApi("/videomaskeditor/deletekeyframe", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            node_id: node.id,
            frame_index: kfIdx,
          }),
        });
      } catch (error) {
        console.error("[VideoMaskEditor] Failed to delete keyframe:", error);
      }
    }

    // Clear current state completely
    maskRect = null;
    paintMaskData = null;
    paintMaskFrameIndex = -1;
    lastRenderedKeyframeIndex = -1;
    isDrawing = false;
    isDragging = false;
    isResizing = false;
    isPainting = false;

    // Force complete redraw - clear the dialog canvas to show unmasked frames
    updateScrubber();
    drawCanvas();
  });
  leftButtonGroup.appendChild(clearAllButton);

  // Add left button group to container
  buttonsContainer.appendChild(leftButtonGroup);

  // Apply button
  const applyButton = createActionButton("Apply", "#0066cc", null);
  buttonsContainer.appendChild(applyButton);

  dialog.appendChild(buttonsContainer);
  overlay.appendChild(dialog);
  document.body.appendChild(overlay);

  // Load unmasked frames for the dialog
  const loadUnmaskedFrames = async () => {
    try {
      const sourceWidget = node.widgets?.find((w) => w.name === "source");
      const sourceValue = sourceWidget?.value;
      if (!sourceValue) {
        console.warn("[VideoMaskEditor] No source video found");
        return;
      }

      const getWidgetValue = (name, fallback) => {
        const widget = node.widgets?.find((w) => w.name === name);
        return widget && widget.value !== undefined ? widget.value : fallback;
      };

      const params = new URLSearchParams({
        video: sourceValue,
        node_id: node.id,
        framerate: getWidgetValue("framerate", 0) || 0,
        custom_width: getWidgetValue("custom_width", 0) || 0,
        custom_height: getWidgetValue("custom_height", 0) || 0,
        frame_load_cap: getWidgetValue("frame_load_cap", 0) || 0,
        skip_first_frames: getWidgetValue("skip_first_frames", 0) || 0,
        select_every_nth: getWidgetValue("select_every_nth", 1) || 1,
        format: getWidgetValue("format", "None") || "None",
        max_preview_frames: 120,
        skip_mask: "true", // IMPORTANT: Skip masks for the dialog editor
      });

      const response = await api.fetchApi(
        `/videomaskeditor/preview?${params.toString()}`,
      );
      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || "Failed to load preview");
      }

      const data = await response.json();
      const frames = data.frames || [];
      if (!frames.length) {
        console.warn("[VideoMaskEditor] No frames loaded");
        return;
      }

      dialogFrames.frameDuration = data.fps > 0 ? 1000 / data.fps : 100;
      dialogFrames.frames = new Array(frames.length);

      console.log(
        `[VideoMaskEditor] Loaded ${frames.length} unmasked frames at ${data.fps} fps`,
      );

      // Progressive loading: load first 8 frames immediately, then load rest in parallel chunks
      const INITIAL_FRAMES = Math.min(8, frames.length);
      const CHUNK_SIZE = 8;

      const decodeFrame = (frameInfo, idx) => {
        return new Promise((resolve) => {
          const img = new Image();
          img.src = "data:image/webp;base64," + frameInfo.data;

          img.onload = () => {
            const tempCanvas = document.createElement("canvas");
            tempCanvas.width = img.width;
            tempCanvas.height = img.height;
            const ctx = tempCanvas.getContext("2d");
            ctx.drawImage(img, 0, 0);
            dialogFrames.frames[idx] = {
              imageData: ctx.getImageData(0, 0, img.width, img.height),
              width: img.width,
              height: img.height,
            };
            resolve(true);
          };

          img.onerror = () => {
            console.error(`[VideoMaskEditor] Failed to load frame ${idx}`);
            resolve(false);
          };
        });
      };

      // Load first frames to show something quickly
      for (let idx = 0; idx < INITIAL_FRAMES; idx++) {
        await decodeFrame(frames[idx], idx);
      }

      // Start animation with initial frames
      dialogFrames.loading = false;
      drawCanvas();
      updateScrubber();
      startAnimation();

      console.log(
        `[VideoMaskEditor] Initial ${INITIAL_FRAMES} frames loaded, loading rest...`,
      );

      // Load remaining frames in parallel chunks (non-blocking)
      if (frames.length > INITIAL_FRAMES) {
        const loadRemainingFrames = async () => {
          for (
            let chunkStart = INITIAL_FRAMES;
            chunkStart < frames.length;
            chunkStart += CHUNK_SIZE
          ) {
            const chunkEnd = Math.min(chunkStart + CHUNK_SIZE, frames.length);
            const chunkPromises = [];

            for (let idx = chunkStart; idx < chunkEnd; idx++) {
              chunkPromises.push(decodeFrame(frames[idx], idx));
            }

            await Promise.all(chunkPromises);
          }

          const loadedCount = dialogFrames.frames.filter(Boolean).length;
          console.log(
            `[VideoMaskEditor] All frames loaded: ${loadedCount}/${frames.length}`,
          );
        };

        // Load rest in background
        loadRemainingFrames().catch((err) => {
          console.error(
            "[VideoMaskEditor] Error loading remaining frames:",
            err,
          );
        });
      }
    } catch (error) {
      console.error("[VideoMaskEditor] Error loading unmasked frames:", error);
      dialogFrames.loading = false;
    }
  };

  // Initial load - keyframes and progressive frame loading
  Promise.all([loadKeyframes(), loadUnmaskedFrames()]);

  // Helper function to close dialog
  const closeDialog = () => {
    stopAnimation();
    document.removeEventListener("keydown", handleKeyDown, true);
    document.body.removeChild(overlay);
  };

  const handleCancel = async () => {
    // Restore initial state
    maskRect = initialMaskRect ? { ...initialMaskRect } : null;
    node.maskRegion = maskRect;

    // Clear all keyframes that were added/modified during this session
    // by restoring the backend state to what it was when we opened the dialog
    try {
      await api.fetchApi("/videomaskeditor/restorekeyframes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          node_id: node.id,
          keyframes: initialKeyframes,
        }),
      });
    } catch (error) {
      console.error(
        "[VideoMaskEditor] Failed to restore keyframes on cancel:",
        error,
      );
    }

    closeDialog();
  };

  closeButton.addEventListener("click", handleCancel);

  applyButton.addEventListener("click", () => {
    // Manually trigger a refresh of the main node's preview to ensure
    // it reflects the final state from the editor.
    if (node.refreshPreview) {
      node.refreshPreview();
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
        background-image:
          linear-gradient(45deg, #383838 25%, transparent 25%),
          linear-gradient(-45deg, #383838 25%, transparent 25%),
          linear-gradient(45deg, transparent 75%, #383838 75%),
          linear-gradient(-45deg, transparent 75%, #383838 75%);
        background-size: 20px 20px;
        background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
        background-color: #2b2b2b;
        margin-bottom: 8px;
        display: none;
      `;
      element.appendChild(previewWidget.parentEl);

      previewWidget.canvasEl = document.createElement("canvas");
      previewWidget.canvasEl.style.width = "100%";
      previewWidget.canvasEl.style.height = "auto";
      previewWidget.canvasEl.style.display = "block";

      const getWidgetValue = (name, fallback) => {
        const widget = previewNode.widgets?.find((w) => w.name === name);
        return widget && widget.value !== undefined ? widget.value : fallback;
      };

      const previewState = {
        pendingTimeout: null,
        requestId: 0,
      };

      // Define loadInputPreview early so it can be used by refresh logic
      const loadInputPreview = async () => {
        previewState.pendingTimeout = null;
        const sourceValue = getWidgetValue("source", null);
        if (!sourceValue) {
          stopAnimation();
          previewWidget.parentEl.hidden = true;
          previewWidget.frames = [];
          return;
        }

        const params = new URLSearchParams({
          video: sourceValue,
          node_id: previewNode.id, // Pass node_id for server-side masking
          framerate: getWidgetValue("framerate", 0) || 0,
          custom_width: getWidgetValue("custom_width", 0) || 0,
          custom_height: getWidgetValue("custom_height", 0) || 0,
          frame_load_cap: getWidgetValue("frame_load_cap", 0) || 0,
          skip_first_frames: getWidgetValue("skip_first_frames", 0) || 0,
          select_every_nth: getWidgetValue("select_every_nth", 1) || 1,
          format: getWidgetValue("format", "None") || "None",
          max_preview_frames: 120,
          skip_mask: "false", // Apply masks to main preview
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

          previewWidget.frameDuration = data.fps > 0 ? 1000 / data.fps : 100;
          previewWidget.frames = new Array(frames.length);

          console.log(
            `[VideoPreview] Loaded ${frames.length} frames at ${data.fps} fps (frame duration: ${previewWidget.frameDuration}ms)`,
          );

          let loadedCount = 0;
          frames.forEach((frameInfo, idx) => {
            const img = new Image();
            img.src = "data:image/webp;base64," + frameInfo.data;
            img.onload = () => {
              const canvas = document.createElement("canvas");
              canvas.width = img.width;
              canvas.height = img.height;
              const ctx = canvas.getContext("2d");
              ctx.drawImage(img, 0, 0);
              previewWidget.frames[idx] = {
                imageData: ctx.getImageData(0, 0, img.width, img.height),
                width: img.width,
                height: img.height,
              };

              loadedCount++;
              if (loadedCount === frames.length) {
                previewWidget.parentEl.style.display = "block";
                startAnimation();
              }
            };
          });
        } catch (error) {
          console.error("[VideoMaskEditor] Error loading preview:", error);
          stopAnimation();
          previewWidget.parentEl.style.display = "none";
          previewWidget.frames = [];
        }
      };

      // --- WAN Frame Snapping Logic ---
      const isWanWidget = this.widgets.find((w) => w.name === "is_wan");
      const frameLoadCapWidget = this.widgets.find(
        (w) => w.name === "frame_load_cap",
      );

      if (isWanWidget && frameLoadCapWidget) {
        let lastKnownValue = Number(frameLoadCapWidget.value);

        const snapLogic = () => {
          if (!isWanWidget.value) {
            // When WAN is off, just track the value.
            lastKnownValue = Number(frameLoadCapWidget.value);
            return;
          }

          const currentValue = Number(frameLoadCapWidget.value);
          if (isNaN(currentValue)) return;

          let snappedValue;
          const delta = currentValue - lastKnownValue;

          // Heuristic: A change of exactly +1 or -1 indicates an arrow click.
          if (delta === 1) {
            // Incrementing: find next WAN count up from the last valid number.
            const n = Math.floor((lastKnownValue - 1) / 4);
            snappedValue = 1 + (n + 1) * 4;
          } else if (delta === -1) {
            // Decrementing: find next WAN count down.
            const n = Math.ceil((lastKnownValue - 1) / 4);
            snappedValue = 1 + (n - 1) * 4;
          } else {
            // Typing or other direct change: snap to nearest valid value.
            snappedValue =
              currentValue <= 1
                ? 1
                : 1 + Math.round((currentValue - 1) / 4) * 4;
          }

          snappedValue = Math.max(1, snappedValue);

          // Update widget value and internal state
          frameLoadCapWidget.value = snappedValue;
          if (frameLoadCapWidget.inputEl) {
            frameLoadCapWidget.inputEl.value = snappedValue;
          }
          lastKnownValue = snappedValue;
        };

        // Monkey-patch callbacks to inject our logic
        const originalIsWanCallback = isWanWidget.callback;
        isWanWidget.callback = function (value) {
          originalIsWanCallback?.apply(this, arguments);
          if (value) {
            // When turning WAN on, snap the current value immediately.
            snapLogic();
          }
        };

        const originalFrameCapCallback = frameLoadCapWidget.callback;
        frameLoadCapWidget.callback = function (value) {
          originalFrameCapCallback?.apply(this, arguments);
          snapLogic();
        };

        // Perform an initial check in case the node loads with is_wan enabled.
        if (isWanWidget.value) {
          snapLogic();
        }
      }
      // --- End WAN Frame Snapping Logic ---

      previewNode.keyframes = {}; // Initialize keyframes store

      const scheduleLoad = () => {
        if (previewState.pendingTimeout) {
          clearTimeout(previewState.pendingTimeout);
        }
        previewState.pendingTimeout = setTimeout(() => {
          loadInputPreview();
        }, 200);
      };

      const getKeyframes = async () => {
        try {
          const response = await api.fetchApi(
            `/videomaskeditor/getkeyframes?node_id=${previewNode.id}`,
          );
          if (response.ok) {
            const data = await response.json();
            previewNode.keyframes = data.keyframes || {};
          }
        } catch (error) {
          console.error("[VideoMaskEditor] Failed to get keyframes:", error);
        }
      };

      const refreshPreview = async () => {
        await getKeyframes();
        await loadInputPreview();
      };
      previewNode.refreshPreview = refreshPreview;

      api.addEventListener("videomaskeditor.mask_updated", async (e) => {
        if (e.detail.node_id === previewNode.id) {
          await getKeyframes();
        }
      });

      // When a widget that affects the preview changes, schedule a reload
      const widgetsToWatch = [
        "source",
        "framerate",
        "custom_width",
        "custom_height",
        "frame_load_cap",
        "skip_first_frames",
        "select_every_nth",
      ];
      for (const widgetName of widgetsToWatch) {
        const widget = previewNode.widgets.find((w) => w.name === widgetName);
        if (widget) {
          const originalCallback = widget.callback;
          widget.callback = function () {
            if (originalCallback) {
              originalCallback.apply(this, arguments);
            }
            scheduleLoad();
          };
        }
      }

      // Initial fetch of keyframes for the editor and initial preview load
      getKeyframes();
      scheduleLoad();

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
        const ctx = previewWidget.canvasEl.getContext("2d", { alpha: true });
        // Clear canvas with transparency
        ctx.clearRect(0, 0, frameData.width, frameData.height);
        ctx.putImageData(frameData.imageData, 0, 0);

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
      const sourceWidget = this.widgets?.find((w) => w.name === "source");
      const triggerPreviewUpdate = () => {
        if (this._vmeScheduleInputPreview) {
          this._vmeScheduleInputPreview();
        }
      };

      if (sourceWidget) {
        const originalVideoCallback = sourceWidget.callback;
        sourceWidget.callback = function (value) {
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
      fileInput.accept = "video/*,.webm,.mp4,.mkv,.gif,.mov,image/*";
      fileInput.setAttribute("webkitdirectory", "");
      fileInput.setAttribute("directory", "");
      fileInput.setAttribute("multiple", "");
      fileInput.style.display = "none";

      fileInput.addEventListener("change", async (e) => {
        const files = Array.from(e.target.files);
        if (!files || files.length === 0) return;

        const button = this.widgets[this.widgets.length - 1];

        try {
          // Handle single video file upload
          if (files.length === 1 && files[0].type.startsWith("video/")) {
            button.label = "Uploading...";
            const formData = new FormData();
            formData.append("image", files[0]);
            formData.append("type", "input");

            const response = await api.fetchApi("/upload/image", {
              method: "POST",
              body: formData,
            });

            if (response.ok) {
              const data = await response.json();
              if (sourceWidget) {
                sourceWidget.value = data.name;
                if (sourceWidget.callback) {
                  sourceWidget.callback(data.name);
                }
              }
            } else {
              throw new Error(
                `Failed to upload video: ${await response.text()}`,
              );
            }
          }
          // Handle folder of images upload
          else {
            button.label = "Uploading...";
            const sortedFiles = files.sort((a, b) =>
              a.webkitRelativePath.localeCompare(b.webkitRelativePath),
            );

            let subfolder = "video_mask_upload";
            const topLevelDir = sortedFiles[0].webkitRelativePath.split("/")[0];
            if (topLevelDir) {
              subfolder = topLevelDir;
            }
            subfolder = `${subfolder}_${Date.now()}`;

            for (const [i, file] of sortedFiles.entries()) {
              button.label = `Uploading... (${i + 1}/${sortedFiles.length})`;
              const formData = new FormData();

              let filename = file.name;
              if (
                topLevelDir &&
                file.webkitRelativePath.startsWith(topLevelDir + "/")
              ) {
                filename = file.webkitRelativePath
                  .substring(topLevelDir.length + 1)
                  .replace(/\//g, "_");
              }
              if (!filename) {
                filename = file.name;
              }

              formData.append("image", file, filename);
              formData.append("subfolder", subfolder);
              formData.append("overwrite", "true");

              const response = await api.fetchApi("/upload/image", {
                method: "POST",
                body: formData,
              });

              if (!response.ok) {
                throw new Error(
                  `Failed to upload ${file.name}: ${await response.text()}`,
                );
              }
            }

            if (sourceWidget) {
              sourceWidget.value = subfolder;
              if (sourceWidget.callback) {
                sourceWidget.callback(subfolder);
              }
            }
          }
        } catch (error) {
          console.error("Upload failed:", error);
          alert("Upload failed: " + error.message);
        } finally {
          button.label = "choose video or folder";
          fileInput.value = "";
        }
      });
      document.body.appendChild(fileInput);

      this.addWidget("button", "choose video or folder", "upload", () => {
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
