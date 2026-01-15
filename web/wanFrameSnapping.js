import { app } from "../../scripts/app.js";

// Snap value to nearest valid WAN frame count (4n+1, minimum 5)
// Valid values: 0 (disabled), 5, 9, 13, 17, 21, 25, 29, ...
function snapToWAN(value) {
  if (value === 0) return 0;

  const n = Math.max(1, Math.floor((value - 1) / 4));
  const snapped = 4 * n + 1;
  return Math.max(5, snapped);
}

app.registerExtension({
  name: "LinkComfy.WANFrameSnapping",

  async nodeCreated(node) {
    if (node.comfyClass !== "VideoMaskEditor") {
      return;
    }

    // Find the frame_load_cap and is_wan widgets
    const frameCapWidget = node.widgets?.find(
      (w) => w.name === "frame_load_cap",
    );
    const isWANWidget = node.widgets?.find((w) => w.name === "is_wan");

    if (!frameCapWidget || !isWANWidget) {
      return;
    }

    // Store original callback
    const originalFrameCapCallback = frameCapWidget.callback;

    // Track the last snapped value to detect when user is trying to change
    let lastSnappedValue = snapToWAN(frameCapWidget.value);

    // Override the callback to snap values when they change
    frameCapWidget.callback = function (value) {
      let finalValue = value;

      // If WAN mode is enabled, snap to valid WAN values
      if (isWANWidget.value === true) {
        const snappedValue = snapToWAN(value);

        // Check if user is trying to increase/decrease from the last snapped value
        if (value > lastSnappedValue) {
          // User is trying to increase - jump to next WAN value
          finalValue =
            snappedValue === lastSnappedValue ? snappedValue + 4 : snappedValue;
        } else if (value < lastSnappedValue) {
          // User is trying to decrease - jump to previous WAN value
          finalValue = snappedValue;
        } else {
          // Value equals last snapped value - keep it
          finalValue = snappedValue;
        }

        // Ensure we never go below 0 or minimum WAN value
        if (finalValue > 0 && finalValue < 5) {
          finalValue = 5;
        }

        // Update tracked value
        lastSnappedValue = finalValue;

        // Update widget to show snapped value
        if (frameCapWidget.value !== finalValue) {
          frameCapWidget.value = finalValue;
        }
      }

      // Call original callback with the final value
      if (originalFrameCapCallback) {
        return originalFrameCapCallback.call(this, finalValue);
      }
      return finalValue;
    };

    // When is_wan changes, snap the current frame_load_cap value
    const originalIsWANCallback = isWANWidget.callback;
    isWANWidget.callback = function (value) {
      if (value === true) {
        // WAN mode enabled - snap current frame_load_cap
        const currentValue = frameCapWidget.value;
        const snapped = snapToWAN(currentValue);
        if (snapped !== currentValue) {
          lastSnappedValue = snapped;
          frameCapWidget.value = snapped;
          if (frameCapWidget.callback) {
            frameCapWidget.callback(snapped);
          }
        } else {
          lastSnappedValue = snapped;
        }
      }

      if (originalIsWANCallback) {
        return originalIsWANCallback.call(this, value);
      }
      return value;
    };
  },
});
