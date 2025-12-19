import { app } from "../../scripts/app.js";

function setWidgetVisible(widget, visible) {
  if (!widget) return;
  widget.hidden = !visible;
  if (widget.parentEl) {
    widget.parentEl.style.display = visible ? "" : "none";
  }
  if (widget.inputEl) {
    widget.inputEl.style.display = visible ? "" : "none";
  }
}

app.registerExtension({
  name: "Comfy.LinkComfy.PixelationDimensions",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "Pixelation Dimensions") {
      return;
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const res = onNodeCreated?.apply(this, arguments);
      const presetWidget = this.widgets?.find((w) => w.name === "preset");
      const customWidthWidget = this.widgets?.find(
        (w) => w.name === "custom_width",
      );
      const customHeightWidget = this.widgets?.find(
        (w) => w.name === "custom_height",
      );

      const updateVisibility = () => {
        const isCustom = presetWidget?.value === "Custom";
        setWidgetVisible(customWidthWidget, isCustom);
        setWidgetVisible(customHeightWidget, isCustom);
        this.computeSize?.();
        this.setDirtyCanvas?.(true, true);
      };

      if (presetWidget) {
        const originalCallback = presetWidget.callback;
        presetWidget.callback = function () {
          const callbackResult = originalCallback?.apply(this, arguments);
          updateVisibility();
          return callbackResult;
        };
      }

      updateVisibility();
      return res;
    };
  },
});
