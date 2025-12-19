import { app } from "../../scripts/app.js";

const removeWidget = (node, name) => {
  if (!node.widgets) {
    return null;
  }
  const index = node.widgets.findIndex((widget) => widget.name === name);
  if (index === -1) {
    return null;
  }
  return node.widgets.splice(index, 1)[0] || null;
};

const restoreWidget = (node, widget, index) => {
  if (!node.widgets || !widget) {
    return;
  }
  if (node.widgets.includes(widget)) {
    return;
  }
  const insertAt = index != null && index >= 0 ? index : node.widgets.length;
  node.widgets.splice(insertAt, 0, widget);
};

const updateNodeSize = (node) => {
  if (!node) {
    return;
  }
  node.computeSize?.();
  node.setDirtyCanvas?.(true, true);
};

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

      this._hiddenWidgets = this._hiddenWidgets || {};
      if (customWidthWidget) {
        this._hiddenWidgets.customWidthIndex =
          this.widgets?.indexOf(customWidthWidget) ?? null;
      }
      if (customHeightWidget) {
        this._hiddenWidgets.customHeightIndex =
          this.widgets?.indexOf(customHeightWidget) ?? null;
      }

      const updateVisibility = () => {
        const isCustom = presetWidget?.value === "Custom";
        if (isCustom) {
          restoreWidget(
            this,
            this._hiddenWidgets.customWidthWidget || customWidthWidget,
            this._hiddenWidgets.customWidthIndex,
          );
          restoreWidget(
            this,
            this._hiddenWidgets.customHeightWidget || customHeightWidget,
            this._hiddenWidgets.customHeightIndex,
          );
          delete this._hiddenWidgets.customWidthWidget;
          delete this._hiddenWidgets.customHeightWidget;
        } else {
          if (!this._hiddenWidgets.customWidthWidget) {
            this._hiddenWidgets.customWidthWidget = removeWidget(
              this,
              "custom_width",
            );
          }
          if (!this._hiddenWidgets.customHeightWidget) {
            this._hiddenWidgets.customHeightWidget = removeWidget(
              this,
              "custom_height",
            );
          }
        }
        updateNodeSize(this);
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

      const onConfigure = this.onConfigure;
      this.onConfigure = function () {
        const result = onConfigure?.apply(this, arguments);
        updateVisibility();
        return result;
      };

      return res;
    };
  },
});
