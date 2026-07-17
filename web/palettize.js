import { app } from "../../scripts/app.js";

const NODE_NAME = "Palettize";
const SOURCE_EXTERNAL = "External Swatch";
const MIN_NODE_WIDTH = 220; // a little wider than the intrinsic computed width so combo values aren't crowded

const findWidget = (node, name) => node.widgets?.find((widget) => widget.name === name);

function setWidgetVisible(widget, visible) {
  if (!widget) {
    return;
  }
  if (widget._lcOrigComputeSize === undefined) {
    widget._lcOrigComputeSize = widget.computeSize;
  }
  widget.hidden = !visible;
  widget.computeSize = visible ? widget._lcOrigComputeSize : () => [0, -4];
}

function updateVisibility(node) {
  const sourceWidget = findWidget(node, "swatch");
  if (!sourceWidget) {
    return;
  }
  const pathWidget = findWidget(node, "swatch_path");
  const numColorsWidget = findWidget(node, "num_colors");

  const showPath = sourceWidget.value === SOURCE_EXTERNAL;
  const showNumColors = sourceWidget.value !== SOURCE_EXTERNAL;
  setWidgetVisible(pathWidget, showPath);
  setWidgetVisible(numColorsWidget, showNumColors);

  // node.computeSize() already reflects only the currently-visible widgets
  // (widget.hidden is respected by LiteGraph's own layout pass), so the
  // default size can just be set to that intrinsic minimum directly instead
  // of padding it -- padding here was making the default bigger than what
  // dragging the resize handle down to its own minimum would give you.
  const computed = node.computeSize();
  node.setSize([Math.max(computed[0], MIN_NODE_WIDTH), computed[1]]);
  node.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "LinkComfyNodes.Palettize",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) {
      return;
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);

      // Patch computeSize (not just the size set at creation time) so the
      // wider floor also applies to the drag-to-shrink minimum, not just
      // the initial default size.
      const originalComputeSize = this.computeSize.bind(this);
      this.computeSize = function (...args) {
        const size = originalComputeSize(...args);
        size[0] = Math.max(size[0], MIN_NODE_WIDTH);
        return size;
      };

      const sourceWidget = findWidget(this, "swatch");
      if (sourceWidget) {
        const originalCallback = sourceWidget.callback;
        sourceWidget.callback = (...args) => {
          const res = originalCallback?.apply(sourceWidget, args);
          updateVisibility(this);
          return res;
        };
      }

      updateVisibility(this);
      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      updateVisibility(this);
      return result;
    };
  },
});
