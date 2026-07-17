import { app } from "../../scripts/app.js";

const NODE_NAME = "Palettize";
const SOURCE_EXTERNAL = "External Swatch";
const SOURCE_FROM_IMAGE = "From Image";

const findWidget = (node, name) => node.widgets?.find((widget) => widget.name === name);

function setWidgetVisible(widget, visible) {
  if (!widget) {
    return;
  }
  if (widget._lcOrigType === undefined) {
    widget._lcOrigType = widget.type;
    widget._lcOrigComputeSize = widget.computeSize;
  }
  if (visible) {
    widget.type = widget._lcOrigType;
    widget.computeSize = widget._lcOrigComputeSize;
  } else {
    widget.type = "lc_palettize_hidden";
    widget.computeSize = () => [0, -4];
  }
}

function updateVisibility(node) {
  const sourceWidget = findWidget(node, "swatch_source");
  if (!sourceWidget) {
    return;
  }
  const pathWidget = findWidget(node, "swatch_path");
  const numColorsWidget = findWidget(node, "num_colors");

  setWidgetVisible(pathWidget, sourceWidget.value === SOURCE_EXTERNAL);
  setWidgetVisible(numColorsWidget, sourceWidget.value === SOURCE_FROM_IMAGE);

  node.setSize(node.computeSize());
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

      const sourceWidget = findWidget(this, "swatch_source");
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
