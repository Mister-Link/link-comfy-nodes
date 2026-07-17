import { app } from "../../scripts/app.js";

const NODE_NAME = "Palettize";
const SOURCE_EXTERNAL = "External Swatch";
const SOURCE_FROM_INPUT = "From Input";
const EXTRA_HEIGHT_PER_FIELD = 34; // room for each conditional widget row so it isn't cramped against the node edge

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
  const sourceWidget = findWidget(node, "swatch_source");
  if (!sourceWidget) {
    return;
  }
  const pathWidget = findWidget(node, "swatch_path");
  const numColorsWidget = findWidget(node, "num_colors");

  const showPath = sourceWidget.value === SOURCE_EXTERNAL;
  const showNumColors = sourceWidget.value !== SOURCE_EXTERNAL;
  setWidgetVisible(pathWidget, showPath);
  setWidgetVisible(numColorsWidget, showNumColors);

  const visibleExtraFields = (showPath ? 1 : 0) + (showNumColors ? 1 : 0);
  const computed = node.computeSize();
  const width = Math.max(computed[0], node.size?.[0] ?? 0);
  const height = computed[1] + visibleExtraFields * EXTRA_HEIGHT_PER_FIELD;
  node.setSize([width, height]);
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
