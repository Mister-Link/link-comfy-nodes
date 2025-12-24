import { app } from "../../scripts/app.js";

const MAX_INPUTS = 8;
const INPUT_PREFIX = "string";
const INPUT_TYPE = "STRING";

const getInputName = (index) => `${INPUT_PREFIX}${index}`;

const removeWidget = (node, name) => {
  if (!node.widgets) {
    return;
  }
  const index = node.widgets.findIndex((widget) => widget.name === name);
  if (index !== -1) {
    node.widgets.splice(index, 1);
  }
};

const ensureInputSlot = (node, index) => {
  const name = getInputName(index);
  const existing = node.inputs?.find((input) => input.name === name);
  if (existing) {
    return existing;
  }
  node.addInput(name, INPUT_TYPE);
  const slot = node.inputs[node.inputs.length - 1];
  if (slot) {
    slot.color_off = "#666";
  }
  return slot;
};

const removeInputSlot = (node, name) => {
  const index = node.inputs?.findIndex((input) => input.name === name);
  if (index != null && index >= 0) {
    node.removeInput(index);
  }
};

const moveWidgetToEnd = (node, widget) => {
  if (!node.widgets || !widget) {
    return;
  }
  const index = node.widgets.indexOf(widget);
  if (index === -1 || index === node.widgets.length - 1) {
    return;
  }
  node.widgets.splice(index, 1);
  node.widgets.push(widget);
};

const syncDynamicInputs = (node) => {
  const inputs = [];
  for (let i = 1; i <= MAX_INPUTS; i += 1) {
    inputs.push(node.inputs?.find((input) => input.name === getInputName(i)));
  }

  let lastConnected = 0;
  inputs.forEach((input, index) => {
    if (input?.link != null) {
      lastConnected = index + 1;
    }
  });

  const maxVisible = Math.min(MAX_INPUTS, lastConnected + 1);
  for (let i = 1; i <= maxVisible; i += 1) {
    ensureInputSlot(node, i);
  }
  for (let i = maxVisible + 1; i <= MAX_INPUTS; i += 1) {
    removeInputSlot(node, getInputName(i));
  }

  // Move template widget to end
  const templateWidget = node.widgets?.find((w) => w.name === "template");
  if (templateWidget) {
    moveWidgetToEnd(node, templateWidget);
  }

  node.setDirtyCanvas(true, true);
};

app.registerExtension({
  name: "AdvancedStringConcat.DynamicInputs",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "Advanced String Concat") {
      return;
    }

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);

      // Remove any pre-existing string inputs from Python definition
      if (this.inputs?.length) {
        const toRemove = this.inputs
          .map((input) => input.name)
          .filter((name) => name.startsWith(INPUT_PREFIX));
        toRemove.forEach((name) => removeInputSlot(this, name));
      }

      syncDynamicInputs(this);

      return result;
    };

    const originalOnConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (
      slotType,
      slotIndex,
      event,
      linkInfo,
      nodeSlot,
    ) {
      const result = originalOnConnectionsChange?.apply(this, arguments);
      syncDynamicInputs(this);
      return result;
    };
  },
});
