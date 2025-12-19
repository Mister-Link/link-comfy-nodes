import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const MAX_INPUTS = 8;
const INPUT_PREFIX = "input";
const INPUT_TYPE = "*";

const getInputName = (index) => `${INPUT_PREFIX}${index}`;
const getPrefixName = (index) => `${INPUT_PREFIX}${index}_prefix`;

const removeWidget = (node, name) => {
  if (!node.widgets) {
    return;
  }
  const index = node.widgets.findIndex((widget) => widget.name === name);
  if (index !== -1) {
    node.widgets.splice(index, 1);
  }
};

const ensurePrefixWidget = (node, index) => {
  const name = getPrefixName(index);
  const existing = node.widgets?.find((widget) => widget.name === name);
  if (existing) {
    return existing;
  }
  return node.addWidget("text", name, `${INPUT_PREFIX}${index}`);
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
    removeWidget(node, getPrefixName(i));
  }

  for (let i = 1; i <= lastConnected; i += 1) {
    ensurePrefixWidget(node, i);
  }
  for (let i = lastConnected + 1; i <= MAX_INPUTS; i += 1) {
    removeWidget(node, getPrefixName(i));
  }

  moveWidgetToEnd(node, node._zipDownloadButton);
  node.setDirtyCanvas(true, true);
};

app.registerExtension({
  name: "SaveToZip.DynamicInputs",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "Save To Zip") {
      return;
    }

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      this.downloadUrl = null;
      this.zipFilename = null;

      if (this.inputs?.length) {
        const toRemove = this.inputs
          .map((input) => input.name)
          .filter(
            (name) =>
              name === "frames" ||
              name === "alpha" ||
              name.startsWith(INPUT_PREFIX),
          );
        toRemove.forEach((name) => removeInputSlot(this, name));
      }

      if (this.widgets?.length) {
        this.widgets
          .map((widget) => widget.name)
          .filter(
            (name) =>
              name === "prefix" ||
              name?.endsWith("_ext") ||
              name?.endsWith("_prefix"),
          )
          .forEach((name) => removeWidget(this, name));
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
      if (slotType === 1 && linkInfo && event === true) {
        const fromNode = this.graph?._nodes?.find(
          (otherNode) => otherNode.id === linkInfo.origin_id,
        );
        if (fromNode) {
          const parentLink = fromNode.outputs?.[linkInfo.origin_slot];
          if (parentLink) {
            nodeSlot.type = parentLink.type;
          }
        }
      }
      syncDynamicInputs(this);
      return result;
    };
  },

  async nodeCreated(node) {
    if (node.comfyClass !== "Save To Zip") {
      return;
    }

    const downloadButton = node.addWidget("button", "No File", null, () => {
      if (!node.downloadUrl) {
        return;
      }
      const link = document.createElement("a");
      link.href = api.apiURL(node.downloadUrl);
      link.download = node.zipFilename || "save_to_zip.zip";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    });

    downloadButton.serialize = false;
    downloadButton.disabled = true;
    node._zipDownloadButton = downloadButton;
    moveWidgetToEnd(node, downloadButton);

    const originalOnExecuted = node.onExecuted;
    node.onExecuted = function (message) {
      if (originalOnExecuted) {
        originalOnExecuted.apply(this, arguments);
      }

      if (message?.text && message.text.length > 0) {
        const textHtml = message.text[0];
        const urlMatch = textHtml.match(/href="([^"]+)"/);
        const filenameMatch = textHtml.match(/Download:\s*([^<]+)/);

        if (urlMatch) {
          node.downloadUrl = urlMatch[1];
          node.zipFilename = filenameMatch
            ? filenameMatch[1].trim()
            : "save_to_zip.zip";
          downloadButton.label = `💾 ${node.zipFilename}`;
          downloadButton.disabled = false;
          console.log(
            `[SaveToZip] ZIP ready for download: ${node.zipFilename}`,
          );
        } else {
          node.downloadUrl = null;
          node.zipFilename = null;
          downloadButton.label = "No File";
          downloadButton.disabled = true;
        }
      }
    };
  },
});
