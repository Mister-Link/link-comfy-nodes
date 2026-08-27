import { app } from "../../scripts/app.js";

const NODE_NAME = "Dropdown Select";
const STYLE_ID = "lc_dropdown_select_styles";
const ROW_HEIGHT = 26;
const CHROME_HEIGHT = 60;
const MIN_NODE_WIDTH = 260;

function ensureStyles() {
  if (document.getElementById(STYLE_ID)) {
    return;
  }

  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
    .lc-dropdown-select {
      box-sizing: border-box;
      width: 100%;
      max-width: 100%;
      min-width: 0;
      padding: 4px 2px 0;
      display: flex;
      flex-direction: column;
      gap: 6px;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      font-size: 12px;
    }
    .lc-dropdown-select-rows {
      width: 100%;
      max-width: 100%;
      min-width: 0;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .lc-dropdown-select-row {
      width: 100%;
      max-width: 100%;
      min-width: 0;
      display: flex;
      align-items: center;
      gap: 4px;
    }
    .lc-dropdown-select-field {
      flex: 1;
      min-width: 0;
      box-sizing: border-box;
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 4px;
      color: #e6e6e6;
      font-size: 12px;
      font-family: inherit;
      padding: 3px 6px;
      cursor: pointer;
    }
    .lc-dropdown-select-label {
      user-select: none;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    input.lc-dropdown-select-field {
      cursor: text;
    }
    .lc-dropdown-select-row.lc-selected .lc-dropdown-select-field {
      background: rgba(90, 200, 130, 0.16);
      border-color: rgba(90, 200, 130, 0.6);
      color: #eafff1;
    }
    .lc-dropdown-select-remove {
      flex: 0 0 auto;
      width: 20px;
      height: 20px;
      line-height: 1;
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 4px;
      background: rgba(255, 255, 255, 0.06);
      color: #e6a6a6;
      cursor: pointer;
    }
    .lc-dropdown-select-remove:hover {
      background: rgba(255, 80, 80, 0.2);
    }
    .lc-dropdown-select-add {
      width: 100%;
      max-width: 100%;
      box-sizing: border-box;
      border: 1px dashed rgba(255, 255, 255, 0.2);
      border-radius: 4px;
      background: transparent;
      color: #9a9a9a;
      font-size: 12px;
      padding: 3px 6px;
      cursor: pointer;
    }
    .lc-dropdown-select-add:hover {
      color: #e6e6e6;
      border-color: rgba(255, 255, 255, 0.4);
    }
  `;

  document.head.appendChild(style);
}

const findWidget = (node, name) => node.widgets?.find((widget) => widget.name === name);

function hideWidgetForGood(widget) {
  widget.hidden = true;
  widget.computeSize = () => [0, -4];
}

function parseOptions(text) {
  const trimmed = (text ?? "").trim();
  if (trimmed.startsWith("[")) {
    try {
      const data = JSON.parse(trimmed);
      if (Array.isArray(data)) {
        return data.map((item) => String(item).trim()).filter(Boolean);
      }
    } catch (_err) {
      // fall through to comma parsing
    }
  }
  return trimmed
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function buildDropdownSelectUI(node, optionsWidget, selectedWidget) {
  ensureStyles();

  const root = document.createElement("div");
  root.className = "lc-dropdown-select";

  const rowsContainer = document.createElement("div");
  rowsContainer.className = "lc-dropdown-select-rows";

  const addBtn = document.createElement("button");
  addBtn.type = "button";
  addBtn.className = "lc-dropdown-select-add";
  addBtn.textContent = "+ Add Option";

  root.appendChild(rowsContainer);
  root.appendChild(addBtn);

  let items = parseOptions(optionsWidget.value);
  let selected = selectedWidget.value;

  const domWidget = node.addDOMWidget("dropdown_select_ui", "dropdown_select", root, {
    serialize: false,
    hideOnZoom: false,
  });

  // LiteGraph can render the node at a different UI scale than the CSS
  // constants above. Measure the actual DOM so the add button remains inside
  // the node instead of drifting below it as rows are added.
  const computeHeight = () => {
    const measuredHeight = root.scrollHeight;
    const estimatedHeight = items.length * ROW_HEIGHT + CHROME_HEIGHT;
    return Math.max(estimatedHeight, measuredHeight + 16);
  };

  // The DOM widget wrapper can be wider than the canvas node (especially when
  // slots or zoom scaling are involved). Anchor the UI to the actual node
  // width instead of allowing the wrapper's intrinsic width to leak through.
  const syncWidthToNode = () => {
    const nodeWidth = Number(node.size?.[0]);
    const scale = Number(app.canvas?.ds?.scale) || 1;
    if (!Number.isFinite(nodeWidth) || nodeWidth <= 0) {
      return;
    }
    const contentWidth = Math.max(0, nodeWidth * scale - 20);
    root.style.width = contentWidth + "px";
    root.style.maxWidth = contentWidth + "px";
  };

  domWidget.computeSize = function (width) {
    return [width, computeHeight()];
  };

  const resizeNode = () => {
    const computed = node.computeSize();
    const width = Math.max(computed[0], node.size?.[0] ?? MIN_NODE_WIDTH, MIN_NODE_WIDTH);
    node.setSize([width, computed[1]]);
    syncWidthToNode();
    node.setDirtyCanvas(true, true);
  };

  const persist = () => {
    optionsWidget.value = JSON.stringify(items);
    if (!items.includes(selected)) {
      selected = items[0] ?? "";
    }
    selectedWidget.value = selected;
    node.setDirtyCanvas(true, true);
  };

  const updateSelectionHighlight = () => {
    rowsContainer.querySelectorAll(".lc-dropdown-select-row").forEach((row) => {
      row.classList.toggle("lc-selected", row.dataset.value === selected);
    });
  };

  const selectItem = (item) => {
    selected = item;
    persist();
    updateSelectionHighlight();
  };

  const renderRows = () => {
    syncWidthToNode();
    rowsContainer.innerHTML = "";
    items.forEach((item, index) => {
      const row = document.createElement("div");
      row.className = "lc-dropdown-select-row";
      row.dataset.value = item;
      row.addEventListener("pointerdown", (e) => e.stopPropagation());
      row.addEventListener("click", () => selectItem(items[index]));

      const commitRename = (rawValue) => {
        const renamed = rawValue.trim() || item;
        const wasSelected = items[index] === selected;
        items[index] = renamed;
        if (wasSelected) {
          selected = renamed;
        }
        persist();
        renderRows();
      };

      const label = document.createElement("div");
      label.className = "lc-dropdown-select-field lc-dropdown-select-label";
      label.textContent = item;
      label.title = "Double-click to rename";
      label.addEventListener("dblclick", (e) => {
        e.stopPropagation();

        const input = document.createElement("input");
        input.type = "text";
        input.className = "lc-dropdown-select-field";
        input.value = item;
        input.addEventListener("pointerdown", (e2) => e2.stopPropagation());
        input.addEventListener("click", (e2) => e2.stopPropagation());
        input.addEventListener("keydown", (e2) => {
          if (e2.key === "Enter") {
            input.blur();
          } else if (e2.key === "Escape") {
            input.value = item;
            input.blur();
          }
        });
        input.addEventListener("blur", () => commitRename(input.value));

        row.replaceChild(input, label);
        input.focus();
        input.select();
      });

      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "lc-dropdown-select-remove";
      removeBtn.textContent = "×";
      removeBtn.addEventListener("pointerdown", (e) => e.stopPropagation());
      removeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        items.splice(index, 1);
        persist();
        renderRows();
        resizeNode();
      });

      row.appendChild(label);
      row.appendChild(removeBtn);
      rowsContainer.appendChild(row);
    });
    updateSelectionHighlight();
  };

  addBtn.addEventListener("pointerdown", (e) => e.stopPropagation());
  addBtn.addEventListener("click", () => {
    let n = items.length + 1;
    let name = `Option ${n}`;
    while (items.includes(name)) {
      n += 1;
      name = `Option ${n}`;
    }
    items.push(name);
    persist();
    renderRows();
    resizeNode();
  });

  renderRows();
  persist();
  updateSelectionHighlight();

  return {
    syncWidthToNode,
    refreshFromWidgets() {
      items = parseOptions(optionsWidget.value);
      selected = selectedWidget.value;
      renderRows();
      persist();
      resizeNode();
    },
  };
}

app.registerExtension({
  name: "LinkComfyNodes.DropdownSelect",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) {
      return;
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);

      const optionsWidget = findWidget(this, "options");
      const selectedWidget = findWidget(this, "selected");
      if (!optionsWidget || !selectedWidget) {
        return result;
      }

      hideWidgetForGood(optionsWidget);
      hideWidgetForGood(selectedWidget);

      this._dropdownSelectUI = buildDropdownSelectUI(this, optionsWidget, selectedWidget);

      const onResize = this.onResize;
      this.onResize = function () {
        const result = onResize?.apply(this, arguments);
        this._dropdownSelectUI?.syncWidthToNode();
        return result;
      };

      if ((this.size?.[0] ?? 0) < MIN_NODE_WIDTH) {
        this.setSize([MIN_NODE_WIDTH, this.size?.[1] ?? 0]);
      }

      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      this._dropdownSelectUI?.refreshFromWidgets();
      return result;
    };
  },
});
