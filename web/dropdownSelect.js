import { app } from "../../scripts/app.js";

const NODE_NAME = "Dropdown Select";
const STYLE_ID = "lc_dropdown_select_styles";
const ROW_HEIGHT = 26;
const CHROME_HEIGHT = 90;
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
      padding: 4px 2px 0;
      display: flex;
      flex-direction: column;
      gap: 6px;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      font-size: 12px;
    }
    .lc-dropdown-select-rows {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .lc-dropdown-select-row {
      display: flex;
      align-items: center;
      gap: 4px;
    }
    .lc-dropdown-select-row input {
      flex: 1;
      min-width: 0;
      box-sizing: border-box;
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 4px;
      color: #e6e6e6;
      font-size: 12px;
      padding: 3px 6px;
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
    .lc-dropdown-select-choice {
      box-sizing: border-box;
      width: 100%;
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid rgba(255, 255, 255, 0.16);
      border-radius: 4px;
      color: #e6e6e6;
      font-size: 12px;
      padding: 4px 6px;
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

  const selectEl = document.createElement("select");
  selectEl.className = "lc-dropdown-select-choice";

  root.appendChild(rowsContainer);
  root.appendChild(addBtn);
  root.appendChild(selectEl);

  let items = parseOptions(optionsWidget.value);
  let selected = selectedWidget.value;

  const domWidget = node.addDOMWidget("dropdown_select_ui", "dropdown_select", root, {
    serialize: false,
    hideOnZoom: false,
  });

  const computeHeight = () => items.length * ROW_HEIGHT + CHROME_HEIGHT;

  domWidget.computeSize = function (width) {
    return [width, computeHeight()];
  };

  const resizeNode = () => {
    const computed = node.computeSize();
    const width = Math.max(computed[0], node.size?.[0] ?? MIN_NODE_WIDTH, MIN_NODE_WIDTH);
    node.setSize([width, computed[1]]);
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

  const renderSelect = () => {
    selectEl.innerHTML = "";
    for (const item of items) {
      const opt = document.createElement("option");
      opt.value = item;
      opt.textContent = item;
      selectEl.appendChild(opt);
    }
    if (!items.includes(selected)) {
      selected = items[0] ?? "";
    }
    selectEl.value = selected;
  };

  const renderRows = () => {
    rowsContainer.innerHTML = "";
    items.forEach((item, index) => {
      const row = document.createElement("div");
      row.className = "lc-dropdown-select-row";

      const input = document.createElement("input");
      input.type = "text";
      input.value = item;
      input.addEventListener("pointerdown", (e) => e.stopPropagation());
      input.addEventListener("change", () => {
        items[index] = input.value.trim() || item;
        persist();
        renderSelect();
      });

      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "lc-dropdown-select-remove";
      removeBtn.textContent = "×";
      removeBtn.addEventListener("pointerdown", (e) => e.stopPropagation());
      removeBtn.addEventListener("click", () => {
        items.splice(index, 1);
        persist();
        renderRows();
        renderSelect();
        resizeNode();
      });

      row.appendChild(input);
      row.appendChild(removeBtn);
      rowsContainer.appendChild(row);
    });
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
    renderSelect();
    resizeNode();
  });

  selectEl.addEventListener("pointerdown", (e) => e.stopPropagation());
  selectEl.addEventListener("change", () => {
    selected = selectEl.value;
    persist();
  });

  renderRows();
  renderSelect();
  persist();

  return {
    refreshFromWidgets() {
      items = parseOptions(optionsWidget.value);
      selected = selectedWidget.value;
      renderRows();
      renderSelect();
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
