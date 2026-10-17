import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "Load Folder by Type";
const DIRECTORY_ROUTE = "/link_comfy/load_folder_filtered_directories";
const PREVIEW_ROUTE = "/link_comfy/load_folder_preview";
// Keep the compact minimum small, while allowing the preview to grow into
// any additional height created by the user resizing the node.
const PREVIEW_HEIGHT = 160;
const PREVIEW_FPS = 12;
const POLL_MS = 3000;
const NODE_WIDTH = 270;
const FIELD_HEIGHT = 22;
const FIELD_GAP = 3;

function getWidget(node, name) {
  return node.widgets?.find((widget) => widget.name === name);
}

function hideWidget(widget) {
  if (!widget) return;
  widget.hidden = true;
  widget.computeSize = () => [0, -4];
}

function imageUrl(frame) {
  const params = new URLSearchParams({
    filename: frame.filename ?? "",
    subfolder: frame.subfolder ?? "",
    type: frame.type ?? "input",
  });
  return api.apiURL("/view?" + params.toString());
}

function setSelectOptions(select, values, preferred) {
  const current = String(preferred ?? select.value ?? "");
  select.replaceChildren();
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  }
  select.value = values.includes(current) ? current : (values[0] || "");
  return select.value;
}

function categoryPrefix(name) {
  let value = String(name ?? "");
  if (value.startsWith("[") && value.includes("]")) {
    value = value.slice(value.indexOf("]") + 1);
  }
  return value.split("_", 1)[0].toLowerCase();
}

function filterDirectories(directories, prefix) {
  if (!prefix || prefix === "all") return directories;
  return directories.filter((directory) => categoryPrefix(directory) === prefix);
}

app.registerExtension({
  name: "LinkComfyNodes.LoadFolderFiltered",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      originalOnNodeCreated?.apply(this, arguments);

      const directoryWidget = getWidget(this, "directory");
      const prefixWidget = getWidget(this, "directory_prefix");
      const state = this._lcFilteredFolders = {
        prefix: String(prefixWidget?.value || "all"),
        directories: [],
        frames: [],
        frameSignature: "",
        image: null,
        imageIndex: 0,
        imageCache: new Map(),
        animationTimer: null,
        directoryRequestKey: "",
        directoryLoading: false,
        directoryPayload: { directories: [], prefixes: [] },
        previewRequestKey: "",
        previewRequestToken: 0,
        pollTimer: null,
        removed: false,
      };

      hideWidget(prefixWidget);
      hideWidget(directoryWidget);
      for (const name of ["image_load_cap", "skip_first_images", "select_every_nth"]) {
        hideWidget(getWidget(this, name));
      }

      const selector = document.createElement("div");
      selector.style.cssText = [
        "display:flex",
        "flex-direction:column",
        `gap:${FIELD_GAP}px`,
        "padding:0 4px 4px",
        "box-sizing:border-box",
        "width:100%",
        "font:14px Arial,sans-serif",
      ].join(";");

      const makeField = (labelText) => {
        const field = document.createElement("div");
        field.style.cssText = [
          "display:flex",
          "align-items:center",
          "width:100%",
          `height:${FIELD_HEIGHT}px`,
          "box-sizing:border-box",
          "padding:0 7px 0 9px",
          "background:#3b3b3b",
          "border:1px solid #686868",
          "border-radius:11px",
          "overflow:hidden",
          "font:14px Arial,sans-serif",
        ].join(";");

        const label = document.createElement("span");
        label.textContent = labelText;
        label.style.cssText = [
          "flex:0 0 auto",
          "margin-right:8px",
          "color:#999999",
          "line-height:20px",
          "white-space:nowrap",
        ].join(";");
        field.appendChild(label);
        return { field, label };
      };

      const makeSelect = (labelText) => {
        const { field } = makeField(labelText);
        const select = document.createElement("select");
        select.style.cssText = [
          "flex:1",
          "min-width:0",
          "height:20px",
          "padding:0",
          "border:0",
          "background:transparent",
          "color:#dddddd",
          "font:14px Arial,sans-serif",
          "appearance:none",
          "-webkit-appearance:none",
          "text-overflow:ellipsis",
          "outline:none",
          "color-scheme:dark",
        ].join(";");

        const arrow = document.createElement("span");
        arrow.textContent = "▾";
        arrow.style.cssText = "color:#999999;flex:0 0 auto;padding-left:6px;pointer-events:none";

        field.append(select, arrow);
        field.addEventListener("click", (event) => {
          if (event.target === select) return;
          select.focus();
          select.click();
        });
        return { field, select };
      };

      const makeNumberField = (labelText, widget) => {
        const { field } = makeField(labelText);
        const input = document.createElement("input");
        input.type = "number";
        input.value = String(widget?.value ?? 0);
        input.min = String(widget?.options?.min ?? 0);
        input.max = String(widget?.options?.max ?? 0);
        input.step = String(widget?.options?.step ?? 1);
        input.style.cssText = [
          "flex:1",
          "min-width:0",
          "width:0",
          "height:20px",
          "padding:0",
          "border:0",
          "outline:none",
          "background:transparent",
          "color:#dddddd",
          "font:14px Arial,sans-serif",
          "text-align:right",
          "appearance:textfield",
          "-moz-appearance:textfield",
        ].join(";");
        input.addEventListener("change", () => {
          let value = Number(input.value);
          if (!Number.isFinite(value)) value = Number(widget?.value ?? 0);
          if (widget?.options?.min != null) value = Math.max(Number(widget.options.min), value);
          if (widget?.options?.max != null) value = Math.min(Number(widget.options.max), value);
          input.value = String(value);
          commitNativeValue(widget, value);
        });
        field.appendChild(input);
        return input;
      };

      const typeControl = makeSelect("type");
      const directoryControl = makeSelect("folder");
      const typeSelect = typeControl.select;
      const directorySelect = directoryControl.select;
      const numberInputs = new Map();
      selector.append(typeControl.field, directoryControl.field);
      for (const name of ["image_load_cap", "skip_first_images", "select_every_nth"]) {
        numberInputs.set(name, makeNumberField(
          name === "image_load_cap" ? "image_load_cap" :
            name === "skip_first_images" ? "skip_first_images" : "select_every_nth",
          getWidget(this, name),
        ));
        selector.append(numberInputs.get(name).parentElement);
      }

      const preview = document.createElement("div");
      preview.style.cssText = [
        `height:${PREVIEW_HEIGHT}px`,
        "width:100%",
        "display:flex",
        "align-items:center",
        "justify-content:center",
        "box-sizing:border-box",
        "overflow:hidden",
        "background:transparent",
      ].join(";");
      const previewImage = document.createElement("img");
      previewImage.alt = "";
      previewImage.style.cssText = [
        "display:block",
        "max-width:100%",
        "max-height:100%",
        "width:auto",
        "height:auto",
        "object-fit:contain",
        "image-rendering:auto",
      ].join(";");
      preview.appendChild(previewImage);
      selector.appendChild(preview);

      const selectorWidget = this.addDOMWidget(
        "folder_selector",
        "customSelect",
        selector,
        { serialize: false, hideOnZoom: false },
      );
      const controlsHeight = (FIELD_HEIGHT * 5) + (FIELD_GAP * 5) + 4;
      const minimumSelectorHeight = controlsHeight + PREVIEW_HEIGHT;
      const updatePreviewLayout = () => {
        const widgetTop = Number(selectorWidget.y) || 0;
        const nodeHeight = Number(this.size?.[1]) || minimumSelectorHeight;
        const available = nodeHeight > widgetTop
          ? nodeHeight - widgetTop - 4
          : minimumSelectorHeight;
        const selectorHeight = Math.max(minimumSelectorHeight, available);
        preview.style.height = `${Math.max(PREVIEW_HEIGHT, selectorHeight - controlsHeight)}px`;
        selector.style.height = `${selectorHeight}px`;
        return selectorHeight;
      };
      // Keep the resize handle's minimum fixed. The preview can grow into
      // extra node height, but must not turn that current height into a new
      // minimum that prevents the user from dragging the node smaller.
      selectorWidget.computeSize = (width) => [width, minimumSelectorHeight];

      // DOM widgets are appended by addDOMWidget; move this one above the
      // hidden execution widgets so the visual order is type, folder, options.
      const selectorIndex = this.widgets.indexOf(selectorWidget);
      if (selectorIndex >= 0) {
        this.widgets.splice(selectorIndex, 1);
        this.widgets.splice(0, 0, selectorWidget);
      }

      const originalComputeSize = this.computeSize.bind(this);
      this.computeSize = (...args) => {
        const size = originalComputeSize(...args);
        size[0] = Math.max(size[0], NODE_WIDTH);
        size[1] = Math.max(size[1], minimumSelectorHeight);
        return size;
      };

      this.setSize([NODE_WIDTH, this.computeSize()[1]]);
      updatePreviewLayout();

      const originalOnResize = this.onResize;
      this.onResize = function () {
        originalOnResize?.apply(this, arguments);
        updatePreviewLayout();
        this.setDirtyCanvas(true, true);
      };

      const initialTypes = prefixWidget?.options?.values || ["all"];
      setSelectOptions(typeSelect, initialTypes, state.prefix);
      setSelectOptions(directorySelect,
        filterDirectories(directoryWidget?.options?.values || [], state.prefix),
        directoryWidget?.value);

      const commitNativeValue = (widget, value) => {
        if (!widget) return;
        widget.value = value;
        widget.callback?.(value);
        this.setDirtyCanvas(true, true);
      };

      const stopAnimation = () => {
        if (state.animationTimer) clearInterval(state.animationTimer);
        state.animationTimer = null;
      };

      const loadFrame = (index) => {
        if (!state.frames.length) {
          state.image = null;
          this.setDirtyCanvas(true, true);
          return;
        }
        state.imageIndex = ((index % state.frames.length) + state.frames.length) % state.frames.length;
        const src = imageUrl(state.frames[state.imageIndex]);
        previewImage.src = src;
        previewImage.style.visibility = "visible";
        const cached = state.imageCache.get(src);
        if (cached?.complete) {
          state.image = cached;
          this.setDirtyCanvas(true, true);
          return;
        }
        const image = cached || new Image();
        if (!cached) state.imageCache.set(src, image);
        image.onload = () => {
          state.image = image;
          this.setDirtyCanvas(true, true);
        };
        image.onerror = () => {
          state.image = null;
          previewImage.style.visibility = "hidden";
          this.setDirtyCanvas(true, true);
        };
        if (!cached) image.src = src;
      };

      const renderFrames = (frames) => {
        const nextFrames = Array.isArray(frames) ? frames : [];
        const signature = nextFrames.map((frame) => imageUrl(frame)).join("|");
        if (signature === state.frameSignature) return;

        stopAnimation();
        state.frameSignature = signature;
        state.frames = nextFrames;
        state.imageIndex = 0;
        state.image = null;
        state.imageCache.clear();
        if (!state.frames.length) {
          previewImage.removeAttribute("src");
          previewImage.style.visibility = "hidden";
          this.setDirtyCanvas(true, true);
          return;
        }

        loadFrame(0);
        if (state.frames.length > 1) {
          state.animationTimer = setInterval(
            () => loadFrame(state.imageIndex + 1),
            1000 / PREVIEW_FPS,
          );
        }
        this.setDirtyCanvas(true, true);
      };

      const refreshPreview = async (force = false) => {
        const directory = String(directoryWidget?.value || "");
        const cap = Number(getWidget(this, "image_load_cap")?.value || 0) || 0;
        const skip = Number(getWidget(this, "skip_first_images")?.value || 0) || 0;
        const every = Number(getWidget(this, "select_every_nth")?.value || 1) || 1;
        const key = [directory, cap, skip, every].join("|");
        if (!force && key === state.previewRequestKey) return;
        state.previewRequestKey = key;
        const requestToken = ++state.previewRequestToken;
        if (!directory) {
          renderFrames([]);
          return;
        }
        try {
          const params = new URLSearchParams({
            directory,
            image_load_cap: String(cap),
            skip_first_images: String(skip),
            select_every_nth: String(every),
          });
          const response = await fetch(api.apiURL(PREVIEW_ROUTE + "?" + params), {
            cache: "no-store",
          });
          if (!response.ok) throw new Error("Preview request failed");
          const payload = await response.json();
          if (requestToken === state.previewRequestToken) {
            renderFrames(payload.frames ?? []);
          }
        } catch (_error) {
          // Preserve the last valid preview across a transient poll/fetch
          // failure, then allow the next background poll to retry it.
          if (requestToken === state.previewRequestToken) {
            state.previewRequestKey = "";
          }
        }
        this.setDirtyCanvas(true, true);
      };

      const applyDirectoryPayload = (payload, chooseNewest = false) => {
        state.directoryPayload = payload || { directories: [], prefixes: [] };
        const prefixes = Array.isArray(payload?.prefixes) ? payload.prefixes : [];
        const availablePrefixes = ["all", ...prefixes.filter((prefix) => prefix !== "all")];
        state.prefix = setSelectOptions(typeSelect, availablePrefixes, state.prefix);
        commitNativeValue(prefixWidget, state.prefix);

        const allDirectories = Array.isArray(payload?.directories) ? payload.directories : [];
        const directories = filterDirectories(allDirectories, state.prefix);
        state.directories = directories;
        const preferredDirectory = chooseNewest ? directories[0] : directoryWidget?.value;
        const directory = setSelectOptions(directorySelect, directories, preferredDirectory);
        commitNativeValue(directoryWidget, directory);
        refreshPreview();
        this.setDirtyCanvas(true, true);
      };

      const refreshDirectories = async (force = false, chooseNewest = false) => {
        if (state.directoryLoading && !force) return;
        state.directoryLoading = true;
        try {
          // Always fetch the complete list. The custom control performs the
          // type filtering locally, so its folder select updates immediately
          // without relying on ComfyUI's static combo widget.
          const params = new URLSearchParams({ prefix: "all" });
          const response = await fetch(api.apiURL(DIRECTORY_ROUTE + "?" + params), {
            cache: "no-store",
          });
          if (!response.ok) throw new Error("Folder list request failed");
          applyDirectoryPayload(await response.json(), chooseNewest);
        } catch (_error) {
          // Keep the current controls and preview when a background poll fails.
        } finally {
          state.directoryLoading = false;
        }
      };

      typeSelect.addEventListener("change", () => {
        state.prefix = typeSelect.value || "all";
        if (state.directoryPayload?.directories?.length) {
          applyDirectoryPayload(state.directoryPayload, true);
        } else {
          refreshDirectories(true, true);
        }
      });

      directorySelect.addEventListener("change", () => {
        commitNativeValue(directoryWidget, directorySelect.value);
        state.previewRequestKey = "";
        refreshPreview(true);
      });

      for (const name of ["image_load_cap", "skip_first_images", "select_every_nth"]) {
        const widget = getWidget(this, name);
        if (!widget) continue;
        const originalCallback = widget.callback;
        widget.callback = (...args) => {
          const result = originalCallback?.apply(widget, args);
          state.previewRequestKey = "";
          refreshPreview(true);
          return result;
        };
      }

      const originalOnExecuted = this.onExecuted;
      this.onExecuted = function (message) {
        originalOnExecuted?.apply(this, arguments);
        if (message?.fast_images) renderFrames(message.fast_images);
      };

      const originalOnConfigure = this.onConfigure;
      this.onConfigure = function () {
        originalOnConfigure?.apply(this, arguments);
        state.prefix = String(prefixWidget?.value || "all");
        setSelectOptions(typeSelect, typeSelect.options.length
          ? [...typeSelect.options].map((option) => option.value)
          : ["all"], state.prefix);
        setSelectOptions(directorySelect,
          filterDirectories(directoryWidget?.options?.values || [], state.prefix),
          directoryWidget?.value);
        for (const [name, input] of numberInputs) {
          const widget = getWidget(this, name);
          if (widget) input.value = String(widget.value ?? 0);
        }
        setTimeout(() => {
          refreshDirectories(true);
          requestAnimationFrame(() => {
            if (state.removed) return;
            updatePreviewLayout();
            this.setSize([NODE_WIDTH, Math.max(this.size?.[1] ?? 0, this.computeSize()[1])]);
            this.setDirtyCanvas(true, true);
          });
        }, 0);
      };

      const originalOnRemoved = this.onRemoved;
      this.onRemoved = function () {
        state.removed = true;
        stopAnimation();
        if (state.pollTimer) clearInterval(state.pollTimer);
        originalOnRemoved?.apply(this, arguments);
      };

      state.pollTimer = setInterval(() => {
        if (state.removed) return;
        refreshDirectories(true);
      }, POLL_MS);
      setTimeout(() => refreshDirectories(true), 0);
    };
  },
});
