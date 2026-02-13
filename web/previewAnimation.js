import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
  name: "LinkComfy.PreviewAnimation",

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "PreviewAnimation") return;

    // --- Setup preview widget on node creation ---
    const origCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      origCreated?.apply(this, arguments);

      const node = this;
      const element = document.createElement("div");

      const previewWidget = node.addDOMWidget(
        "videopreview",
        "preview",
        element,
        {
          serialize: false,
          hideOnZoom: false,
          getValue() {
            return element.value;
          },
          setValue(v) {
            element.value = v;
          },
        },
      );

      previewWidget.computeSize = function (width) {
        if (this.aspectRatio && !this.parentEl.hidden) {
          let height = (node.size[0] - 20) / this.aspectRatio + 10;
          if (!(height > 0)) height = 0;
          this.computedHeight = height + 10;
          return [width, height];
        }
        return [width, -4];
      };

      previewWidget.parentEl = document.createElement("div");
      previewWidget.parentEl.style.width = "100%";
      previewWidget.parentEl.hidden = true;
      element.appendChild(previewWidget.parentEl);

      previewWidget.videoEl = document.createElement("video");
      previewWidget.videoEl.controls = false;
      previewWidget.videoEl.loop = true;
      previewWidget.videoEl.muted = true;
      previewWidget.videoEl.autoplay = true;
      previewWidget.videoEl.playsInline = true;
      previewWidget.videoEl.style.width = "100%";

      previewWidget.videoEl.addEventListener("loadedmetadata", () => {
        previewWidget.aspectRatio =
          previewWidget.videoEl.videoWidth / previewWidget.videoEl.videoHeight;
        node.setSize([
          node.size[0],
          node.computeSize([node.size[0], node.size[1]])[1],
        ]);
        node.setDirtyCanvas(true, true);
      });

      previewWidget.videoEl.addEventListener("error", () => {
        console.warn(
          "[PreviewAnimation] Video error for src:",
          previewWidget.videoEl.src,
        );
        previewWidget.parentEl.hidden = true;
        node.setDirtyCanvas(true, true);
      });

      previewWidget.parentEl.appendChild(previewWidget.videoEl);
      node._previewWidget = previewWidget;
    };

    // --- Handle execution result ---
    const origExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      origExecuted?.apply(this, arguments);

      const gifs = message?.gifs;
      if (!gifs?.length) return;

      const gif = gifs[0];
      console.log("[PreviewAnimation] onExecuted gif:", gif);

      const pw = this._previewWidget;
      if (!pw) {
        console.warn("[PreviewAnimation] No preview widget on node");
        return;
      }

      const params = new URLSearchParams({
        filename: gif.filename,
        type: gif.type || "output",
        subfolder: gif.subfolder || "",
        t: Date.now(),
      });
      const url = api.apiURL("/view?" + params.toString());
      console.log("[PreviewAnimation] Setting video src:", url);

      pw.videoEl.src = url;
      pw.videoEl.load();
      pw.videoEl.play().catch(() => {});
      pw.parentEl.hidden = false;
      node.setDirtyCanvas(true, true);

      // context menu extras
      this._previewGif = gif;
      this._previewUrl = url;
    };

    // --- Context menu ---
    const origMenu = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function (_, options) {
      origMenu?.apply(this, arguments);
      if (!this._previewUrl) return;
      const url = this._previewUrl;
      const gif = this._previewGif;
      options.unshift(
        { content: "Open preview", callback: () => window.open(url, "_blank") },
        {
          content: "Save preview",
          callback: () => {
            const a = document.createElement("a");
            a.href = url;
            a.download = gif.filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
          },
        },
        null,
      );
    };
  },
});
