import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const STYLE_ID = "lc_preview_animation_styles";

function ensureStyles() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
    .lc-preview-animation-wrapper {
      width: 100%;
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 6px;
      overflow: hidden;
      box-sizing: border-box;
    }

    .lc-preview-animation-video {
      max-width: 100%;
      max-height: 100%;
      display: block;
      object-fit: contain;
    }
  `;
  document.head.appendChild(style);
}

// Extension to handle PreviewAnimation node output
app.registerExtension({
  name: "LinkComfy.PreviewAnimation",

  beforeRegisterNodeDef: async function (nodeType, nodeData, app) {
    if (nodeData.name === "PreviewAnimation") {
      // Hook into onResize to update video sizing when node is resized
      const onResize = nodeType.prototype.onResize;
      nodeType.prototype.onResize = function () {
        if (onResize) {
          onResize.apply(this, arguments);
        }
        // Update wrapper height to fill available space
        if (this.widgets) {
          for (const widget of this.widgets) {
            if (widget.type === "preview_animation" && widget.wrapperElement) {
              // Calculate available height for the preview
              let usedHeight = 26; // header
              for (const w of this.widgets) {
                if (w.type !== "preview_animation") {
                  usedHeight += LiteGraph.NODE_WIDGET_HEIGHT + 4;
                }
              }
              const availableHeight = this.size[1] - usedHeight - 20;
              widget.wrapperElement.style.height = `${Math.max(50, availableHeight)}px`;
            }
          }
        }
        this.setDirtyCanvas(true, true);
      };

      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        if (onExecuted) {
          onExecuted.apply(this, arguments);
        }

        const gifs =
          (message && message.gifs) ||
          (message && message.ui && message.ui.gifs) ||
          [];

        // Clear existing widgets and videos
        if (this.widgets && this.widgets.length) {
          this.widgets = this.widgets.filter((widget) => {
            if (widget.type !== "preview_animation") return true;
            if (widget.element && widget.element.parentNode) {
              widget.element.parentNode.removeChild(widget.element);
            }
            return false;
          });
        }
        if (
          this._previewAnimationVideos &&
          this._previewAnimationVideos.length
        ) {
          this._previewAnimationVideos.forEach((video) => {
            video.pause();
            video.src = "";
            video.remove();
          });
        }
        this._previewAnimationVideos = [];
        this._previewAnimationState = null;

        if (!gifs.length) {
          this.setDirtyCanvas(true, true);
          return;
        }

        ensureStyles();

        const node = this;
        const minNodeHeight = 260;

        for (let i = 0; i < gifs.length; i++) {
          const gif = gifs[i];

          const params = new URLSearchParams({
            filename: gif.filename,
            type: gif.type,
            subfolder: gif.subfolder || "",
          });
          const url = api.apiURL(`/view?${params.toString()}`);

          // Load video completely off-DOM
          const videoElement = document.createElement("video");
          videoElement.loop = true;
          videoElement.muted = true;
          videoElement.playsInline = true;
          videoElement.controls = false;

          if (!node._previewAnimationVideos) {
            node._previewAnimationVideos = [];
          }
          node._previewAnimationVideos.push(videoElement);

          // Only add widget after first frame is ready
          const onFirstFrame = () => {
            videoElement.removeEventListener("canplay", onFirstFrame);

            const videoWidth = videoElement.videoWidth;
            const videoHeight = videoElement.videoHeight;
            const aspectRatio = videoWidth / videoHeight;

            // Store state for resize handling
            node._previewAnimationState = {
              aspectRatio,
              videoWidth,
              videoHeight,
            };

            // Video is ready - now create the widget
            videoElement.className = "lc-preview-animation-video";

            const wrapper = document.createElement("div");
            wrapper.className = "lc-preview-animation-wrapper";
            wrapper.appendChild(videoElement);

            const widget = node.addDOMWidget(
              "preview_" + i,
              "preview_animation",
              wrapper,
              { serialize: false },
            );

            widget.videoElement = videoElement;
            widget.wrapperElement = wrapper;
            widget.aspectRatio = aspectRatio;

            // computeSize tells LiteGraph how much space the widget needs
            // It receives the node width and returns [width, height]
            widget.computeSize = function (nodeWidth) {
              // Return a minimal height - actual sizing handled by wrapper CSS
              return [nodeWidth, 200];
            };

            // Calculate initial node height
            const nodeWidth = node.size && node.size[0] ? node.size[0] : 240;

            // Get space used by header + other widgets
            let usedHeight = 26; // header
            if (node.widgets) {
              for (const w of node.widgets) {
                if (w.type !== "preview_animation") {
                  usedHeight += LiteGraph.NODE_WIDGET_HEIGHT + 4;
                }
              }
            }

            // Calculate preview height to maintain aspect ratio
            const availableWidth = nodeWidth - 20;
            const previewHeight = Math.round(availableWidth / aspectRatio);
            const totalHeight = usedHeight + previewHeight + 20;
            const requiredHeight = Math.max(minNodeHeight, totalHeight);

            node.setSize([nodeWidth, requiredHeight]);
            node.setDirtyCanvas(true, true);

            videoElement.play().catch(() => {});
          };

          videoElement.addEventListener("canplay", onFirstFrame);
          videoElement.src = url;
          videoElement.load();

          // Cleanup handler
          const originalOnRemove = node.onRemoved;
          node.onRemoved = function () {
            if (
              this._previewAnimationVideos &&
              this._previewAnimationVideos.length
            ) {
              this._previewAnimationVideos.forEach((v) => {
                v.pause();
                v.src = "";
                v.remove();
              });
              this._previewAnimationVideos = [];
            }
            this._previewAnimationState = null;
            if (originalOnRemove) {
              originalOnRemove.apply(this, arguments);
            }
          };

          // Context menu
          const originalGetExtraMenuOptions = node.getExtraMenuOptions;
          node.getExtraMenuOptions = function (_, options) {
            if (originalGetExtraMenuOptions) {
              originalGetExtraMenuOptions.apply(this, arguments);
            }
            options.unshift(
              {
                content: "Open preview",
                callback: () => window.open(url, "_blank"),
              },
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
              {
                content: "Copy output filepath",
                callback: () => {
                  const filepath = `${gif.subfolder ? gif.subfolder + "/" : ""}${gif.filename}`;
                  navigator.clipboard.writeText(filepath);
                },
              },
              null,
              {
                content: videoElement.paused ? "Play preview" : "Pause preview",
                callback: () => {
                  if (videoElement.paused) {
                    videoElement.play();
                  } else {
                    videoElement.pause();
                  }
                },
              },
              {
                content: "Mute Preview",
                callback: () => {
                  videoElement.muted = !videoElement.muted;
                },
              },
              null,
            );
          };
        }
      };
    }
  },
});
