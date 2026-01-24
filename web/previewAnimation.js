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
      display: flex;
      align-items: center;
      justify-content: center;
      background: #000000;
      border-radius: 6px;
      overflow: hidden;
      box-sizing: border-box;
    }

    .lc-preview-animation-video {
      width: 100%;
      height: 100%;
      display: block;
      object-fit: contain;
      background: #000000;
    }
  `;
  document.head.appendChild(style);
}

// Extension to handle PreviewAnimation node output
app.registerExtension({
  name: "LinkComfy.PreviewAnimation",

  beforeRegisterNodeDef: async function (nodeType, nodeData, app) {
    if (nodeData.name === "PreviewAnimation") {
      // Add callback for when the node is executed
      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        if (onExecuted) {
          onExecuted.apply(this, arguments);
        }

        const gifs =
          (message && message.gifs) ||
          (message && message.ui && message.ui.gifs) ||
          [];
        const clearExisting = () => {
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
        };

        if (!gifs.length) {
          clearExisting();
          this.setDirtyCanvas(true, true);
          return;
        }

        ensureStyles();
        clearExisting();

        const minNodeHeight = 260;

        // Create preview widget for each gif/video
        for (let i = 0; i < gifs.length; i++) {
          const gif = gifs[i];
          const wrapper = document.createElement("div");
          wrapper.className = "lc-preview-animation-wrapper";

          const videoElement = document.createElement("video");
          videoElement.className = "lc-preview-animation-video";
          videoElement.autoplay = true;
          videoElement.loop = true;
          videoElement.muted = true;
          videoElement.playsInline = true;
          videoElement.controls = false;

          wrapper.appendChild(videoElement);

          const widget = this.addDOMWidget(
            "preview_" + i,
            "preview_animation",
            wrapper,
            { serialize: false },
          );

          widget.videoElement = videoElement;
          widget.wrapperElement = wrapper;
          widget.computeSize = function (width) {
            if (
              this.videoElement &&
              this.videoElement.videoWidth > 0 &&
              this.videoElement.videoHeight > 0
            ) {
              const aspectRatio =
                this.videoElement.videoHeight / this.videoElement.videoWidth;
              const previewWidth = Math.max(1, width - 20);
              const previewHeight = Math.round(previewWidth * aspectRatio);
              if (this.wrapperElement) {
                this.wrapperElement.style.height = `${previewHeight}px`;
              }
              return [width, previewHeight + 20];
            }
            return [width, 220];
          };
          if (!this._previewAnimationVideos) {
            this._previewAnimationVideos = [];
          }
          this._previewAnimationVideos.push(videoElement);

          // Build the URL for the preview
          const params = new URLSearchParams({
            filename: gif.filename,
            type: gif.type,
            subfolder: gif.subfolder || "",
          });

          const url = api.apiURL(`/view?${params.toString()}`);

          // Load the video
          videoElement.src = url;
          videoElement.load();

          // Play when loaded
          const updateSize = (forceResize = false) => {
            const nodeWidth = this.size && this.size[0] ? this.size[0] : 240;
            const nodeHeight = this.size && this.size[1] ? this.size[1] : 0;
            const newSize = widget.computeSize(nodeWidth);
            const requiredHeight = Math.max(minNodeHeight, newSize[1]);
            // Resize if height is insufficient OR if video dimensions are now available (forceResize)
            if (nodeHeight < requiredHeight || forceResize) {
              this.setSize([nodeWidth, requiredHeight]);
            }
            this.setDirtyCanvas(true, true);
          };

          const scheduleSizeUpdate = (attempts = 0) => {
            if (videoElement.videoWidth > 0 && videoElement.videoHeight > 0) {
              // Force resize when video dimensions are available
              updateSize(true);
              return;
            }
            if (attempts < 60) {
              requestAnimationFrame(() => scheduleSizeUpdate(attempts + 1));
            }
          };

          videoElement.addEventListener("loadedmetadata", scheduleSizeUpdate);
          videoElement.addEventListener("resize", scheduleSizeUpdate);
          videoElement.addEventListener("canplay", scheduleSizeUpdate);
          videoElement.addEventListener("loadeddata", () => {
            videoElement
              .play()
              .catch((e) => console.log("Auto-play failed:", e));
            scheduleSizeUpdate();
          });

          // Clean up on widget removal
          const originalOnRemove = this.onRemoved;
          this.onRemoved = function () {
            if (
              this._previewAnimationVideos &&
              this._previewAnimationVideos.length
            ) {
              this._previewAnimationVideos.forEach((video) => {
                video.pause();
                video.src = "";
                video.remove();
              });
              this._previewAnimationVideos = [];
            }
            if (originalOnRemove) {
              originalOnRemove.apply(this, arguments);
            }
          };

          // Add context menu options
          const originalGetExtraMenuOptions = this.getExtraMenuOptions;
          this.getExtraMenuOptions = function (_, options) {
            if (originalGetExtraMenuOptions) {
              originalGetExtraMenuOptions.apply(this, arguments);
            }

            // Add video-specific options
            options.unshift(
              {
                content: "Open preview",
                callback: () => {
                  window.open(url, "_blank");
                },
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
                  navigator.clipboard.writeText(filepath).then(() => {
                    console.log("Copied to clipboard:", filepath);
                  });
                },
              },
              null, // separator
              {
                content: widget.videoElement.paused
                  ? "Play preview"
                  : "Pause preview",
                callback: () => {
                  if (widget.videoElement.paused) {
                    widget.videoElement.play();
                  } else {
                    widget.videoElement.pause();
                  }
                },
              },
              {
                content: "Mute Preview",
                callback: () => {
                  widget.videoElement.muted = !widget.videoElement.muted;
                },
              },
              null, // separator
            );
          };
        }

        const nodeWidth = this.size && this.size[0] ? this.size[0] : 240;
        const nodeHeight = this.size && this.size[1] ? this.size[1] : 0;
        if (nodeHeight < minNodeHeight) {
          this.setSize([nodeWidth, minNodeHeight]);
        }
        this.setDirtyCanvas(true, true);
      };
    }
  },
});
