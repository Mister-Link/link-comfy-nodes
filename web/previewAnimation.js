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

          const existingMedia =
            this._previewAnimationElements ||
            this._previewAnimationVideos ||
            [];
          if (existingMedia.length) {
            existingMedia.forEach((element) => {
              if (element.tagName === "VIDEO") {
                element.pause();
                element.src = "";
              }
              element.remove();
            });
          }
          this._previewAnimationElements = [];
          this._previewAnimationVideos = [];
        };

        if (!gifs.length) {
          clearExisting();
          this.setDirtyCanvas(true, true);
          return;
        }

        ensureStyles();
        clearExisting();

        // Create preview widget for each gif/video
        for (let i = 0; i < gifs.length; i++) {
          const gif = gifs[i];
          const wrapper = document.createElement("div");
          wrapper.className = "lc-preview-animation-wrapper";

          const isAnimatedImage = /\.(gif|webp|apng)$/i.test(
            gif.filename || "",
          );
          const mediaElement = document.createElement(
            isAnimatedImage ? "img" : "video",
          );
          mediaElement.className = "lc-preview-animation-video";
          if (!isAnimatedImage) {
            mediaElement.autoplay = true;
            mediaElement.loop = true;
            mediaElement.muted = true;
            mediaElement.playsInline = true;
            mediaElement.controls = false;
          }

          wrapper.appendChild(mediaElement);

          const widget = this.addDOMWidget(
            "preview_" + i,
            "preview_animation",
            wrapper,
            { serialize: false },
          );

          widget.mediaElement = mediaElement;
          widget.wrapperElement = wrapper;
          widget.computeSize = function (width) {
            const videoWidth = this.mediaElement?.videoWidth || 0;
            const videoHeight = this.mediaElement?.videoHeight || 0;
            const naturalWidth = this.mediaElement?.naturalWidth || 0;
            const naturalHeight = this.mediaElement?.naturalHeight || 0;
            const mediaWidth = videoWidth || naturalWidth;
            const mediaHeight = videoHeight || naturalHeight;
            if (mediaWidth > 0 && mediaHeight > 0) {
              const aspectRatio = mediaHeight / mediaWidth;
              const previewWidth = Math.max(1, width - 20);
              const previewHeight = Math.round(previewWidth * aspectRatio);
              if (this.wrapperElement) {
                this.wrapperElement.style.height = `${previewHeight}px`;
              }
              return [width, previewHeight + 20];
            }
            return [width, 220];
          };
          if (!this._previewAnimationElements) {
            this._previewAnimationElements = [];
          }
          this._previewAnimationElements.push(mediaElement);

          // Build the URL for the preview
          const params = new URLSearchParams({
            filename: gif.filename,
            type: gif.type,
            subfolder: gif.subfolder || "",
          });

          const url = api.apiURL(`/view?${params.toString()}`);

          // Load the media
          mediaElement.src = url;
          if (!isAnimatedImage) {
            mediaElement.load();
          }

          // Play when loaded
          const updateSize = () => {
            const nodeWidth = this.size && this.size[0] ? this.size[0] : 240;
            const nodeHeight = this.size && this.size[1] ? this.size[1] : 0;
            const newSize = widget.computeSize(nodeWidth);
            if (nodeHeight < newSize[1]) {
              this.setSize([nodeWidth, newSize[1]]);
            }
            this.setDirtyCanvas(true, true);
          };

          const scheduleSizeUpdate = (attempts = 0) => {
            const videoWidth = mediaElement.videoWidth || 0;
            const videoHeight = mediaElement.videoHeight || 0;
            const naturalWidth = mediaElement.naturalWidth || 0;
            const naturalHeight = mediaElement.naturalHeight || 0;
            if (
              (videoWidth > 0 && videoHeight > 0) ||
              (naturalWidth > 0 && naturalHeight > 0)
            ) {
              updateSize();
              return;
            }
            if (attempts < 10) {
              requestAnimationFrame(() => scheduleSizeUpdate(attempts + 1));
            }
          };

          if (isAnimatedImage) {
            mediaElement.addEventListener("load", scheduleSizeUpdate);
          } else {
            mediaElement.addEventListener("loadedmetadata", scheduleSizeUpdate);
            mediaElement.addEventListener("loadeddata", () => {
              mediaElement
                .play()
                .catch((e) => console.log("Auto-play failed:", e));
              scheduleSizeUpdate();
            });
          }

          // Clean up on widget removal
          const originalOnRemove = this.onRemoved;
          this.onRemoved = function () {
            if (
              this._previewAnimationElements &&
              this._previewAnimationElements.length
            ) {
              this._previewAnimationElements.forEach((element) => {
                if (element.tagName === "VIDEO") {
                  element.pause();
                  element.src = "";
                }
                element.remove();
              });
              this._previewAnimationElements = [];
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
            const isVideo = mediaElement.tagName === "VIDEO";
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
            );
            if (isVideo) {
              options.unshift(
                null, // separator
                {
                  content: mediaElement.paused
                    ? "Play preview"
                    : "Pause preview",
                  callback: () => {
                    if (mediaElement.paused) {
                      mediaElement.play();
                    } else {
                      mediaElement.pause();
                    }
                  },
                },
                {
                  content: "Mute Preview",
                  callback: () => {
                    mediaElement.muted = !mediaElement.muted;
                  },
                },
                null, // separator
              );
            }
          };
        }

        const minNodeHeight = 260;
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
