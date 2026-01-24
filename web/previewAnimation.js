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
    }

    .lc-preview-animation-video {
      width: 100%;
      height: auto;
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

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === "PreviewAnimation") {
      // Add callback for when the node is executed
      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        if (onExecuted) {
          onExecuted.apply(this, arguments);
        }

        const gifs = message?.gifs ?? message?.ui?.gifs ?? [];
        const clearExisting = () => {
          if (this.widgets?.length) {
            this.widgets = this.widgets.filter((widget) => {
              if (widget.type !== "preview_animation") return true;
              if (widget.element?.parentNode) {
                widget.element.parentNode.removeChild(widget.element);
              }
              return false;
            });
          }

          if (this._previewAnimationVideos?.length) {
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
            widget.computeSize = function (width) {
              if (this.videoElement?.videoWidth > 0) {
                const aspectRatio =
                  this.videoElement.videoHeight / this.videoElement.videoWidth;
                const previewWidth = width - 20;
                const previewHeight = previewWidth * aspectRatio;
                return [width, previewHeight + 20];
              }
              return [width, 220];
            };
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
            videoElement.addEventListener("loadeddata", () => {
              videoElement
                .play()
                .catch((e) => console.log("Auto-play failed:", e));
              const newSize = widget.computeSize(this.size?.[0] ?? 240);
              if (this.size?.[1] < newSize[1]) {
                this.setSize([this.size?.[0] ?? 240, newSize[1]]);
              }
              this.setDirtyCanvas(true, true);
            });

            // Clean up on widget removal
            const originalOnRemove = this.onRemoved;
            this.onRemoved = function () {
              if (this._previewAnimationVideos?.length) {
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

          const minNodeHeight = 260;
          if ((this.size?.[1] ?? 0) < minNodeHeight) {
            this.setSize([this.size?.[0] ?? 240, minNodeHeight]);
          }
          this.setDirtyCanvas(true, true);
        }
      };
    }
  },
});
