import { app } from "../../scripts/app.js";

let prevOpenLasso = false;
let prevVideoPlayMode = false;

app.registerExtension({
  name: "Comfy.MaskEditorOverride",
  init(app) {
    const ComfyDialog = app.ui.dialog.constructor;
    const oldcreateButtons = ComfyDialog.prototype.createButtons;
    const useNewEditor = app.extensionManager.setting.get(
      "Comfy.MaskEditor.UseNewEditor",
    );
    if (!useNewEditor) {
      console.warn("Comfy.MaskEditorOverride: Not using old mask editor");
      return;
    }
    ComfyDialog.prototype.createButtons = function (...args) {
      const res = oldcreateButtons.apply(this, args);
      if (this.constructor.name === "MaskEditorDialog") {
        // eslint-disable-next-line @typescript-eslint/no-this-alias
        const self = this;
        queueMicrotask(() => {
          this.paths = [];
          this.videoFrames = [];
          this.currentFrameIndex = 0;
          this.isPlayingVideo = false;
          this.videoFrameRate = 24;
          this.nodeFps = null; // Will be set by the node if available

          this.messageBroker.createPushTopic("lassoChange");
          this.messageBroker.createPushTopic("videoPlayModeChange");
          this.messageBroker.createPushTopic("setVideoFpsRequest");

          const oldCreateBrushSettings = this.uiManager.createBrushSettings;
          this.uiManager.createBrushSettings = async function (...args) {
            const res = await oldCreateBrushSettings.apply(this, args);

            const toggle = this.createToggle("Lasso", (event, value) => {
              this.messageBroker.publish("lassoChange", value);
            });
            toggle.querySelector('input[type="checkbox"]').checked =
              prevOpenLasso;
            this.messageBroker.publish("lassoChange", prevOpenLasso);
            res.appendChild(toggle);

            const videoToggle = this.createToggle(
              "Video Play Mode",
              (event, value) => {
                this.messageBroker.publish("videoPlayModeChange", value);
              },
            );
            videoToggle.querySelector('input[type="checkbox"]').checked =
              prevVideoPlayMode;
            this.messageBroker.publish(
              "videoPlayModeChange",
              prevVideoPlayMode,
            );
            res.appendChild(videoToggle);

            return res;
          };

          this.messageBroker.subscribe("lassoChange", (open) => {
            self.openLasso = open;
            prevOpenLasso = open;
          });

          this.messageBroker.subscribe("videoPlayModeChange", (enabled) => {
            self.videoPlayMode = enabled;
            prevVideoPlayMode = enabled;
            if (enabled && self.videoFrames.length > 0) {
              self.startVideoPlayback();
            } else {
              self.stopVideoPlayback();
            }
          });

          this.messageBroker.subscribe("setVideoFpsRequest", (fps) => {
            if (self.setVideoFps) {
              self.setVideoFps(fps);
            }
          });

          const oldHandlePointerDown = this.toolManager.handlePointerDown;
          this.toolManager.handlePointerDown = function (...args) {
            const res = oldHandlePointerDown.apply(this, args);
            self.paths = [];
            return res;
          };

          const oldHandlePointerUp = this.toolManager.handlePointerUp;

          this.toolManager.handlePointerUp = async function (...args) {
            const res = oldHandlePointerUp.apply(this, args);
            if (self.paths.length === 0 || !self.openLasso) {
              return res;
            }
            const maskColor = await this.messageBroker.pull("getMaskColor");
            const maskCtx =
              this.maskCtx || (await this.messageBroker.pull("maskCtx"));
            maskCtx.beginPath();
            maskCtx.moveTo(self.paths[0].x, self.paths[0].y);
            const lastPoint = self.paths[self.paths.length - 1];
            for (const path of self.paths) {
              maskCtx.lineTo(path.x, path.y);
            }
            maskCtx.closePath();
            maskCtx.fillStyle = `rgb(${maskColor.r}, ${maskColor.g}, ${maskColor.b})`;
            maskCtx.fill();
            self.brushTool.drawLine(lastPoint, self.paths[0], "source-over");
            return res;
          };

          const oldDraw_shap = this.brushTool.draw_shape;

          this.brushTool.draw_shape = async function (...args) {
            const point = args[0];
            const maskCtx =
              this.maskCtx || (await this.messageBroker.pull("maskCtx"));
            const isErasing =
              maskCtx.globalCompositeOperation === "destination-out";
            if (!isErasing && self.openLasso) {
              self.paths.push(point);
            }
            const res = await oldDraw_shap.apply(this, args);
            return res;
          };

          // Video playback functionality
          self.startVideoPlayback = function () {
            if (self.videoPlaybackInterval) {
              clearInterval(self.videoPlaybackInterval);
            }
            self.isPlayingVideo = true;
            self.currentFrameIndex = 0;

            const frameInterval = 1000 / self.videoFrameRate;
            self.videoPlaybackInterval = setInterval(() => {
              if (self.videoFrames.length === 0) {
                self.stopVideoPlayback();
                return;
              }

              self.displayVideoFrame(self.currentFrameIndex);
              self.currentFrameIndex =
                (self.currentFrameIndex + 1) % self.videoFrames.length;
            }, frameInterval);
          };

          self.stopVideoPlayback = function () {
            self.isPlayingVideo = false;
            if (self.videoPlaybackInterval) {
              clearInterval(self.videoPlaybackInterval);
              self.videoPlaybackInterval = null;
            }
          };

          self.displayVideoFrame = function (frameIndex) {
            if (!self.videoFrames || frameIndex >= self.videoFrames.length)
              return;

            const canvas =
              self.canvas ||
              document.querySelector('canvas[data-type="image-canvas"]');
            if (!canvas) return;

            const ctx = canvas.getContext("2d");
            const frame = self.videoFrames[frameIndex];

            // Draw frame to canvas
            ctx.drawImage(frame, 0, 0);
          };

          self.setVideoFrames = function (frames) {
            self.videoFrames = frames;
          };

          self.setVideoFps = function (fps) {
            if (fps && fps > 0) {
              self.nodeFps = fps;
              self.videoFrameRate = fps;
              // If video is currently playing, restart with new fps
              if (self.isPlayingVideo && self.videoPlayMode) {
                self.stopVideoPlayback();
                self.startVideoPlayback();
              }
            }
          };
        });
      }
      return res;
    };
  },
});
