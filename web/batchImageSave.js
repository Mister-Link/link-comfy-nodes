import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
  name: "LinkComfy.BatchImageSaveZip",

  async nodeCreated(node) {
    if (node.comfyClass !== "BatchImageSave") {
      return;
    }

    node.downloadUrl = null;
    node.zipFilename = null;

    const downloadButton = node.addWidget("button", "No File", null, () => {
      if (!node.downloadUrl) {
        return;
      }
      const link = document.createElement("a");
      link.href = api.apiURL(node.downloadUrl);
      link.download = node.zipFilename || "batch.zip";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    });

    downloadButton.serialize = false;
    downloadButton.disabled = true;

    const resetDownload = () => {
      node.downloadUrl = null;
      node.zipFilename = null;
      downloadButton.label = "No File";
      downloadButton.tooltip = "";
      downloadButton.disabled = true;
    };

    const originalOnExecuted = node.onExecuted;
    node.onExecuted = function (message) {
      originalOnExecuted?.apply(this, arguments);

      const textHtml = message?.text?.find(
        (text) => typeof text === "string" && text.includes('href="'),
      );
      if (!textHtml) {
        resetDownload();
        return;
      }

      const urlMatch = textHtml.match(/href="([^"]+)"/);
      if (!urlMatch) {
        resetDownload();
        return;
      }

      const filenameMatch = textHtml.match(/Download:\s*([^<]+)/);
      node.downloadUrl = urlMatch[1].replaceAll("&amp;", "&");
      node.zipFilename = filenameMatch
        ? filenameMatch[1].trim()
        : "batch.zip";
      downloadButton.label = "Download ZIP";
      downloadButton.tooltip = `Download ${node.zipFilename}`;
      downloadButton.disabled = false;
    };
  },
});
