import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
  name: "SaveImageSequenceZip.DownloadButton",

  async nodeCreated(node) {
    if (node.comfyClass !== "SaveImageSequenceZip") {
      return;
    }

    // Store the latest download URL
    node.downloadUrl = null;
    node.zipFilename = null;

    // Add download button widget
    const downloadButton = node.addWidget(
      "button",
      "⬇ Download ZIP (run workflow first)",
      null,
      () => {
        if (node.downloadUrl) {
          // Create a temporary link and click it to trigger download
          const link = document.createElement("a");
          link.href = api.apiURL(node.downloadUrl);
          link.download = node.zipFilename || "sequence.zip";
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
        } else {
          alert("No ZIP file available. Please run the workflow first.");
        }
      },
    );

    // Don't serialize this widget
    downloadButton.serialize = false;

    // Hook into the onExecuted callback to capture the download URL
    const originalOnExecuted = node.onExecuted;
    node.onExecuted = function (message) {
      if (originalOnExecuted) {
        originalOnExecuted.apply(this, arguments);
      }

      // Parse the text output to extract download URL
      if (message?.text && message.text.length > 0) {
        const textHtml = message.text[0];

        // Extract URL from the HTML link
        const urlMatch = textHtml.match(/href="([^"]+)"/);
        const filenameMatch = textHtml.match(/Download:\s*([^<]+)/);

        if (urlMatch) {
          node.downloadUrl = urlMatch[1];
          node.zipFilename = filenameMatch
            ? filenameMatch[1].trim()
            : "sequence.zip";

          // Update the download button label
          downloadButton.label = `⬇ Download ${node.zipFilename}`;

          console.log(
            `[SaveImageSequenceZip] ZIP ready for download: ${node.zipFilename}`,
          );
        }
      }
    };
  },
});
