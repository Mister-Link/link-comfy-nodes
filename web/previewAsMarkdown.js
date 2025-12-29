import { app } from "../../scripts/app.js";

// Simple markdown parser for basic syntax
function parseMarkdown(text) {
  if (!text) return "";

  let html = text;

  // Escape HTML first
  html = html
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // Headers
  html = html.replace(/^### (.*$)/gim, "<h3>$1</h3>");
  html = html.replace(/^## (.*$)/gim, "<h2>$1</h2>");
  html = html.replace(/^# (.*$)/gim, "<h1>$1</h1>");

  // Bold
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/__(.+?)__/g, "<strong>$1</strong>");

  // Italic
  html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");
  html = html.replace(/_(.+?)_/g, "<em>$1</em>");

  // Code blocks
  html = html.replace(/```([\s\S]*?)```/g, "<pre><code>$1</code></pre>");

  // Inline code
  html = html.replace(/`(.+?)`/g, "<code>$1</code>");

  // Links
  html = html.replace(
    /\[([^\]]+)\]\(([^)]+)\)/g,
    '<a href="$2" target="_blank">$1</a>',
  );

  // Line breaks
  html = html.replace(/\n\n/g, "</p><p>");
  html = html.replace(/\n/g, "<br>");

  // Wrap in paragraph
  if (!html.startsWith("<h") && !html.startsWith("<pre>")) {
    html = `<p>${html}</p>`;
  }

  return html;
}

app.registerExtension({
  name: "PreviewAsMarkdown.Render",

  async nodeCreated(node) {
    if (node.comfyClass !== "PreviewAsMarkdown") {
      return;
    }

    const PREVIEW_MIN_HEIGHT = 140;
    const PREVIEW_MAX_HEIGHT = 320;
    const NODE_HEIGHT_PADDING = 100; // Increased from 84 to give more margin

    // Create a wrapper and container for the markdown preview
    const wrapper = document.createElement("div");
    wrapper.style.cssText = `
      box-sizing: border-box;
      padding: 8px;
      width: 100%;
      height: 100%;
    `;

    const container = document.createElement("div");
    container.style.cssText = `
      box-sizing: border-box;
      background: rgba(0, 0, 0, 0.2);
      border-radius: 6px;
      padding: 12px 14px;
      color: #e0e0e0;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      font-size: 13px;
      line-height: 1.6;
      height: 100%;
      overflow-y: auto;
      overflow-x: hidden;
      overflow-wrap: break-word;
      word-wrap: break-word;
    `;
    wrapper.appendChild(container);

    // Add style for markdown elements
    const style = document.createElement("style");
    style.textContent = `
      .markdown-preview h1 { font-size: 1.5em; margin: 0.5em 0; font-weight: bold; }
      .markdown-preview h2 { font-size: 1.3em; margin: 0.5em 0; font-weight: bold; }
      .markdown-preview h3 { font-size: 1.1em; margin: 0.5em 0; font-weight: bold; }
      .markdown-preview p { margin: 0.5em 0; }
      .markdown-preview code {
        background: rgba(255, 255, 255, 0.1);
        padding: 2px 4px;
        border-radius: 3px;
        font-family: 'Courier New', monospace;
      }
      .markdown-preview pre {
        background: rgba(255, 255, 255, 0.1);
        padding: 8px;
        border-radius: 4px;
        overflow-x: auto;
      }
      .markdown-preview pre code {
        background: none;
        padding: 0;
      }
      .markdown-preview a {
        color: #4a9eff;
        text-decoration: underline;
      }
      .markdown-preview strong { font-weight: bold; }
      .markdown-preview em { font-style: italic; }
    `;
    document.head.appendChild(style);

    container.className = "markdown-preview";
    container.innerHTML = '<em style="color: #888;">Waiting for input...</em>';

    // Add widget to display the markdown
    const previewWidget = node.addDOMWidget(
      "preview",
      "markdown_preview",
      wrapper,
    );
    previewWidget.serialize = false;
    previewWidget.computeSize = function (width) {
      const availableHeight = (node?.size?.[1] ?? 0) - NODE_HEIGHT_PADDING;
      const height = Math.min(
        PREVIEW_MAX_HEIGHT,
        Math.max(PREVIEW_MIN_HEIGHT, availableHeight),
      );
      return [width, height];
    };

    // Store reference for updates
    node._markdownContainer = container;

    let lastResizeHeight = null;
    const originalOnResize = node.onResize;
    node.onResize = function (size) {
      const minHeight = PREVIEW_MIN_HEIGHT + NODE_HEIGHT_PADDING;
      const maxHeight = PREVIEW_MAX_HEIGHT + NODE_HEIGHT_PADDING;

      // Only apply clamping if the user is manually resizing
      // Prevent feedback loop by checking if height changed from last resize
      if (lastResizeHeight !== size[1]) {
        size[1] = Math.min(maxHeight, Math.max(minHeight, size[1]));
        lastResizeHeight = size[1];
      }

      if (originalOnResize) {
        originalOnResize.apply(this, arguments);
      }
    };

    // Handle execution results
    const originalOnExecuted = node.onExecuted;
    node.onExecuted = function (message) {
      if (originalOnExecuted) {
        originalOnExecuted.apply(this, arguments);
      }

      if (message?.markdown && message.markdown.length > 0) {
        const markdownText = message.markdown[0];
        if (markdownText && markdownText.trim()) {
          const html = parseMarkdown(markdownText);
          node._markdownContainer.innerHTML = html;
        } else {
          node._markdownContainer.innerHTML =
            '<em style="color: #888;">Empty input</em>';
        }
      }
    };
  },
});
