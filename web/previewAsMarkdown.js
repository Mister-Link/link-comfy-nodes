import { app } from "../../scripts/app.js";

const STYLE_ID = "lc_markdown_preview_styles";
const PREVIEW_MIN_HEIGHT = 80;
const NODE_HEIGHT_PADDING = 54;

function ensureStyles() {
  if (document.getElementById(STYLE_ID)) {
    return;
  }

  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
    .lc-md-preview-wrapper {
      box-sizing: border-box;
      width: 100%;
      height: 100%;
      padding: 6px 0 0 0;
      display: block;
    }
    .lc-md-preview {
      box-sizing: border-box;
      width: 100%;
      height: 100%;
      padding: 12px 12px 14px;
      border-radius: 8px;
      background:
        linear-gradient(180deg, rgba(22, 22, 22, 0.9), rgba(18, 18, 18, 0.85));
      border: 1px solid rgba(255, 255, 255, 0.06);
      box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.25);
      color: #e6e6e6;
      font-family: "IBM Plex Sans", "Space Grotesk", "Segoe UI", sans-serif;
      font-size: 13px;
      line-height: 1.5;
      overflow: auto;
      overflow-wrap: break-word;
      word-break: break-word;
    }
    .lc-md-preview-empty {
      color: #9a9a9a;
      font-style: italic;
      padding: 4px 0;
    }
    .lc-md-preview h1 {
      font-size: 1.4em;
      margin: 0.6em 0 0.35em;
      font-weight: 600;
    }
    .lc-md-preview h2 {
      font-size: 1.2em;
      margin: 0.6em 0 0.35em;
      font-weight: 600;
    }
    .lc-md-preview h3 {
      font-size: 1.05em;
      margin: 0.55em 0 0.3em;
      font-weight: 600;
    }
    .lc-md-preview p {
      margin: 0.4em 0;
    }
    .lc-md-preview code {
      font-family: "JetBrains Mono", "Fira Code", "SFMono-Regular", monospace;
      font-size: 0.95em;
      background: rgba(255, 255, 255, 0.08);
      padding: 1px 4px;
      border-radius: 3px;
    }
    .lc-md-preview pre {
      margin: 0.6em 0;
      padding: 10px 12px;
      border-radius: 6px;
      background: rgba(255, 255, 255, 0.08);
      overflow-x: auto;
    }
    .lc-md-preview pre code {
      background: none;
      padding: 0;
    }
    .lc-md-preview a {
      color: #7bb7ff;
      text-decoration: underline;
    }
    .lc-md-preview strong {
      font-weight: 600;
    }
    .lc-md-preview em {
      font-style: italic;
    }
    .lc-md-preview ul,
    .lc-md-preview ol {
      margin: 0.4em 0 0.4em 1.2em;
      padding: 0;
    }
    .lc-md-preview li {
      margin: 0.2em 0;
    }
  `;

  document.head.appendChild(style);
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderMarkdown(text) {
  if (!text) {
    return "";
  }

  let html = escapeHtml(text);

  const codeBlocks = [];
  html = html.replace(/```([\s\S]*?)```/g, (_, code) => {
    codeBlocks.push(code);
    return `@@CODEBLOCK_${codeBlocks.length - 1}@@`;
  });

  const inlineCode = [];
  html = html.replace(/`([^`]+)`/g, (_, code) => {
    inlineCode.push(code);
    return `@@INLINECODE_${inlineCode.length - 1}@@`;
  });

  // Extract links IMMEDIATELY after code, BEFORE emphasis processing
  const links = [];
  html = html.replace(/\[([^\]]+)\]\s*\(([^)]+)\)/g, (_, text, url) => {
    links.push({ text, url });
    return `@@LINK_${links.length - 1}@@`;
  });

  html = html.replace(/^### (.*$)/gim, "<h3>$1</h3>");
  html = html.replace(/^## (.*$)/gim, "<h2>$1</h2>");
  html = html.replace(/^# (.*$)/gim, "<h1>$1</h1>");

  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(
    /(^|[^\w])__(.+?)__([^\w]|$)/g,
    (_, lead, text, tail) => `${lead}<strong>${text}</strong>${tail}`,
  );
  html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");
  html = html.replace(
    /(^|[^\w])_(.+?)_([^\w]|$)/g,
    (_, lead, text, tail) => `${lead}<em>${text}</em>${tail}`,
  );

  // Restore links after emphasis processing
  html = html.replace(/@@LINK_(\d+)@@/g, (_, index) => {
    const link = links[Number(index)];
    return `<a href="${link.url}" target="_blank" rel="noopener noreferrer">${link.text}</a>`;
  });

  html = html.replace(/^(\s*[-*] .+(?:\n\s*[-*] .+)*)/gm, (list) => {
    const items = list
      .trim()
      .split("\n")
      .map((line) => line.replace(/^\s*[-*]\s+/, "").trim())
      .map((item) => `<li>${item}</li>`)
      .join("");
    return `<ul>${items}</ul>`;
  });

  html = html.replace(/^(\s*\d+\.\s+.+(?:\n\s*\d+\.\s+.+)*)/gm, (list) => {
    const items = list
      .trim()
      .split("\n")
      .map((line) => line.replace(/^\s*\d+\.\s+/, "").trim())
      .map((item) => `<li>${item}</li>`)
      .join("");
    return `<ol>${items}</ol>`;
  });

  const blocks = html
    .split(/\n{2,}/)
    .map((block) => block.trim())
    .filter(Boolean)
    .map((block) => {
      if (
        block.startsWith("<h") ||
        block.startsWith("<pre>") ||
        block.startsWith("<ul>") ||
        block.startsWith("<ol>")
      ) {
        return block;
      }
      return `<p>${block.replace(/\n/g, "<br>")}</p>`;
    });

  html = blocks.join("");

  html = html.replace(/@@CODEBLOCK_(\d+)@@/g, (_, index) => {
    const code = escapeHtml(codeBlocks[Number(index)] ?? "");
    return `<pre><code>${code}</code></pre>`;
  });

  html = html.replace(/@@INLINECODE_(\d+)@@/g, (_, index) => {
    const code = escapeHtml(inlineCode[Number(index)] ?? "");
    return `<code>${code}</code>`;
  });

  return html;
}

app.registerExtension({
  name: "PreviewAsMarkdown.Render",

  async nodeCreated(node) {
    if (node.comfyClass !== "PreviewAsMarkdown") {
      return;
    }

    ensureStyles();

    const wrapper = document.createElement("div");
    wrapper.className = "lc-md-preview-wrapper";

    const container = document.createElement("div");
    container.className = "lc-md-preview";
    container.innerHTML =
      '<div class="lc-md-preview-empty">Waiting for input...</div>';

    wrapper.appendChild(container);

    const previewWidget = node.addDOMWidget(
      "preview",
      "markdown_preview",
      wrapper,
      {
        serialize: false,
        hideOnZoom: false,
      },
    );

    previewWidget.computeSize = function (width) {
      const available = Math.max(
        0,
        (node?.size?.[1] ?? 0) - NODE_HEIGHT_PADDING,
      );
      const height = Math.max(PREVIEW_MIN_HEIGHT, available);
      this.computedHeight = height;
      return [width, height];
    };

    const minNodeHeight = PREVIEW_MIN_HEIGHT + NODE_HEIGHT_PADDING;
    if ((node?.size?.[1] ?? 0) < minNodeHeight) {
      node.setSize([node.size?.[0] ?? 240, minNodeHeight]);
    }

    node._markdownContainer = container;

    const originalOnExecuted = node.onExecuted;
    node.onExecuted = function (message) {
      if (originalOnExecuted) {
        originalOnExecuted.apply(this, arguments);
      }

      const markdownText = message?.markdown?.[0] ?? "";
      if (markdownText && markdownText.trim()) {
        node._markdownContainer.innerHTML = renderMarkdown(markdownText);
      } else if (message?.markdown) {
        node._markdownContainer.innerHTML =
          '<div class="lc-md-preview-empty">Empty input</div>';
      }
    };
  },
});
