import { app } from "/scripts/app.js";
import { WorkflowUI } from "./ui.js";
import { ModelAPI } from "./models.js";
import { watchWorkflow } from "./workflow.js";

app.registerExtension({
  name: "workflow_checker",
  async setup() {
    console.log("✓ Workflow Checker loaded");
    
    let uiInstance = null;

    function collectFullWorkflow(graph = app.graph) {
      const visited = new Set();

      function safeCopy(obj) {
        if (!obj || typeof obj !== "object") return obj;
        const plain = {};
        for (const [k, v] of Object.entries(obj)) {
          if (k.startsWith("_") || k === "__vueParentComponent") continue;
          try {
            plain[k] = typeof v === "object" ? safeCopy(v) : v;
          } catch {}
        }
        return plain;
      }

      function visitGraph(g) {
        if (!g || visited.has(g)) return null;
        visited.add(g);

        const result = {
          nodes: [],
          links: safeCopy(g.links || []),
          extra: safeCopy(g.extra || {}),
        };

        const nodes = g._nodes || g.nodes || [];
        for (const node of nodes) {
          if (!node) continue;

          let nodeData;
          try {
            nodeData = node.serialize ? node.serialize() : safeCopy(node);
          } catch {
            nodeData = { id: node.id, type: node.type };
          }

          // When a widget is converted to a linked input ("link_to_input"),
          // its value is dropped from widgets_values in the serialized form.
          // Recover all widget values directly from the live node.widgets array
          // so model filenames on linked widgets are still detected.
          if (Array.isArray(node.widgets)) {
            const extraValues = node.widgets.map((w) => w.value);
            nodeData._widget_values_all = extraValues;
          }

          result.nodes.push(nodeData);

          if (node.isSubgraphNode?.() && node.subgraph) {
            const sub = visitGraph(node.subgraph);
            if (sub) nodeData.subgraph = sub;
          }
        }
        return result;
      }

      return visitGraph(graph);
    }

    async function updateWorkflowAnalysis() {
      if (!uiInstance) return;

      let workflow;
      try {
        workflow = collectFullWorkflow(app.graph);
      } catch (err) {
        console.error("Workflow traversal failed:", err);
        workflow = { nodes: [] };
      }

      // Also include the API-format workflow as a supplemental source.
      // Some nodes (e.g. UpscaleModelLoader) may not surface their widget
      // values via graph serialisation, but the API prompt always includes
      // every input value explicitly (e.g. model_name: "4x-AnimeSharp.pth").
      // Python's traverse() will recurse into _api_output automatically.
      try {
        const apiPrompt = await app.graphToPrompt?.();
        if (apiPrompt?.output) {
          workflow._api_output = apiPrompt.output;
        }
      } catch (err) {
        // non-critical – graph-format traversal is the primary path
      }

      try {
        const response = await fetch("/workflow_checker/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ workflow }),
        });

        const res = await response.json();
        res.ok ? uiInstance.showModels(res) : uiInstance.setError(res.reason || "Failed to analyze");
      } catch (e) {
        uiInstance.setError(e.message || "Request failed");
      }
    }

    function toggleWindow() {
      if (!uiInstance) {
        uiInstance = new WorkflowUI();
        uiInstance.setRefreshCallback(updateWorkflowAnalysis);
      }
      uiInstance.toggle();
    }

    function addSidebarButton() {
      const sidebar = document.querySelector("nav.side-tool-bar-container");
      if (!sidebar) {
        setTimeout(addSidebarButton, 1000);
        return;
      }

      if (document.querySelector(".workflow-checker-button")) return;

      const btn = document.createElement("button");
      btn.className = "p-button p-component p-button-icon-only p-button-text side-bar-button p-button-secondary workflow-checker-button";
      btn.type = "button";
      btn.title = "Models Used in Workflow";
      btn.setAttribute("aria-label", "Models Used in Workflow");
      btn.setAttribute("data-pc-name", "button");
      btn.setAttribute("data-p-disabled", "false");
      btn.setAttribute("data-pc-section", "root");
      btn.setAttribute("data-pd-tooltip", "true");
      btn.style.width = "100%";
      
      btn.innerHTML = `
        <div data-v-179455cd="" class="side-bar-button-content">
          <i data-v-179455cd="" class="pi pi-box side-bar-button-icon"></i>
          <span data-v-179455cd="" class="side-bar-button-label">Models</span>
        </div>
        <span class="p-button-label" data-pc-section="label">&nbsp;</span>
      `;
      
      btn.onclick = toggleWindow;

      const templateBtn = sidebar.querySelector(".templates-tab-button");
      if (templateBtn?.nextSibling) {
        sidebar.insertBefore(btn, templateBtn.nextSibling);
      } else {
        sidebar.appendChild(btn);
      }
    }

    const observer = new MutationObserver(() => {
      if (document.querySelector("nav.side-tool-bar-container")) {
        observer.disconnect();
        addSidebarButton();
      }
    });

    observer.observe(document.body, { childList: true, subtree: true });
    setTimeout(addSidebarButton, 2000);

    watchWorkflow(app);
  },
});