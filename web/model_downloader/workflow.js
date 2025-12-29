let workflowChangeCallback = null;

export function setWorkflowChangeCallback(callback) {
  workflowChangeCallback = callback;
}

export function watchWorkflow(app) {
  const origLoad = app.loadGraphData;

  app.loadGraphData = async function (graphData) {
    const found = new Set();

    for (const node of graphData.nodes || []) {
      const traverse = (obj) => {
        if (typeof obj === "string" && obj.endsWith(".safetensors")) {
          found.add(obj);
        } else if (Array.isArray(obj)) {
          obj.forEach(traverse);
        } else if (obj && typeof obj === "object") {
          Object.values(obj).forEach(traverse);
        }
      };
      traverse(node);
      if (Array.isArray(node.widgets_values)) {
        node.widgets_values.forEach(traverse);
      }
    }

    // Wait for graph load to complete, then trigger callback
    const result = await origLoad.apply(this, arguments);
    
    // Defer callback to next tick so graph is fully loaded
    setTimeout(() => {
      if (workflowChangeCallback) {
        workflowChangeCallback();
      }
    }, 0);
    
    return result;
  };
}