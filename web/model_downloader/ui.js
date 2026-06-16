import { QueuePoller } from "./queue.js";
import { setWorkflowChangeCallback } from "./workflow.js";

let maxDownloadSpeed = "";
let poller = new QueuePoller();

async function showAddModelDialog(defaultName = "", parentPanel) {
  const modal = document.createElement("div");
  modal.id = "add-model-modal";
  
  Object.assign(modal.style, {
    position: "absolute",
    left: "0", right: "0", bottom: "0",
    background: "#1e1e1e",
    borderTop: "1px solid #444",
    padding: "16px 20px",
    color: "#fff",
    display: "flex",
    flexDirection: "column",
    gap: "12px",
    zIndex: "10",
    boxShadow: "0 -2px 10px rgba(0,0,0,0.5)",
  });

  modal.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <h3 style="margin:0;font-size:16px;font-weight:600;">Add Model</h3>
      <button id="am_close" type="button" style="background:none;border:none;color:#aaa;font-size:20px;cursor:pointer;padding:0;line-height:1;">×</button>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
      <label style="display:flex;flex-direction:column;gap:4px;font-size:13px;">
        Filename
        <input id="am_filename" type="text" value="${defaultName}" style="padding:6px 8px;border-radius:4px;border:1px solid #555;background:#111;color:#fff;font-family:inherit;">
      </label>
      <label style="display:flex;flex-direction:column;gap:4px;font-size:13px;">
        Path
        <input id="am_path" type="text" placeholder="diffusion_models" style="padding:6px 8px;border-radius:4px;border:1px solid #555;background:#111;color:#fff;font-family:inherit;">
      </label>
      <label style="display:flex;flex-direction:column;gap:4px;font-size:13px;">
        Size
        <input id="am_size" type="text" placeholder="e.g. 18.4 GB" style="padding:6px 8px;border-radius:4px;border:1px solid #555;background:#111;color:#fff;font-family:inherit;">
      </label>
      <label style="display:flex;flex-direction:column;gap:4px;font-size:13px;">
        URL
        <input id="am_url" type="text" placeholder="https://..." style="padding:6px 8px;border-radius:4px;border:1px solid #555;background:#111;color:#fff;font-family:inherit;">
      </label>
    </div>
    <div style="display:flex;justify-content:flex-end;gap:10px;align-items:center;">
      <div id="am_msg" style="font-size:13px;color:#aaa;margin-right:auto;"></div>
      <button id="am_cancel" type="button" style="padding:6px 14px;background:#333;border:1px solid #555;border-radius:4px;color:#fff;cursor:pointer;font-family:inherit;">Cancel</button>
      <button id="am_submit" type="button" style="padding:6px 14px;background:#555;border:1px solid #777;border-radius:4px;color:#fff;cursor:pointer;font-family:inherit;">Submit</button>
    </div>
  `;

  const close = () => modal.remove();
  modal.querySelector("#am_close").onclick = close;
  modal.querySelector("#am_cancel").onclick = close;
  
  modal.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      modal.querySelector("#am_submit").click();
    }
  });

  modal.querySelector("#am_submit").onclick = async () => {
    const filename = modal.querySelector("#am_filename").value.trim();
    const path = modal.querySelector("#am_path").value.trim();
    const size = modal.querySelector("#am_size").value.trim();
    const url = modal.querySelector("#am_url").value.trim();
    const msg = modal.querySelector("#am_msg");

    if (!filename || !path) {
      msg.textContent = "Filename and Path required.";
      msg.style.color = "#f88";
      return;
    }

    msg.textContent = "Adding model...";
    msg.style.color = "#aaa";

    try {
      const resp = await fetch("/workflow_checker/add_model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename, path, size, url }),
      });

      const result = await resp.json();

      if (result.ok) {
        msg.textContent = "Model added!";
        msg.style.color = "#7cf";
        setTimeout(() => {
          modal.remove();
          document.querySelector("#workflow-refresh-btn")?.click();
        }, 800);
      } else {
        msg.textContent = result.error || "Unknown error";
        msg.style.color = "#f88";
      }
    } catch (err) {
      msg.textContent = "Network error.";
      msg.style.color = "#f88";
    }
  };

  document.getElementById("add-model-modal")?.remove();
  parentPanel.appendChild(modal);
}

export class WorkflowUI {
  constructor() {
    this.panel = document.createElement("div");
    this.currentModels = [];
    this.refreshCallback = null;
    this.queueState = {};
    
    Object.assign(this.panel.style, {
      position: "fixed",
      top: "8%", left: "25%",
      width: "50%", height: "60%",
      background: "rgba(20,20,20,0.95)",
      color: "#fff",
      border: "1px solid #444",
      borderRadius: "8px",
      boxShadow: "0 0 20px rgba(0,0,0,0.5)",
      zIndex: "9999",
      fontFamily: "monospace",
      resize: "both",
      display: "none",
      flexDirection: "column",
    });

    this.header = document.createElement("div");
    Object.assign(this.header.style, {
      padding: "8px 12px",
      background: "#222",
      cursor: "move",
      fontWeight: "bold",
      userSelect: "none",
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
    });
    this.header.textContent = "Models Used in Workflow";

    const closeBtn = document.createElement("button");
    closeBtn.textContent = "×";
    Object.assign(closeBtn.style, {
      background: "none",
      color: "#fff",
      border: "none",
      cursor: "pointer",
      fontSize: "16px",
    });
    closeBtn.onclick = () => this.toggle();
    this.header.appendChild(closeBtn);

    this.content = document.createElement("div");
    this.content.style.cssText = "padding:12px;overflow-y:auto;flex:1;";

    this.panel.appendChild(this.header);
    this.panel.appendChild(this.content);
    document.body.appendChild(this.panel);
    
    this.makeDraggable();
    
    poller.onQueueUpdate((queueState) => this.onQueueUpdate(queueState));
  }

  toggle() {
    const isHidden = this.panel.style.display === "none";
    this.panel.style.display = isHidden ? "flex" : "none";
    
    // If opening, refresh and load queue state
    if (isHidden) {
      this.loadInitialQueueState();
      if (this.refreshCallback) {
        this.refreshCallback();
      }
      poller.start();
    } else {
      poller.stop();
    }
  }

  async loadInitialQueueState() {
    try {
      const res = await fetch("/workflow_checker/queue_status");
      if (res.ok) {
        const data = await res.json();
        if (data.ok && data.queue) {
          this.queueState = data.queue;
          this.updateButtonStates();
        }
      }
    } catch (err) {
      console.error("Failed to load initial queue state:", err);
    }
  }

  setRefreshCallback(callback) {
    this.refreshCallback = callback;
    // Pass the workflow change callback so it triggers refresh only if panel is visible
    setWorkflowChangeCallback(async () => {
      // Only refresh if the panel is visible
      if (this.panel.style.display !== "none" && this.refreshCallback) {
        await this.refreshCallback();
      }
    });
  }

  onQueueUpdate(queueState) {
    this.queueState = queueState;
    this.updateButtonStates();
  }

  updateButtonStates() {
    const { downloads } = this.queueState;
    if (!downloads) return;

    for (const [downloadId, dlInfo] of Object.entries(downloads)) {
      let btn = document.querySelector(`button[data-download-id="${downloadId}"]`);
      
      // If not found by download ID, try finding by model name
      if (!btn && dlInfo.filename) {
        btn = document.querySelector(`button[data-model="${dlInfo.filename}"]`);
      }
      
      if (!btn) continue;

      btn.setAttribute("data-download-id", downloadId);
      const btnText = btn.querySelector("span:last-child");
      const progressFill = btn.querySelector("div");
      
      if (dlInfo.status === "downloading") {
        const progress = parseFloat(dlInfo.progress) || 0;
        btnText.textContent = dlInfo.phase ? dlInfo.phase : `${progress.toFixed(1)}%`;
        if (progressFill) progressFill.style.width = `${progress}%`;
        btn.style.background = "#555";
        btn.disabled = true;
        btn.style.cursor = "not-allowed";
      } else if (dlInfo.status === "completed") {
        btnText.textContent = "Found";
        btn.style.opacity = ".3";
        if (progressFill) progressFill.remove();
        btn.disabled = true;
        btn.style.cursor = "not-allowed";
      } else if (dlInfo.status === "failed") {
        btnText.textContent = "Failed";
        btn.style.background = "#f55";
        btn.disabled = true;
        btn.style.cursor = "not-allowed";
      } else if (dlInfo.status === "pending") {
        btnText.textContent = "Pending";
        btn.style.background = "#666";
        btn.disabled = true;
        btn.style.cursor = "not-allowed";
      }
    }
  }

  createControlButtons() {
    const controlBar = document.createElement("div");
    Object.assign(controlBar.style, {
      display: "flex",
      gap: "12px",
      marginBottom: "12px",
      alignItems: "flex-end",
    });

    const refreshBtn = document.createElement("button");
    refreshBtn.id = "workflow-refresh-btn";
    Object.assign(refreshBtn.style, {
      background: "#444",
      border: "1px solid #777",
      color: "#fff",
      borderRadius: "4px",
      padding: "8px 12px",
      cursor: "pointer",
      fontSize: "14px",
      display: "flex",
      alignItems: "center",
      gap: "8px",
      flex: "1",
    });
    refreshBtn.innerHTML = '<span>↻</span> Refresh';
    refreshBtn.onclick = async () => {
      if (this.refreshCallback) await this.refreshCallback();
    };
    refreshBtn.onmouseenter = () => { refreshBtn.style.background = "#555"; };
    refreshBtn.onmouseleave = () => { refreshBtn.style.background = "#444"; };

    const downloadAllBtn = document.createElement("button");
    downloadAllBtn.id = "workflow-download-all-btn";
    Object.assign(downloadAllBtn.style, {
      background: "#555",
      border: "1px solid #777",
      color: "#fff",
      borderRadius: "4px",
      padding: "8px 12px",
      cursor: "pointer",
      fontSize: "14px",
      display: "flex",
      alignItems: "center",
      gap: "8px",
      flex: "1",
    });
    downloadAllBtn.innerHTML = '<span>⬇</span> Download All';
    downloadAllBtn.onclick = async () => {
      await this.downloadAll();
    };
    downloadAllBtn.onmouseenter = () => { downloadAllBtn.style.background = "#666"; };
    downloadAllBtn.onmouseleave = () => { downloadAllBtn.style.background = "#555"; };

    const speedContainer = document.createElement("label");
    Object.assign(speedContainer.style, {
      display: "flex",
      flexDirection: "column",
      gap: "4px",
      fontSize: "12px",
      minWidth: "140px",
    });
    speedContainer.innerHTML = `
      <span style="color:#aaa;">Max Speed (MB/s)</span>
      <input id="max-speed-input" type="number" min="0" step="0.5" placeholder="Unlimited" value="${maxDownloadSpeed}" 
             style="padding:6px 8px;border-radius:4px;border:1px solid #555;background:#111;color:#fff;font-family:inherit;width:100%;">
    `;
    speedContainer.querySelector("#max-speed-input").addEventListener("input", (e) => {
      maxDownloadSpeed = e.target.value;
    });

    controlBar.appendChild(refreshBtn);
    controlBar.appendChild(downloadAllBtn);
    controlBar.appendChild(speedContainer);
    this.content.appendChild(controlBar);
  }

  async downloadAll() {
    if (!this.currentModels?.length) {
      alert("No models to download");
      return;
    }

    const { downloads } = this.queueState;
    const toDownload = this.currentModels.filter(m => {
      if (!m.available || (!m.url && !m.shards) || m.exists) return false;
      const isAlreadyQueued = Object.values(downloads || {}).some(
        d => d.filename === m.name && (d.status === "downloading" || d.status === "pending")
      );
      return !isAlreadyQueued;
    });

    if (!toDownload.length) {
      alert("No models available to download");
      return;
    }

    for (const model of toDownload) {
      try {
        const resp = await fetch("/workflow_checker/download_model", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            url: model.url || null,
            shards: model.shards || null,
            path: model.type,
            filename: model.name,
            max_speed_mbps: maxDownloadSpeed || null
          }),
        });

        const result = await resp.json();
        if (result.ok) {
          const btn = document.querySelector(`button[data-model="${model.name}"]`);
          if (btn) {
            btn.setAttribute("data-download-id", result.download_id);
          }
        }
      } catch (err) {
        console.error("Failed to start download:", model.name, err);
      }

      await new Promise(resolve => setTimeout(resolve, 200));
    }

    poller.start();
  }

  showModels(data) {
    if (!data?.models) {
      this.showError("No models found in workflow");
      return;
    }
    this.currentModels = data.models;
    this.renderModels(data.models);
  }

  setError(msg) {
    this.showError(msg);
  }

  showError(text) {
    this.content.innerHTML = "";
    this.createControlButtons();
    const errorMsg = document.createElement("p");
    errorMsg.style.color = "#f55";
    errorMsg.textContent = text;
    this.content.appendChild(errorMsg);
  }

  async renderModels(models) {
    if (!models.length) {
      this.content.innerHTML = `<p style="color:#f88;">No models detected</p>`;
      return;
    }

    this.content.innerHTML = "";
    this.createControlButtons();

    const table = document.createElement("table");
    Object.assign(table.style, {
      width: "100%",
      borderCollapse: "collapse",
      fontSize: "14px",
    });

    table.innerHTML = `
      <thead>
        <tr style="background:#333;">
          <th style="text-align:left;padding:6px;">File Name</th>
          <th style="text-align:left;padding:6px;">Type</th>
          <th style="text-align:left;padding:6px;">Size</th>
          <th style="text-align:left;padding:6px;">Action</th>
        </tr>
      </thead>
    `;

    const tbody = document.createElement("tbody");

    for (const m of models) {
      const row = this.createModelRow(m);
      tbody.appendChild(row);
    }

    table.appendChild(tbody);
    this.content.appendChild(table);
    
    // After DOM is rendered, sync button states from queue
    this.updateButtonStates();
  }

  createModelRow(model) {
    const row = document.createElement("tr");
    Object.assign(row.style, {
      background: model.available ? "#2b2b2b" : "rgb(32, 32, 32)",
      borderBottom: "1px solid #444",
    });

    const cells = [
      { content: model.name, style: "word-break:break-all;" },
      { 
        html: model.available 
          ? `<span style="color:#7cf;">${model.type}</span>` 
          : `<span style="color:#f88;">Not in models.json</span>` 
      },
      { content: model.size || "--" },
    ];

    cells.forEach(({ content, html, style = "" }) => {
      const cell = document.createElement("td");
      cell.style.padding = "6px";
      if (style) cell.style.cssText += style;
      if (!model.available) cell.style.opacity = "0.4";
      if (html) cell.innerHTML = html;
      else cell.textContent = content;
      row.appendChild(cell);
    });

    const actionCell = document.createElement("td");
    actionCell.style.padding = "6px";
    actionCell.appendChild(this.createActionButton(model));
    row.appendChild(actionCell);

    return row;
  }

  createActionButton(model) {
    const btnContainer = document.createElement("div");
    btnContainer.style.cssText = "position:relative;min-width:80px;";

    const btn = document.createElement("button");
    btn.setAttribute("data-model", model.name);
    Object.assign(btn.style, {
      padding: "3px 8px",
      background: "#555",
      color: "#fff",
      border: "1px solid #777",
      borderRadius: "4px",
      cursor: "pointer",
      minWidth: "80px",
      position: "relative",
      overflow: "hidden",
    });

    const progressFill = document.createElement("div");
    Object.assign(progressFill.style, {
      position: "absolute",
      left: "0", top: "0",
      height: "100%",
      width: "0%",
      background: "linear-gradient(90deg, #4a4 0%, #6c6 100%)",
      transition: "width 0.3s ease",
      zIndex: "0",
      pointerEvents: "none",
    });
    btn.appendChild(progressFill);

    const btnText = document.createElement("span");
    Object.assign(btnText.style, {
      position: "relative",
      zIndex: "1",
    });
    btn.appendChild(btnText);

    if (model.exists) {
      btnText.textContent = "Found";
      btn.disabled = true;
      btn.style.opacity = ".3";
      btn.style.cursor = "not-allowed";
      btnContainer.appendChild(btn);
      return btnContainer;
    }

    if (model.available) {
      btnText.textContent = "Download";
      btn.onclick = async () => {
        if (!model.url && !model.shards) {
          alert("No URL defined for this model.");
          return;
        }

        btn.disabled = true;
        btn.style.cursor = "not-allowed";
        btnText.textContent = "Pending";
        btn.style.background = "#666";

        try {
          const resp = await fetch("/workflow_checker/download_model", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              url: model.url || null,
              shards: model.shards || null,
              path: model.type,
              filename: model.name,
              max_speed_mbps: maxDownloadSpeed || null
            }),
          });

          const result = await resp.json();
          if (result.ok) {
            btn.setAttribute("data-download-id", result.download_id);
            poller.start();
          } else {
            btnText.textContent = "Failed";
            btn.style.background = "#f55";
            setTimeout(() => {
              btn.disabled = false;
              btn.style.cursor = "pointer";
              btn.style.background = "#555";
              btnText.textContent = "Download";
            }, 3000);
          }
        } catch (err) {
          console.error(err);
          btnText.textContent = "Error";
          btn.style.background = "#f55";
          setTimeout(() => {
            btn.disabled = false;
            btn.style.cursor = "pointer";
            btn.style.background = "#555";
            btnText.textContent = "Download";
          }, 3000);
        }
      };
    } else {
      btnText.textContent = "Add";
      btn.onclick = () => showAddModelDialog(model.name, this.panel);
    }

    btnContainer.appendChild(btn);
    return btnContainer;
  }

  makeDraggable() {
    let isDragging = false;
    let offsetX, offsetY;

    this.header.addEventListener("mousedown", (e) => {
      isDragging = true;
      offsetX = e.clientX - this.panel.offsetLeft;
      offsetY = e.clientY - this.panel.offsetTop;
      document.body.style.userSelect = "none";
    });

    document.addEventListener("mouseup", () => {
      isDragging = false;
      document.body.style.userSelect = "";
    });

    document.addEventListener("mousemove", (e) => {
      if (isDragging) {
        this.panel.style.left = (e.clientX - offsetX) + "px";
        this.panel.style.top = (e.clientY - offsetY) + "px";
      }
    });
  }
}