export class DownloadManager {
  constructor() {
    this.activePolls = new Map();
    this.currentDownloadId = null;
    this.statusCheckInterval = null;
  }

  async startDownload(model, maxSpeed) {
    try {
      const resp = await fetch("/workflow_checker/download_model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          url: model.url,
          path: model.type,
          filename: model.name,
          max_speed_mbps: maxSpeed || null
        }),
      });

      const result = await resp.json();
      return result.ok ? result.download_id : null;
    } catch (err) {
      console.error("Download start failed:", model.name, err);
      return null;
    }
  }

  startStatusChecker(onDownloadChanged) {
    if (this.statusCheckInterval) clearInterval(this.statusCheckInterval);
    
    this.statusCheckInterval = setInterval(async () => {
      try {
        const res = await fetch("/workflow_checker/queue_status");
        const data = await res.json();
        
        if (data.ok && data.queue && data.queue.downloads) {
          const downloads = data.queue.downloads;
          const currentDownloading = Object.entries(downloads).find(([id, d]) => d.status === "downloading");
          
          // If a new download started, notify UI
          if (currentDownloading && currentDownloading[0] !== this.currentDownloadId) {
            this.currentDownloadId = currentDownloading[0];
            onDownloadChanged?.({ ...currentDownloading[1], download_id: currentDownloading[0] });
          }
          
          // If no more downloads, stop checking
          const hasPending = Object.values(downloads).some(d => d.status === "pending");
          const hasDownloading = Object.values(downloads).some(d => d.status === "downloading");
          
          if (!hasDownloading && !hasPending) {
            clearInterval(this.statusCheckInterval);
            this.statusCheckInterval = null;
            this.currentDownloadId = null;
          }
        }
      } catch (err) {
        console.error("Status check error:", err);
      }
    }, 1000);
  }

  poll(downloadId, onProgress, onComplete, onError) {
    // Only poll the current download
    if (this.currentDownloadId !== null && this.currentDownloadId !== downloadId) {
      return;
    }

    if (this.activePolls.has(downloadId)) {
      return;
    }

    this.currentDownloadId = downloadId;

    const pollInterval = setInterval(async () => {
      try {
        const statusResp = await fetch(`/workflow_checker/download_status/${downloadId}`);
        const status = await statusResp.json();

        if (!status.ok) {
          clearInterval(pollInterval);
          this.activePolls.delete(downloadId);
          if (typeof onError === 'function') {
            onError("Status check failed");
          }
          return;
        }

        const { status: downloadStatus, progress } = status;

        if (downloadStatus === "downloading") {
          if (typeof onProgress === 'function') {
            onProgress(parseFloat(progress) || 0);
          }
        } else if (downloadStatus === "completed") {
          clearInterval(pollInterval);
          this.activePolls.delete(downloadId);
          this.currentDownloadId = null;
          if (typeof onComplete === 'function') {
            onComplete();
          }
        } else if (downloadStatus === "failed") {
          clearInterval(pollInterval);
          this.activePolls.delete(downloadId);
          this.currentDownloadId = null;
          if (typeof onError === 'function') {
            onError(status.error || "Download failed");
          }
        }
      } catch (err) {
        console.error("Poll error:", err);
        clearInterval(pollInterval);
        this.activePolls.delete(downloadId);
        if (typeof onError === 'function') {
          onError(err.message);
        }
      }
    }, 1000);

    this.activePolls.set(downloadId, pollInterval);
  }

  stopPoll(downloadId) {
    const interval = this.activePolls.get(downloadId);
    if (interval) {
      clearInterval(interval);
      this.activePolls.delete(downloadId);
      if (this.currentDownloadId === downloadId) {
        this.currentDownloadId = null;
      }
    }
  }

  stopAllPolls() {
    this.activePolls.forEach(interval => clearInterval(interval));
    this.activePolls.clear();
    if (this.statusCheckInterval) clearInterval(this.statusCheckInterval);
    this.currentDownloadId = null;
  }
}