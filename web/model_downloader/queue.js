/**
 * Simple queue poller that polls ONE endpoint for all queue state.
 * No more DownloadManager overhead.
 */

export class QueuePoller {
  constructor() {
    this.pollInterval = null;
    this.listeners = [];
  }

  onQueueUpdate(callback) {
    this.listeners.push(callback);
  }

  start() {
    if (this.pollInterval) return;
    this.pollInterval = setInterval(() => this.poll(), 1000);
  }

  stop() {
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
    }
  }

  async poll() {
    try {
      const res = await fetch("/workflow_checker/queue_status");
      if (!res.ok) return;
      
      const data = await res.json();
      if (data.ok && data.queue) {
        this.listeners.forEach(cb => cb(data.queue));
      }
    } catch (err) {
      console.error("Queue poll error:", err);
    }
  }
}