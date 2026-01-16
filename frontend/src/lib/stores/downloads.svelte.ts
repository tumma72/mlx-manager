/**
 * Downloads state management using Svelte 5 runes.
 *
 * Manages download progress state globally so it persists across navigation.
 * Uses Server-Sent Events (SSE) for real-time progress updates.
 */

import { models as modelsApi, type ActiveDownload } from "$api";

export interface DownloadState {
  model_id: string;
  task_id: string;
  status: "pending" | "starting" | "downloading" | "completed" | "failed";
  progress: number;
  downloaded_bytes: number;
  total_bytes: number;
  error?: string;
}

class DownloadsStore {
  // Map of model_id -> download state
  downloads = $state<Map<string, DownloadState>>(new Map());

  // Track active EventSource connections by model_id
  private eventSources = new Map<string, EventSource>();

  /**
   * Start a new download for a model.
   */
  async startDownload(modelId: string): Promise<void> {
    // Don't start if already downloading
    if (this.isDownloading(modelId)) {
      return;
    }

    // Set initial state
    const initialState: DownloadState = {
      model_id: modelId,
      task_id: "",
      status: "pending",
      progress: 0,
      downloaded_bytes: 0,
      total_bytes: 0,
    };

    // Create new map to trigger reactivity
    const newMap = new Map(this.downloads);
    newMap.set(modelId, initialState);
    this.downloads = newMap;

    try {
      // Start download on backend
      const { task_id } = await modelsApi.startDownload(modelId);

      // Update with task_id
      this.updateDownload(modelId, { task_id });

      // Connect to SSE stream for progress
      this.connectSSE(modelId, task_id);
    } catch (error) {
      this.updateDownload(modelId, {
        status: "failed",
        error: error instanceof Error ? error.message : "Download failed",
      });
    }
  }

  /**
   * Connect to SSE stream for download progress.
   */
  private connectSSE(modelId: string, taskId: string): void {
    // Close existing connection if any
    this.closeSSE(modelId);

    const eventSource = new EventSource(`/api/models/download/${taskId}/progress`);
    this.eventSources.set(modelId, eventSource);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.updateDownload(modelId, {
          status: data.status,
          progress: data.progress || 0,
          downloaded_bytes: data.downloaded_bytes || 0,
          total_bytes: data.total_bytes || 0,
          error: data.error,
        });

        // Close connection when download completes or fails
        if (data.status === "completed" || data.status === "failed") {
          this.closeSSE(modelId);
        }
      } catch {
        // Ignore parse errors
      }
    };

    eventSource.onerror = () => {
      // Don't update status on error - the connection might recover
      // Just close and let the active downloads refresh handle it
      this.closeSSE(modelId);
    };
  }

  /**
   * Close SSE connection for a model.
   */
  private closeSSE(modelId: string): void {
    const eventSource = this.eventSources.get(modelId);
    if (eventSource) {
      eventSource.close();
      this.eventSources.delete(modelId);
    }
  }

  /**
   * Update download state for a model.
   */
  private updateDownload(modelId: string, update: Partial<DownloadState>): void {
    const current = this.downloads.get(modelId);
    if (current) {
      const newMap = new Map(this.downloads);
      newMap.set(modelId, { ...current, ...update });
      this.downloads = newMap;
    }
  }

  /**
   * Reconnect to an existing download (after navigation).
   */
  reconnect(modelId: string, taskId: string, currentState: Partial<DownloadState>): void {
    // Only reconnect if not already connected
    if (this.eventSources.has(modelId)) {
      return;
    }

    const state: DownloadState = {
      model_id: modelId,
      task_id: taskId,
      status: (currentState.status as DownloadState["status"]) || "downloading",
      progress: currentState.progress || 0,
      downloaded_bytes: currentState.downloaded_bytes || 0,
      total_bytes: currentState.total_bytes || 0,
    };

    const newMap = new Map(this.downloads);
    newMap.set(modelId, state);
    this.downloads = newMap;

    this.connectSSE(modelId, taskId);
  }

  /**
   * Load active downloads from backend and reconnect SSE.
   * Call this on app init to resume tracking in-progress downloads.
   */
  async loadActiveDownloads(): Promise<void> {
    try {
      const active = await modelsApi.getActiveDownloads();

      for (const download of active) {
        // Only reconnect to downloads that are still in progress
        if (
          download.status === "starting" ||
          download.status === "downloading" ||
          download.status === "pending"
        ) {
          this.reconnect(download.model_id, download.task_id, {
            status: download.status as DownloadState["status"],
            progress: download.progress,
            downloaded_bytes: download.downloaded_bytes,
            total_bytes: download.total_bytes,
          });
        }
      }
    } catch {
      // Ignore errors loading active downloads
    }
  }

  /**
   * Check if a model is currently downloading.
   */
  isDownloading(modelId: string): boolean {
    const download = this.downloads.get(modelId);
    if (!download) return false;
    return (
      download.status === "pending" ||
      download.status === "starting" ||
      download.status === "downloading"
    );
  }

  /**
   * Get download progress for a model.
   */
  getProgress(modelId: string): DownloadState | undefined {
    return this.downloads.get(modelId);
  }

  /**
   * Get all downloads as an array.
   */
  getAllDownloads(): DownloadState[] {
    return Array.from(this.downloads.values());
  }

  /**
   * Clear completed/failed downloads from the store.
   */
  clearCompleted(): void {
    const newMap = new Map<string, DownloadState>();
    for (const [modelId, download] of this.downloads) {
      if (download.status !== "completed" && download.status !== "failed") {
        newMap.set(modelId, download);
      }
    }
    this.downloads = newMap;
  }

  /**
   * Clean up all SSE connections.
   */
  cleanup(): void {
    for (const [modelId] of this.eventSources) {
      this.closeSSE(modelId);
    }
  }
}

export const downloadsStore = new DownloadsStore();
