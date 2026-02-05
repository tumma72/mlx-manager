/**
 * Downloads state management using Svelte 5 runes.
 *
 * Manages download progress state globally so it persists across navigation.
 * Uses Server-Sent Events (SSE) for real-time progress updates.
 */

/* eslint-disable svelte/prefer-svelte-reactivity -- Using Map with reassignment for reactivity */
import { models as modelsApi } from "$api";

export interface DownloadState {
  model_id: string;
  task_id: string;
  download_id?: number;
  status:
    | "pending"
    | "starting"
    | "downloading"
    | "paused"
    | "completed"
    | "failed"
    | "cancelled";
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
   * Pause an active download.
   */
  async pauseDownload(modelId: string): Promise<void> {
    const state = this.downloads.get(modelId);
    if (!state?.download_id) return;

    await modelsApi.pauseDownload(state.download_id);

    // Close SSE connection (download is paused)
    this.closeSSE(modelId);

    // Update local state
    this.updateDownload(modelId, { status: "paused" });
  }

  /**
   * Resume a paused download.
   */
  async resumeDownload(modelId: string): Promise<void> {
    const state = this.downloads.get(modelId);
    if (!state?.download_id) return;

    const { task_id, progress, downloaded_bytes, total_bytes } =
      await modelsApi.resumeDownload(state.download_id);

    // Update state with current progress and reconnect SSE with new task_id
    this.updateDownload(modelId, {
      status: "downloading",
      task_id,
      progress,
      downloaded_bytes,
      total_bytes,
    });
    this.connectSSE(modelId, task_id);
  }

  /**
   * Cancel a download and remove from store.
   */
  async cancelDownload(modelId: string): Promise<void> {
    const state = this.downloads.get(modelId);
    if (!state?.download_id) return;

    await modelsApi.cancelDownload(state.download_id);

    // Close SSE and remove from store
    this.closeSSE(modelId);

    const newMap = new Map(this.downloads);
    newMap.delete(modelId);
    this.downloads = newMap;
  }

  /**
   * Check if a model download is paused.
   */
  isPaused(modelId: string): boolean {
    const download = this.downloads.get(modelId);
    return download?.status === "paused";
  }

  /**
   * Connect to SSE stream for download progress.
   */
  private connectSSE(modelId: string, taskId: string): void {
    // Close existing connection if any
    this.closeSSE(modelId);

    const eventSource = new EventSource(
      `/api/models/download/${taskId}/progress`,
    );
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
  private updateDownload(
    modelId: string,
    update: Partial<DownloadState>,
  ): void {
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
  reconnect(
    modelId: string,
    taskId: string,
    currentState: Partial<DownloadState>,
  ): void {
    // Only reconnect if not already connected
    if (this.eventSources.has(modelId)) {
      return;
    }

    const state: DownloadState = {
      model_id: modelId,
      task_id: taskId,
      download_id: currentState.download_id,
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
        if (download.status === "paused") {
          // Paused downloads: add to store but don't connect SSE
          const state: DownloadState = {
            model_id: download.model_id,
            task_id: download.task_id,
            download_id: download.download_id,
            status: "paused",
            progress: download.progress,
            downloaded_bytes: download.downloaded_bytes,
            total_bytes: download.total_bytes,
          };
          const newMap = new Map(this.downloads);
          newMap.set(download.model_id, state);
          this.downloads = newMap;
        } else if (
          download.status === "starting" ||
          download.status === "downloading" ||
          download.status === "pending"
        ) {
          // Active downloads: reconnect SSE
          this.reconnect(download.model_id, download.task_id, {
            download_id: download.download_id,
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
      if (
        download.status !== "completed" &&
        download.status !== "failed" &&
        download.status !== "cancelled"
      ) {
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
