import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// Mock the API module before importing the store
const mockModelsApi = {
  startDownload: vi.fn(),
  getActiveDownloads: vi.fn(),
  pauseDownload: vi.fn(),
  resumeDownload: vi.fn(),
  cancelDownload: vi.fn(),
};

vi.mock("$api", () => ({
  models: mockModelsApi,
}));

vi.mock("$lib/stores", () => ({
  authStore: { token: "test-jwt-token" },
}));

describe("DownloadsStore", () => {
  // We need to dynamically import the store for each test to reset state
  let downloadsStore: Awaited<
    typeof import("./downloads.svelte")
  >["downloadsStore"];

  // Track created EventSource instances
  let mockEventSources: MockEventSource[] = [];

  // Mock EventSource class
  class MockEventSource {
    url: string;
    onmessage: ((event: MessageEvent) => void) | null = null;
    onerror: ((event: Event) => void) | null = null;
    close = vi.fn();

    constructor(url: string) {
      this.url = url;
      mockEventSources.push(this);
    }

    // Helper to simulate receiving a message
    simulateMessage(data: unknown) {
      if (this.onmessage) {
        this.onmessage({
          data: JSON.stringify(data),
        } as MessageEvent);
      }
    }

    // Helper to simulate an error
    simulateError() {
      if (this.onerror) {
        this.onerror(new Event("error"));
      }
    }
  }

  beforeEach(async () => {
    vi.clearAllMocks();
    mockEventSources = [];

    // Mock EventSource globally
    global.EventSource = MockEventSource as unknown as typeof EventSource;

    // Reset mocks
    mockModelsApi.startDownload.mockResolvedValue({ task_id: "test-task-123" });
    mockModelsApi.getActiveDownloads.mockResolvedValue([]);
    mockModelsApi.pauseDownload.mockResolvedValue(undefined);
    mockModelsApi.resumeDownload.mockResolvedValue({
      task_id: "resumed-task",
      progress: 0,
      downloaded_bytes: 0,
      total_bytes: 0,
    });
    mockModelsApi.cancelDownload.mockResolvedValue(undefined);

    // Reset modules to get a fresh store instance
    vi.resetModules();

    // Re-mock after reset
    vi.doMock("$api", () => ({
      models: mockModelsApi,
    }));

    vi.doMock("$lib/stores", () => ({
      authStore: { token: "test-jwt-token" },
    }));

    // Import fresh store
    const module = await import("./downloads.svelte");
    downloadsStore = module.downloadsStore;
  });

  afterEach(() => {
    // Clean up any SSE connections
    downloadsStore.cleanup();
    vi.resetModules();
  });

  describe("initialization", () => {
    it("starts with empty downloads map", () => {
      expect(downloadsStore.downloads.size).toBe(0);
    });

    it("getAllDownloads returns empty array initially", () => {
      expect(downloadsStore.getAllDownloads()).toEqual([]);
    });
  });

  describe("startDownload", () => {
    it("sets initial pending state", async () => {
      const promise = downloadsStore.startDownload("mlx-community/test-model");

      // Check state before API call completes
      const download = downloadsStore.getProgress("mlx-community/test-model");
      expect(download?.status).toBe("pending");
      expect(download?.progress).toBe(0);
      expect(download?.downloaded_bytes).toBe(0);
      expect(download?.total_bytes).toBe(0);

      await promise;
    });

    it("calls API to start download", async () => {
      await downloadsStore.startDownload("mlx-community/test-model");

      expect(mockModelsApi.startDownload).toHaveBeenCalledWith(
        "mlx-community/test-model",
      );
    });

    it("updates task_id after API call", async () => {
      mockModelsApi.startDownload.mockResolvedValue({ task_id: "abc-123" });

      await downloadsStore.startDownload("mlx-community/test-model");

      const download = downloadsStore.getProgress("mlx-community/test-model");
      expect(download?.task_id).toBe("abc-123");
    });

    it("connects to SSE stream for progress", async () => {
      mockModelsApi.startDownload.mockResolvedValue({ task_id: "task-456" });

      await downloadsStore.startDownload("mlx-community/test-model");

      expect(mockEventSources).toHaveLength(1);
      expect(mockEventSources[0].url).toBe(
        "/api/models/download/task-456/progress?token=test-jwt-token",
      );
    });

    it("does not start if already downloading", async () => {
      await downloadsStore.startDownload("mlx-community/test-model");

      // Try to start again
      await downloadsStore.startDownload("mlx-community/test-model");

      // Should only have one API call and one EventSource
      expect(mockModelsApi.startDownload).toHaveBeenCalledTimes(1);
      expect(mockEventSources).toHaveLength(1);
    });

    it("sets failed state on API error", async () => {
      mockModelsApi.startDownload.mockRejectedValue(
        new Error("Download failed"),
      );

      await downloadsStore.startDownload("mlx-community/test-model");

      const download = downloadsStore.getProgress("mlx-community/test-model");
      expect(download?.status).toBe("failed");
      expect(download?.error).toBe("Download failed");
    });

    it("handles non-Error exceptions", async () => {
      mockModelsApi.startDownload.mockRejectedValue("String error");

      await downloadsStore.startDownload("mlx-community/test-model");

      const download = downloadsStore.getProgress("mlx-community/test-model");
      expect(download?.status).toBe("failed");
      expect(download?.error).toBe("Download failed");
    });
  });

  describe("SSE progress updates", () => {
    it("updates progress on SSE message", async () => {
      await downloadsStore.startDownload("mlx-community/test-model");

      // Simulate progress message
      mockEventSources[0].simulateMessage({
        status: "downloading",
        progress: 50,
        downloaded_bytes: 500000,
        total_bytes: 1000000,
      });

      const download = downloadsStore.getProgress("mlx-community/test-model");
      expect(download?.status).toBe("downloading");
      expect(download?.progress).toBe(50);
      expect(download?.downloaded_bytes).toBe(500000);
      expect(download?.total_bytes).toBe(1000000);
    });

    it("handles missing optional fields in progress", async () => {
      await downloadsStore.startDownload("mlx-community/test-model");

      mockEventSources[0].simulateMessage({
        status: "downloading",
      });

      const download = downloadsStore.getProgress("mlx-community/test-model");
      expect(download?.progress).toBe(0);
      expect(download?.downloaded_bytes).toBe(0);
      expect(download?.total_bytes).toBe(0);
    });

    it("closes SSE on completion", async () => {
      await downloadsStore.startDownload("mlx-community/test-model");
      const eventSource = mockEventSources[0];

      mockEventSources[0].simulateMessage({
        status: "completed",
        progress: 100,
      });

      expect(eventSource.close).toHaveBeenCalled();
    });

    it("closes SSE on failure", async () => {
      await downloadsStore.startDownload("mlx-community/test-model");
      const eventSource = mockEventSources[0];

      mockEventSources[0].simulateMessage({
        status: "failed",
        error: "Disk full",
      });

      expect(eventSource.close).toHaveBeenCalled();
    });

    it("stores error from failed message", async () => {
      await downloadsStore.startDownload("mlx-community/test-model");

      mockEventSources[0].simulateMessage({
        status: "failed",
        error: "Disk full",
      });

      const download = downloadsStore.getProgress("mlx-community/test-model");
      expect(download?.error).toBe("Disk full");
    });

    it("closes SSE on error event", async () => {
      await downloadsStore.startDownload("mlx-community/test-model");
      const eventSource = mockEventSources[0];

      mockEventSources[0].simulateError();

      expect(eventSource.close).toHaveBeenCalled();
    });

    it("ignores parse errors from malformed messages", async () => {
      await downloadsStore.startDownload("mlx-community/test-model");

      // Simulate receiving invalid JSON
      if (mockEventSources[0].onmessage) {
        mockEventSources[0].onmessage({
          data: "not valid json",
        } as MessageEvent);
      }

      // Should not throw or update status
      const download = downloadsStore.getProgress("mlx-community/test-model");
      expect(download?.status).toBe("pending");
    });
  });

  describe("isDownloading", () => {
    it("returns false for unknown model", () => {
      expect(downloadsStore.isDownloading("mlx-community/unknown")).toBe(false);
    });

    it("returns true for pending download", async () => {
      const promise = downloadsStore.startDownload("mlx-community/test-model");

      expect(downloadsStore.isDownloading("mlx-community/test-model")).toBe(
        true,
      );

      await promise;
    });

    it("returns true for starting download", async () => {
      await downloadsStore.startDownload("mlx-community/test-model");

      mockEventSources[0].simulateMessage({
        status: "starting",
      });

      expect(downloadsStore.isDownloading("mlx-community/test-model")).toBe(
        true,
      );
    });

    it("returns true for downloading status", async () => {
      await downloadsStore.startDownload("mlx-community/test-model");

      mockEventSources[0].simulateMessage({
        status: "downloading",
        progress: 50,
      });

      expect(downloadsStore.isDownloading("mlx-community/test-model")).toBe(
        true,
      );
    });

    it("returns false for completed download", async () => {
      await downloadsStore.startDownload("mlx-community/test-model");

      mockEventSources[0].simulateMessage({
        status: "completed",
        progress: 100,
      });

      expect(downloadsStore.isDownloading("mlx-community/test-model")).toBe(
        false,
      );
    });

    it("returns false for failed download", async () => {
      await downloadsStore.startDownload("mlx-community/test-model");

      mockEventSources[0].simulateMessage({
        status: "failed",
        error: "Error",
      });

      expect(downloadsStore.isDownloading("mlx-community/test-model")).toBe(
        false,
      );
    });
  });

  describe("getProgress", () => {
    it("returns undefined for unknown model", () => {
      expect(
        downloadsStore.getProgress("mlx-community/unknown"),
      ).toBeUndefined();
    });

    it("returns download state for known model", async () => {
      await downloadsStore.startDownload("mlx-community/test-model");

      const progress = downloadsStore.getProgress("mlx-community/test-model");
      expect(progress).toBeDefined();
      expect(progress?.model_id).toBe("mlx-community/test-model");
    });
  });

  describe("getAllDownloads", () => {
    it("returns empty array when no downloads", () => {
      expect(downloadsStore.getAllDownloads()).toEqual([]);
    });

    it("returns all downloads as array", async () => {
      mockModelsApi.startDownload
        .mockResolvedValueOnce({ task_id: "task-1" })
        .mockResolvedValueOnce({ task_id: "task-2" });

      await downloadsStore.startDownload("mlx-community/model-1");
      // Mark first as completed so we can start another
      mockEventSources[0].simulateMessage({ status: "completed" });

      await downloadsStore.startDownload("mlx-community/model-2");

      const downloads = downloadsStore.getAllDownloads();
      expect(downloads).toHaveLength(2);
      expect(downloads.map((d) => d.model_id)).toContain(
        "mlx-community/model-1",
      );
      expect(downloads.map((d) => d.model_id)).toContain(
        "mlx-community/model-2",
      );
    });
  });

  describe("clearCompleted", () => {
    it("removes completed downloads", async () => {
      await downloadsStore.startDownload("mlx-community/test-model");
      mockEventSources[0].simulateMessage({ status: "completed" });

      expect(
        downloadsStore.getProgress("mlx-community/test-model"),
      ).toBeDefined();

      downloadsStore.clearCompleted();

      expect(
        downloadsStore.getProgress("mlx-community/test-model"),
      ).toBeUndefined();
    });

    it("removes failed downloads", async () => {
      await downloadsStore.startDownload("mlx-community/test-model");
      mockEventSources[0].simulateMessage({ status: "failed", error: "Error" });

      expect(
        downloadsStore.getProgress("mlx-community/test-model"),
      ).toBeDefined();

      downloadsStore.clearCompleted();

      expect(
        downloadsStore.getProgress("mlx-community/test-model"),
      ).toBeUndefined();
    });

    it("keeps in-progress downloads", async () => {
      await downloadsStore.startDownload("mlx-community/test-model");
      mockEventSources[0].simulateMessage({ status: "downloading" });

      downloadsStore.clearCompleted();

      expect(
        downloadsStore.getProgress("mlx-community/test-model"),
      ).toBeDefined();
    });
  });

  describe("reconnect", () => {
    it("reconnects to existing download SSE", () => {
      downloadsStore.reconnect("mlx-community/test-model", "existing-task", {
        status: "downloading",
        progress: 50,
      });

      expect(mockEventSources).toHaveLength(1);
      expect(mockEventSources[0].url).toBe(
        "/api/models/download/existing-task/progress?token=test-jwt-token",
      );
    });

    it("sets download state from provided values", () => {
      downloadsStore.reconnect("mlx-community/test-model", "existing-task", {
        status: "downloading",
        progress: 50,
        downloaded_bytes: 500000,
        total_bytes: 1000000,
      });

      const download = downloadsStore.getProgress("mlx-community/test-model");
      expect(download?.status).toBe("downloading");
      expect(download?.progress).toBe(50);
      expect(download?.downloaded_bytes).toBe(500000);
      expect(download?.total_bytes).toBe(1000000);
    });

    it("does not reconnect if already connected", async () => {
      await downloadsStore.startDownload("mlx-community/test-model");
      expect(mockEventSources).toHaveLength(1);

      // Try to reconnect - should be ignored
      downloadsStore.reconnect("mlx-community/test-model", "another-task", {});

      expect(mockEventSources).toHaveLength(1);
    });

    it("uses default values for missing state", () => {
      downloadsStore.reconnect("mlx-community/test-model", "existing-task", {});

      const download = downloadsStore.getProgress("mlx-community/test-model");
      expect(download?.status).toBe("downloading");
      expect(download?.progress).toBe(0);
      expect(download?.downloaded_bytes).toBe(0);
      expect(download?.total_bytes).toBe(0);
    });
  });

  describe("loadActiveDownloads", () => {
    it("fetches active downloads from API", async () => {
      await downloadsStore.loadActiveDownloads();

      expect(mockModelsApi.getActiveDownloads).toHaveBeenCalled();
    });

    it("reconnects to in-progress downloads", async () => {
      mockModelsApi.getActiveDownloads.mockResolvedValue([
        {
          model_id: "mlx-community/model-1",
          task_id: "task-1",
          status: "downloading",
          progress: 30,
          downloaded_bytes: 300000,
          total_bytes: 1000000,
        },
      ]);

      await downloadsStore.loadActiveDownloads();

      expect(mockEventSources).toHaveLength(1);
      expect(downloadsStore.getProgress("mlx-community/model-1")).toBeDefined();
    });

    it("reconnects to pending downloads", async () => {
      mockModelsApi.getActiveDownloads.mockResolvedValue([
        {
          model_id: "mlx-community/model-1",
          task_id: "task-1",
          status: "pending",
          progress: 0,
        },
      ]);

      await downloadsStore.loadActiveDownloads();

      expect(mockEventSources).toHaveLength(1);
    });

    it("reconnects to starting downloads", async () => {
      mockModelsApi.getActiveDownloads.mockResolvedValue([
        {
          model_id: "mlx-community/model-1",
          task_id: "task-1",
          status: "starting",
          progress: 0,
        },
      ]);

      await downloadsStore.loadActiveDownloads();

      expect(mockEventSources).toHaveLength(1);
    });

    it("does not reconnect to completed downloads", async () => {
      mockModelsApi.getActiveDownloads.mockResolvedValue([
        {
          model_id: "mlx-community/model-1",
          task_id: "task-1",
          status: "completed",
          progress: 100,
        },
      ]);

      await downloadsStore.loadActiveDownloads();

      expect(mockEventSources).toHaveLength(0);
    });

    it("does not reconnect to failed downloads", async () => {
      mockModelsApi.getActiveDownloads.mockResolvedValue([
        {
          model_id: "mlx-community/model-1",
          task_id: "task-1",
          status: "failed",
          progress: 0,
        },
      ]);

      await downloadsStore.loadActiveDownloads();

      expect(mockEventSources).toHaveLength(0);
    });

    it("handles API errors gracefully", async () => {
      mockModelsApi.getActiveDownloads.mockRejectedValue(
        new Error("Network error"),
      );

      // Should not throw
      await downloadsStore.loadActiveDownloads();

      expect(mockEventSources).toHaveLength(0);
    });
  });

  describe("pauseDownload", () => {
    it("calls API with download_id and closes SSE", async () => {
      // Use reconnect to set up a download with download_id and SSE
      downloadsStore.reconnect("mlx-community/test-model", "task-1", {
        download_id: 42,
        status: "downloading",
        progress: 30,
      });
      const eventSource = mockEventSources[0];

      await downloadsStore.pauseDownload("mlx-community/test-model");

      expect(mockModelsApi.pauseDownload).toHaveBeenCalledWith(42);
      expect(eventSource.close).toHaveBeenCalled();

      const download = downloadsStore.getProgress("mlx-community/test-model");
      expect(download?.status).toBe("paused");
    });

    it("does nothing if model has no download_id", async () => {
      await downloadsStore.startDownload("mlx-community/test-model");

      // No download_id set (initial state has no download_id)
      await downloadsStore.pauseDownload("mlx-community/test-model");

      expect(mockModelsApi.pauseDownload).not.toHaveBeenCalled();
    });

    it("does nothing for unknown model", async () => {
      await downloadsStore.pauseDownload("mlx-community/unknown");

      expect(mockModelsApi.pauseDownload).not.toHaveBeenCalled();
    });
  });

  describe("resumeDownload", () => {
    it("calls API and reconnects SSE with new task_id", async () => {
      // Set up a paused download with download_id
      downloadsStore.reconnect("mlx-community/test-model", "task-1", {
        download_id: 42,
        status: "downloading",
        progress: 30,
      });
      // Pause it first
      await downloadsStore.pauseDownload("mlx-community/test-model");

      mockModelsApi.resumeDownload.mockResolvedValue({
        task_id: "new-task-789",
        progress: 45,
        downloaded_bytes: 4500000,
        total_bytes: 10000000,
      });

      await downloadsStore.resumeDownload("mlx-community/test-model");

      expect(mockModelsApi.resumeDownload).toHaveBeenCalledWith(42);

      const download = downloadsStore.getProgress("mlx-community/test-model");
      expect(download?.status).toBe("downloading");
      expect(download?.task_id).toBe("new-task-789");
      expect(download?.progress).toBe(45);
      expect(download?.downloaded_bytes).toBe(4500000);
      expect(download?.total_bytes).toBe(10000000);

      // New SSE connection created
      const lastEventSource = mockEventSources[mockEventSources.length - 1];
      expect(lastEventSource.url).toBe(
        "/api/models/download/new-task-789/progress?token=test-jwt-token",
      );
    });

    it("does nothing if model has no download_id", async () => {
      await downloadsStore.startDownload("mlx-community/test-model");

      await downloadsStore.resumeDownload("mlx-community/test-model");

      expect(mockModelsApi.resumeDownload).not.toHaveBeenCalled();
    });

    it("does nothing for unknown model", async () => {
      await downloadsStore.resumeDownload("mlx-community/unknown");

      expect(mockModelsApi.resumeDownload).not.toHaveBeenCalled();
    });
  });

  describe("cancelDownload", () => {
    it("calls API, closes SSE, and removes from store", async () => {
      // Use reconnect to set up a download with download_id and SSE
      downloadsStore.reconnect("mlx-community/test-model", "task-1", {
        download_id: 42,
        status: "downloading",
        progress: 30,
      });
      const eventSource = mockEventSources[0];

      await downloadsStore.cancelDownload("mlx-community/test-model");

      expect(mockModelsApi.cancelDownload).toHaveBeenCalledWith(42);
      expect(eventSource.close).toHaveBeenCalled();
      expect(
        downloadsStore.getProgress("mlx-community/test-model"),
      ).toBeUndefined();
    });

    it("does nothing if model has no download_id", async () => {
      await downloadsStore.startDownload("mlx-community/test-model");

      await downloadsStore.cancelDownload("mlx-community/test-model");

      expect(mockModelsApi.cancelDownload).not.toHaveBeenCalled();
    });

    it("does nothing for unknown model", async () => {
      await downloadsStore.cancelDownload("mlx-community/unknown");

      expect(mockModelsApi.cancelDownload).not.toHaveBeenCalled();
    });
  });

  describe("isPaused", () => {
    it("returns false for unknown model", () => {
      expect(downloadsStore.isPaused("mlx-community/unknown")).toBe(false);
    });

    it("returns false for downloading model", async () => {
      await downloadsStore.startDownload("mlx-community/test-model");
      mockEventSources[0].simulateMessage({ status: "downloading" });

      expect(downloadsStore.isPaused("mlx-community/test-model")).toBe(false);
    });

    it("returns true for paused model", async () => {
      // Use reconnect + pause to set up paused state with download_id
      downloadsStore.reconnect("mlx-community/test-model", "task-1", {
        download_id: 42,
        status: "downloading",
      });
      await downloadsStore.pauseDownload("mlx-community/test-model");

      expect(downloadsStore.isPaused("mlx-community/test-model")).toBe(true);
    });
  });

  describe("loadActiveDownloads with paused downloads", () => {
    it("loads paused downloads without SSE connection", async () => {
      mockModelsApi.getActiveDownloads.mockResolvedValue([
        {
          model_id: "mlx-community/paused-model",
          task_id: "task-paused",
          download_id: 99,
          status: "paused",
          progress: 50,
          downloaded_bytes: 500000,
          total_bytes: 1000000,
        },
      ]);

      await downloadsStore.loadActiveDownloads();

      // No SSE connection for paused downloads
      expect(mockEventSources).toHaveLength(0);

      // But state is tracked in the store
      const download = downloadsStore.getProgress(
        "mlx-community/paused-model",
      );
      expect(download).toBeDefined();
      expect(download?.status).toBe("paused");
      expect(download?.download_id).toBe(99);
      expect(download?.progress).toBe(50);
      expect(download?.downloaded_bytes).toBe(500000);
      expect(download?.total_bytes).toBe(1000000);
    });

    it("loads mix of paused and active downloads", async () => {
      mockModelsApi.getActiveDownloads.mockResolvedValue([
        {
          model_id: "mlx-community/paused-model",
          task_id: "task-paused",
          download_id: 99,
          status: "paused",
          progress: 50,
          downloaded_bytes: 500000,
          total_bytes: 1000000,
        },
        {
          model_id: "mlx-community/active-model",
          task_id: "task-active",
          download_id: 100,
          status: "downloading",
          progress: 30,
          downloaded_bytes: 300000,
          total_bytes: 1000000,
        },
      ]);

      await downloadsStore.loadActiveDownloads();

      // Only active download gets SSE connection
      expect(mockEventSources).toHaveLength(1);
      expect(mockEventSources[0].url).toBe(
        "/api/models/download/task-active/progress?token=test-jwt-token",
      );

      // Both are in the store
      expect(
        downloadsStore.getProgress("mlx-community/paused-model"),
      ).toBeDefined();
      expect(
        downloadsStore.getProgress("mlx-community/active-model"),
      ).toBeDefined();
    });
  });

  describe("clearCompleted with cancelled downloads", () => {
    it("removes cancelled downloads", async () => {
      await downloadsStore.startDownload("mlx-community/test-model");
      mockEventSources[0].simulateMessage({
        status: "cancelled",
      });

      expect(
        downloadsStore.getProgress("mlx-community/test-model"),
      ).toBeDefined();

      downloadsStore.clearCompleted();

      expect(
        downloadsStore.getProgress("mlx-community/test-model"),
      ).toBeUndefined();
    });
  });

  describe("cleanup", () => {
    it("closes all SSE connections", async () => {
      mockModelsApi.startDownload
        .mockResolvedValueOnce({ task_id: "task-1" })
        .mockResolvedValueOnce({ task_id: "task-2" });

      await downloadsStore.startDownload("mlx-community/model-1");
      mockEventSources[0].simulateMessage({ status: "completed" });
      await downloadsStore.startDownload("mlx-community/model-2");

      downloadsStore.cleanup();

      // Second EventSource should be closed (first was already closed on completion)
      expect(mockEventSources[1].close).toHaveBeenCalled();
    });
  });

  describe("closing existing connection on new start", () => {
    it("closes existing SSE when starting same model again", async () => {
      mockModelsApi.startDownload
        .mockResolvedValueOnce({ task_id: "task-1" })
        .mockResolvedValueOnce({ task_id: "task-2" });

      await downloadsStore.startDownload("mlx-community/test-model");
      const firstEventSource = mockEventSources[0];

      // Mark as completed so we can start again
      mockEventSources[0].simulateMessage({ status: "completed" });

      await downloadsStore.startDownload("mlx-community/test-model");

      // First connection should have been closed
      expect(firstEventSource.close).toHaveBeenCalled();
    });
  });
});
