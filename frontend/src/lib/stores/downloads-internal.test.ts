/**
 * Internal edge case tests for DownloadsStore
 * Tests private method branches that are hard to hit through public API
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

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

describe("DownloadsStore - Internal Edge Cases", () => {
  let downloadsStore: Awaited<
    typeof import("./downloads.svelte")
  >["downloadsStore"];

  let mockEventSources: MockEventSource[] = [];

  class MockEventSource {
    url: string;
    onmessage: ((event: MessageEvent) => void) | null = null;
    onerror: ((event: Event) => void) | null = null;
    close = vi.fn();

    constructor(url: string) {
      this.url = url;
      mockEventSources.push(this);
    }

    simulateMessage(data: unknown) {
      if (this.onmessage) {
        this.onmessage({
          data: JSON.stringify(data),
        } as MessageEvent);
      }
    }
  }

  beforeEach(async () => {
    vi.clearAllMocks();
    mockEventSources = [];

    global.EventSource = MockEventSource as unknown as typeof EventSource;

    mockModelsApi.startDownload.mockResolvedValue({ task_id: "test-task-123" });
    mockModelsApi.getActiveDownloads.mockResolvedValue([]);

    vi.resetModules();

    vi.doMock("$api", () => ({
      models: mockModelsApi,
    }));

    vi.doMock("$lib/stores", () => ({
      authStore: { token: "test-jwt-token" },
    }));

    const module = await import("./downloads.svelte");
    downloadsStore = module.downloadsStore;
  });

  afterEach(() => {
    downloadsStore.cleanup();
    vi.resetModules();
  });

  it("reconnects SSE after error when download is in active status", async () => {
    vi.useFakeTimers();

    mockModelsApi.startDownload.mockResolvedValue({
      task_id: "task-reconnect",
      download_id: 42,
    });

    await downloadsStore.startDownload("mlx-community/reconnect-model");

    expect(mockEventSources).toHaveLength(1);
    const eventSource = mockEventSources[0];

    // Simulate an SSE message to set status to "downloading"
    eventSource.simulateMessage({
      status: "downloading",
      progress: 25,
      downloaded_bytes: 250000,
      total_bytes: 1000000,
      download_id: 42,
    });

    // Trigger SSE error
    eventSource.onerror!(new Event("error"));

    // Should have closed the SSE
    expect(eventSource.close).toHaveBeenCalled();

    // Before timer fires, no reconnection yet
    expect(mockEventSources).toHaveLength(1);

    // Advance by 3000ms to trigger reconnection
    await vi.advanceTimersByTimeAsync(3000);

    // Should have created a new EventSource for reconnection
    expect(mockEventSources).toHaveLength(2);
    expect(mockEventSources[1].url).toContain("task-reconnect");

    vi.useRealTimers();
  });

  it("does NOT reconnect SSE after error if download was removed during delay", async () => {
    vi.useFakeTimers();

    mockModelsApi.startDownload.mockResolvedValue({
      task_id: "task-cancel",
      download_id: 43,
    });
    mockModelsApi.cancelDownload.mockResolvedValue(undefined);

    await downloadsStore.startDownload("mlx-community/cancel-model");

    expect(mockEventSources).toHaveLength(1);
    const eventSource = mockEventSources[0];

    // Set status to downloading
    eventSource.simulateMessage({
      status: "downloading",
      progress: 10,
      downloaded_bytes: 100000,
      total_bytes: 1000000,
      download_id: 43,
    });

    // Trigger SSE error (starts 3s reconnection timer)
    eventSource.onerror!(new Event("error"));

    // Remove the download from the store before the timer fires
    downloadsStore.downloads = new Map();

    // Advance by 3000ms
    await vi.advanceTimersByTimeAsync(3000);

    // Should NOT have created a new EventSource — download was removed
    expect(mockEventSources).toHaveLength(1);

    vi.useRealTimers();
  });

  it("does NOT reconnect SSE after error if download status is completed", async () => {
    vi.useFakeTimers();

    mockModelsApi.startDownload.mockResolvedValue({
      task_id: "task-done",
      download_id: 44,
    });

    await downloadsStore.startDownload("mlx-community/done-model");

    expect(mockEventSources).toHaveLength(1);
    const eventSource = mockEventSources[0];

    // Simulate SSE message setting status to "completed"
    eventSource.simulateMessage({
      status: "completed",
      progress: 100,
      downloaded_bytes: 1000000,
      total_bytes: 1000000,
      download_id: 44,
    });

    // The onmessage handler closes SSE on "completed", creating no new sources
    // But let's test the onerror path: if status is completed, onerror should NOT reconnect
    // Reset the event sources tracking to just track new ones
    const countBefore = mockEventSources.length;

    // Trigger error on the (already-closed) event source
    eventSource.onerror!(new Event("error"));

    // Advance timer
    await vi.advanceTimersByTimeAsync(3000);

    // No new EventSource should have been created
    expect(mockEventSources).toHaveLength(countBefore);

    vi.useRealTimers();
  });

  it("handles SSE message for model removed from downloads map (updateDownload false branch)", async () => {
    await downloadsStore.startDownload("mlx-community/test-model");

    expect(mockEventSources).toHaveLength(1);
    const eventSource = mockEventSources[0];

    // Manually clear the downloads map to simulate race condition
    // where SSE message arrives after model is removed
    downloadsStore.downloads = new Map();

    // Simulate SSE message arriving - updateDownload will be called but current will be undefined
    eventSource.simulateMessage({
      status: "downloading",
      progress: 50,
      downloaded_bytes: 500000,
      total_bytes: 1000000,
    });

    // Should not throw or cause errors
    // The download should not be in the map (updateDownload did nothing)
    expect(
      downloadsStore.getProgress("mlx-community/test-model"),
    ).toBeUndefined();
  });
});
