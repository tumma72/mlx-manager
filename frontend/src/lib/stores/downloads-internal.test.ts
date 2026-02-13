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
