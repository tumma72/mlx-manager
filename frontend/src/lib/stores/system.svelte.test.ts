import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import type { SystemMemory, SystemInfo } from "$lib/api/types";

// Mock the API module before importing the store
vi.mock("$api", () => ({
  system: {
    memory: vi.fn(),
    info: vi.fn(),
  },
}));

// Mock the polling coordinator
vi.mock("$lib/services", () => ({
  pollingCoordinator: {
    register: vi.fn(),
    start: vi.fn(),
    stop: vi.fn(),
    refresh: vi.fn(),
  },
}));

// Helper to create mock memory data
function createMockMemory(overrides: Partial<SystemMemory> = {}): SystemMemory {
  return {
    total_gb: 128,
    available_gb: 64,
    used_gb: 64,
    percent_used: 50,
    mlx_recommended_gb: 102.4,
    ...overrides,
  };
}

// Helper to create mock system info
function createMockInfo(overrides: Partial<SystemInfo> = {}): SystemInfo {
  return {
    os_version: "macOS 15.0",
    chip: "Apple M4 Pro",
    memory_gb: 128,
    python_version: "3.12.0",
    mlx_version: "0.20.0",
    mlx_openai_server_version: "1.5.0",
    ...overrides,
  };
}

describe("SystemStore", () => {
  let systemStore: Awaited<typeof import("./system.svelte")>["systemStore"];
  let mockSystemApi: {
    memory: ReturnType<typeof vi.fn>;
    info: ReturnType<typeof vi.fn>;
  };
  let mockPollingCoordinator: {
    register: ReturnType<typeof vi.fn>;
    start: ReturnType<typeof vi.fn>;
    stop: ReturnType<typeof vi.fn>;
    refresh: ReturnType<typeof vi.fn>;
  };

  beforeEach(async () => {
    vi.clearAllMocks();

    // Reset modules to get fresh store instance
    vi.resetModules();

    // Re-mock after reset
    mockSystemApi = {
      memory: vi.fn().mockResolvedValue(createMockMemory()),
      info: vi.fn().mockResolvedValue(createMockInfo()),
    };

    mockPollingCoordinator = {
      register: vi.fn(),
      start: vi.fn(),
      stop: vi.fn(),
      refresh: vi.fn().mockResolvedValue(undefined),
    };

    vi.doMock("$api", () => ({
      system: mockSystemApi,
    }));

    vi.doMock("$lib/services", () => ({
      pollingCoordinator: mockPollingCoordinator,
    }));

    // Import fresh store
    const module = await import("./system.svelte");
    systemStore = module.systemStore;
  });

  afterEach(() => {
    vi.resetModules();
  });

  describe("initialization", () => {
    it("registers with polling coordinator on creation", () => {
      expect(mockPollingCoordinator.register).toHaveBeenCalledWith(
        "system-memory",
        expect.objectContaining({
          interval: 30000,
          minInterval: 5000,
        }),
      );
    });

    it("starts with null memory", () => {
      expect(systemStore.memory).toBeNull();
    });

    it("starts with null info", () => {
      expect(systemStore.info).toBeNull();
    });

    it("starts with loading false", () => {
      expect(systemStore.loading).toBe(false);
    });

    it("starts with no error", () => {
      expect(systemStore.error).toBeNull();
    });
  });

  describe("refreshMemory", () => {
    it("delegates to polling coordinator", async () => {
      await systemStore.refreshMemory();

      expect(mockPollingCoordinator.refresh).toHaveBeenCalledWith(
        "system-memory",
      );
    });
  });

  describe("startMemoryPolling", () => {
    it("starts polling via coordinator", () => {
      systemStore.startMemoryPolling();

      expect(mockPollingCoordinator.start).toHaveBeenCalledWith(
        "system-memory",
      );
    });
  });

  describe("stopMemoryPolling", () => {
    it("stops polling via coordinator", () => {
      systemStore.stopMemoryPolling();

      expect(mockPollingCoordinator.stop).toHaveBeenCalledWith("system-memory");
    });
  });

  describe("refreshInfo", () => {
    it("fetches system info from API", async () => {
      const mockInfo = createMockInfo();
      mockSystemApi.info.mockResolvedValue(mockInfo);

      await systemStore.refreshInfo();

      expect(mockSystemApi.info).toHaveBeenCalled();
      expect(systemStore.info).toEqual(mockInfo);
    });

    it("sets loading true during fetch", async () => {
      let loadingDuringFetch = false;
      mockSystemApi.info.mockImplementation(async () => {
        loadingDuringFetch = systemStore.loading;
        return createMockInfo();
      });

      await systemStore.refreshInfo();

      expect(loadingDuringFetch).toBe(true);
    });

    it("sets loading false after fetch completes", async () => {
      mockSystemApi.info.mockResolvedValue(createMockInfo());

      await systemStore.refreshInfo();

      expect(systemStore.loading).toBe(false);
    });

    it("sets error on fetch failure", async () => {
      mockSystemApi.info.mockRejectedValue(new Error("Network error"));

      await systemStore.refreshInfo();

      expect(systemStore.error).toBe("Network error");
    });

    it("sets loading false even on error", async () => {
      mockSystemApi.info.mockRejectedValue(new Error("Network error"));

      await systemStore.refreshInfo();

      expect(systemStore.loading).toBe(false);
    });

    it("sets generic error message for non-Error exceptions", async () => {
      mockSystemApi.info.mockRejectedValue("string error");

      await systemStore.refreshInfo();

      expect(systemStore.error).toBe("Failed to fetch system info");
    });

    it("only updates info if data changed", async () => {
      const info1 = createMockInfo({ chip: "Apple M4" });
      const info2 = createMockInfo({ chip: "Apple M4" }); // Same data
      mockSystemApi.info.mockResolvedValueOnce(info1);

      await systemStore.refreshInfo();
      const firstInfo = systemStore.info;

      mockSystemApi.info.mockResolvedValueOnce(info2);
      await systemStore.refreshInfo();

      // Should be the same object reference (not replaced)
      expect(systemStore.info).toBe(firstInfo);
    });

    it("updates info when data differs", async () => {
      const info1 = createMockInfo({ chip: "Apple M4" });
      const info2 = createMockInfo({ chip: "Apple M4 Pro" }); // Different
      mockSystemApi.info.mockResolvedValueOnce(info1);

      await systemStore.refreshInfo();
      const firstInfo = systemStore.info;

      mockSystemApi.info.mockResolvedValueOnce(info2);
      await systemStore.refreshInfo();

      // Should be replaced with new object
      expect(systemStore.info).not.toBe(firstInfo);
      expect(systemStore.info?.chip).toBe("Apple M4 Pro");
    });
  });

  describe("refresh", () => {
    it("calls both refreshMemory and refreshInfo", async () => {
      const refreshMemorySpy = vi.spyOn(systemStore, "refreshMemory");
      const refreshInfoSpy = vi.spyOn(systemStore, "refreshInfo");

      await systemStore.refresh();

      expect(refreshMemorySpy).toHaveBeenCalled();
      expect(refreshInfoSpy).toHaveBeenCalled();
    });

    it("runs both refreshes in parallel", async () => {
      let memoryStarted = false;
      let infoStarted = false;
      let memorySeesInfo = false;
      let infoSeesMemory = false;

      mockPollingCoordinator.refresh.mockImplementation(async () => {
        memoryStarted = true;
        // Check if info started before memory completes
        await new Promise((r) => setTimeout(r, 10));
        memorySeesInfo = infoStarted;
      });

      mockSystemApi.info.mockImplementation(async () => {
        infoStarted = true;
        // Check if memory started before info completes
        await new Promise((r) => setTimeout(r, 10));
        infoSeesMemory = memoryStarted;
        return createMockInfo();
      });

      await systemStore.refresh();

      // Both should see each other started (parallel execution)
      expect(memorySeesInfo).toBe(true);
      expect(infoSeesMemory).toBe(true);
    });
  });

  describe("doRefreshMemory (internal)", () => {
    it("fetches memory and updates state", async () => {
      const memory = createMockMemory({ total_gb: 256, available_gb: 128 });
      mockSystemApi.memory.mockResolvedValue(memory);

      // Get the registered refresh function
      const registerCall = mockPollingCoordinator.register.mock.calls[0];
      const refreshFn = registerCall[1].refreshFn;

      await refreshFn();

      expect(mockSystemApi.memory).toHaveBeenCalled();
      expect(systemStore.memory).toEqual(memory);
    });

    it("only updates memory if data changed", async () => {
      const memory1 = createMockMemory({ total_gb: 128 });
      const memory2 = createMockMemory({ total_gb: 128 }); // Same
      mockSystemApi.memory.mockResolvedValueOnce(memory1);

      const registerCall = mockPollingCoordinator.register.mock.calls[0];
      const refreshFn = registerCall[1].refreshFn;

      await refreshFn();
      const firstMemory = systemStore.memory;

      mockSystemApi.memory.mockResolvedValueOnce(memory2);
      await refreshFn();

      // Should be the same object (not replaced)
      expect(systemStore.memory).toBe(firstMemory);
    });

    it("updates memory when data differs", async () => {
      const memory1 = createMockMemory({ available_gb: 64 });
      const memory2 = createMockMemory({ available_gb: 32 }); // Different
      mockSystemApi.memory.mockResolvedValueOnce(memory1);

      const registerCall = mockPollingCoordinator.register.mock.calls[0];
      const refreshFn = registerCall[1].refreshFn;

      await refreshFn();
      const firstMemory = systemStore.memory;

      mockSystemApi.memory.mockResolvedValueOnce(memory2);
      await refreshFn();

      // Should be replaced
      expect(systemStore.memory).not.toBe(firstMemory);
      expect(systemStore.memory?.available_gb).toBe(32);
    });

    it("handles fetch errors gracefully", async () => {
      mockSystemApi.memory.mockRejectedValue(new Error("Network error"));

      const registerCall = mockPollingCoordinator.register.mock.calls[0];
      const refreshFn = registerCall[1].refreshFn;

      // Should not throw
      await expect(refreshFn()).resolves.toBeUndefined();
    });
  });
});
