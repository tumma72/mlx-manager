import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import type { RunningServer } from "$api";

// Mock the API module before importing the store
vi.mock("$api", () => ({
  servers: {
    list: vi.fn(),
    start: vi.fn(),
    stop: vi.fn(),
    restart: vi.fn(),
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

// Mock svelte/reactivity SvelteSet and SvelteMap
vi.mock("svelte/reactivity", () => ({
  SvelteSet: Set,
  SvelteMap: Map,
}));

// Import mocked modules
import { servers as serversApi } from "$api";
import { pollingCoordinator } from "$lib/services";

// Helper to create mock server
function createMockServer(
  overrides: Partial<RunningServer> = {}
): RunningServer {
  return {
    profile_id: 1,
    profile_name: "Test Server",
    pid: 12345,
    port: 10240,
    health_status: "healthy",
    uptime_seconds: 3600,
    memory_mb: 512,
    memory_percent: 25,
    cpu_percent: 10,
    ...overrides,
  };
}

describe("ServerStore", () => {
  // We need to dynamically import the store for each test to reset state
  let serverStore: Awaited<
    typeof import("./servers.svelte")
  >["serverStore"];

  beforeEach(async () => {
    vi.clearAllMocks();

    // Clear window state for HMR preservation
    if (typeof window !== "undefined") {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      delete (window as any).__serverStoreHmrState;
    }

    // Reset modules to get a fresh store instance
    vi.resetModules();

    // Re-mock after reset
    vi.doMock("$api", () => ({
      servers: {
        list: vi.fn().mockResolvedValue([]),
        start: vi.fn().mockResolvedValue(undefined),
        stop: vi.fn().mockResolvedValue(undefined),
        restart: vi.fn().mockResolvedValue(undefined),
      },
    }));

    vi.doMock("$lib/services", () => ({
      pollingCoordinator: {
        register: vi.fn(),
        start: vi.fn(),
        stop: vi.fn(),
        refresh: vi.fn().mockResolvedValue(undefined),
      },
    }));

    vi.doMock("svelte/reactivity", () => ({
      SvelteSet: Set,
      SvelteMap: Map,
    }));

    // Import fresh store
    const module = await import("./servers.svelte");
    serverStore = module.serverStore;
  });

  afterEach(() => {
    vi.resetModules();
  });

  describe("initialization", () => {
    it("registers with polling coordinator on creation", async () => {
      const { pollingCoordinator } = await import("$lib/services");

      expect(pollingCoordinator.register).toHaveBeenCalledWith(
        "servers",
        expect.objectContaining({
          interval: 5000,
          minInterval: 1000,
        })
      );
    });

    it("starts with empty servers array", () => {
      expect(serverStore.servers).toEqual([]);
    });

    it("starts with loading false", () => {
      expect(serverStore.loading).toBe(false);
    });

    it("starts with error null", () => {
      expect(serverStore.error).toBe(null);
    });
  });

  describe("start", () => {
    it("adds profile to startingProfiles", async () => {
      const { servers: serversApi } = await import("$api");
      vi.mocked(serversApi.start).mockResolvedValue(undefined);

      await serverStore.start(42);

      expect(serverStore.isStarting(42)).toBe(true);
    });

    it("clears any existing failure", async () => {
      const { servers: serversApi } = await import("$api");
      vi.mocked(serversApi.start).mockResolvedValue(undefined);

      // Set up a failure first
      serverStore.markStartupFailed(42, "Previous error");
      expect(serverStore.isFailed(42)).toBe(true);

      await serverStore.start(42);

      expect(serverStore.isFailed(42)).toBe(false);
    });

    it("calls API to start server", async () => {
      const { servers: serversApi } = await import("$api");
      vi.mocked(serversApi.start).mockResolvedValue(undefined);

      await serverStore.start(42);

      expect(serversApi.start).toHaveBeenCalledWith(42);
    });
  });

  describe("stop", () => {
    it("removes profile from startingProfiles", async () => {
      const { servers: serversApi } = await import("$api");
      vi.mocked(serversApi.start).mockResolvedValue(undefined);
      vi.mocked(serversApi.stop).mockResolvedValue(undefined);

      await serverStore.start(42);
      expect(serverStore.isStarting(42)).toBe(true);

      await serverStore.stop(42);
      expect(serverStore.isStarting(42)).toBe(false);
    });

    it("clears any existing failure", async () => {
      const { servers: serversApi } = await import("$api");
      vi.mocked(serversApi.stop).mockResolvedValue(undefined);

      serverStore.markStartupFailed(42, "Previous error");
      expect(serverStore.isFailed(42)).toBe(true);

      await serverStore.stop(42);

      expect(serverStore.isFailed(42)).toBe(false);
    });

    it("calls API to stop server", async () => {
      const { servers: serversApi } = await import("$api");
      vi.mocked(serversApi.stop).mockResolvedValue(undefined);

      await serverStore.stop(42);

      expect(serversApi.stop).toHaveBeenCalledWith(42, false);
    });

    it("calls API with force flag when specified", async () => {
      const { servers: serversApi } = await import("$api");
      vi.mocked(serversApi.stop).mockResolvedValue(undefined);

      await serverStore.stop(42, true);

      expect(serversApi.stop).toHaveBeenCalledWith(42, true);
    });

    it("triggers refresh after stop", async () => {
      const { servers: serversApi } = await import("$api");
      const { pollingCoordinator } = await import("$lib/services");
      vi.mocked(serversApi.stop).mockResolvedValue(undefined);

      await serverStore.stop(42);

      expect(pollingCoordinator.refresh).toHaveBeenCalledWith("servers");
    });
  });

  describe("restart", () => {
    it("adds profile to startingProfiles", async () => {
      const { servers: serversApi } = await import("$api");
      vi.mocked(serversApi.restart).mockResolvedValue(undefined);

      await serverStore.restart(42);

      expect(serverStore.isStarting(42)).toBe(true);
    });

    it("clears any existing failure", async () => {
      const { servers: serversApi } = await import("$api");
      vi.mocked(serversApi.restart).mockResolvedValue(undefined);

      serverStore.markStartupFailed(42, "Previous error");
      expect(serverStore.isFailed(42)).toBe(true);

      await serverStore.restart(42);

      expect(serverStore.isFailed(42)).toBe(false);
    });

    it("calls API to restart server", async () => {
      const { servers: serversApi } = await import("$api");
      vi.mocked(serversApi.restart).mockResolvedValue(undefined);

      await serverStore.restart(42);

      expect(serversApi.restart).toHaveBeenCalledWith(42);
    });
  });

  describe("markStartupSuccess", () => {
    it("removes profile from startingProfiles", async () => {
      const { servers: serversApi } = await import("$api");
      vi.mocked(serversApi.start).mockResolvedValue(undefined);

      await serverStore.start(42);
      expect(serverStore.isStarting(42)).toBe(true);

      serverStore.markStartupSuccess(42);

      expect(serverStore.isStarting(42)).toBe(false);
    });

    it("clears any existing failure", () => {
      serverStore.markStartupFailed(42, "Error");
      expect(serverStore.isFailed(42)).toBe(true);

      serverStore.markStartupSuccess(42);

      expect(serverStore.isFailed(42)).toBe(false);
    });

    it("triggers refresh", async () => {
      const { pollingCoordinator } = await import("$lib/services");

      serverStore.markStartupSuccess(42);

      expect(pollingCoordinator.refresh).toHaveBeenCalledWith("servers");
    });
  });

  describe("markStartupFailed", () => {
    it("removes profile from startingProfiles", async () => {
      const { servers: serversApi } = await import("$api");
      vi.mocked(serversApi.start).mockResolvedValue(undefined);

      await serverStore.start(42);
      expect(serverStore.isStarting(42)).toBe(true);

      serverStore.markStartupFailed(42, "Error message");

      expect(serverStore.isStarting(42)).toBe(false);
    });

    it("adds profile to failedProfiles", () => {
      serverStore.markStartupFailed(42, "Error message");

      expect(serverStore.isFailed(42)).toBe(true);
    });

    it("stores error message", () => {
      serverStore.markStartupFailed(42, "Error message");

      const failure = serverStore.getFailure(42);
      expect(failure?.error).toBe("Error message");
    });

    it("stores error details", () => {
      serverStore.markStartupFailed(42, "Error message", "Stack trace here");

      const failure = serverStore.getFailure(42);
      expect(failure?.details).toBe("Stack trace here");
    });

    it("preserves detailsOpen state when re-failing", () => {
      serverStore.markStartupFailed(42, "First error");
      serverStore.toggleDetailsOpen(42);

      // Get the current state
      const firstFailure = serverStore.getFailure(42);
      expect(firstFailure?.detailsOpen).toBe(true);

      // Fail again - should preserve detailsOpen
      serverStore.markStartupFailed(42, "Second error");

      const secondFailure = serverStore.getFailure(42);
      expect(secondFailure?.detailsOpen).toBe(true);
      expect(secondFailure?.error).toBe("Second error");
    });

    it("triggers refresh", async () => {
      const { pollingCoordinator } = await import("$lib/services");

      serverStore.markStartupFailed(42, "Error");

      expect(pollingCoordinator.refresh).toHaveBeenCalledWith("servers");
    });
  });

  describe("toggleDetailsOpen", () => {
    it("toggles detailsOpen state", () => {
      serverStore.markStartupFailed(42, "Error");

      // Initially closed
      expect(serverStore.getFailure(42)?.detailsOpen).toBe(false);

      // Toggle open
      serverStore.toggleDetailsOpen(42);
      expect(serverStore.getFailure(42)?.detailsOpen).toBe(true);

      // Toggle closed
      serverStore.toggleDetailsOpen(42);
      expect(serverStore.getFailure(42)?.detailsOpen).toBe(false);
    });

    it("does nothing if profile is not failed", () => {
      // Should not throw
      serverStore.toggleDetailsOpen(42);

      // No failure should exist
      expect(serverStore.getFailure(42)).toBeUndefined();
    });
  });

  describe("clearFailure", () => {
    it("removes profile from failedProfiles", () => {
      serverStore.markStartupFailed(42, "Error");
      expect(serverStore.isFailed(42)).toBe(true);

      serverStore.clearFailure(42);

      expect(serverStore.isFailed(42)).toBe(false);
    });

    it("does nothing if profile is not failed", () => {
      // Should not throw
      serverStore.clearFailure(42);

      expect(serverStore.isFailed(42)).toBe(false);
    });
  });

  describe("isRunning", () => {
    it("returns false when server not in list", () => {
      expect(serverStore.isRunning(42)).toBe(false);
    });

    it("returns false when server in list but starting", async () => {
      const { servers: serversApi } = await import("$api");
      vi.mocked(serversApi.start).mockResolvedValue(undefined);

      // Add server to list
      serverStore.servers.push(createMockServer({ profile_id: 42 }));

      // Mark as starting
      await serverStore.start(42);

      expect(serverStore.isRunning(42)).toBe(false);
    });

    it("returns true when server in list and not starting", () => {
      serverStore.servers.push(createMockServer({ profile_id: 42 }));

      expect(serverStore.isRunning(42)).toBe(true);
    });
  });

  describe("isStarting", () => {
    it("returns false initially", () => {
      expect(serverStore.isStarting(42)).toBe(false);
    });

    it("returns true after start called", async () => {
      const { servers: serversApi } = await import("$api");
      vi.mocked(serversApi.start).mockResolvedValue(undefined);

      await serverStore.start(42);

      expect(serverStore.isStarting(42)).toBe(true);
    });
  });

  describe("isFailed", () => {
    it("returns false initially", () => {
      expect(serverStore.isFailed(42)).toBe(false);
    });

    it("returns true after markStartupFailed called", () => {
      serverStore.markStartupFailed(42, "Error");

      expect(serverStore.isFailed(42)).toBe(true);
    });
  });

  describe("getFailure", () => {
    it("returns undefined when not failed", () => {
      expect(serverStore.getFailure(42)).toBeUndefined();
    });

    it("returns failure object when failed", () => {
      serverStore.markStartupFailed(42, "Error message", "Details");

      const failure = serverStore.getFailure(42);
      expect(failure).toEqual({
        error: "Error message",
        details: "Details",
        detailsOpen: false,
      });
    });
  });

  describe("getServer", () => {
    it("returns undefined when server not in list", () => {
      expect(serverStore.getServer(42)).toBeUndefined();
    });

    it("returns server when in list", () => {
      const server = createMockServer({ profile_id: 42 });
      serverStore.servers.push(server);

      expect(serverStore.getServer(42)).toEqual(server);
    });
  });

  describe("profile polling tracking", () => {
    it("isProfilePolling returns false initially", () => {
      expect(serverStore.isProfilePolling(42)).toBe(false);
    });

    it("startProfilePolling returns true and marks profile as polling", () => {
      const result = serverStore.startProfilePolling(42);

      expect(result).toBe(true);
      expect(serverStore.isProfilePolling(42)).toBe(true);
    });

    it("startProfilePolling returns false if already polling", () => {
      serverStore.startProfilePolling(42);

      const result = serverStore.startProfilePolling(42);

      expect(result).toBe(false);
    });

    it("stopProfilePolling marks profile as not polling", () => {
      serverStore.startProfilePolling(42);
      expect(serverStore.isProfilePolling(42)).toBe(true);

      serverStore.stopProfilePolling(42);

      expect(serverStore.isProfilePolling(42)).toBe(false);
    });
  });

  describe("refresh", () => {
    it("delegates to polling coordinator", async () => {
      const { pollingCoordinator } = await import("$lib/services");

      await serverStore.refresh();

      expect(pollingCoordinator.refresh).toHaveBeenCalledWith("servers");
    });
  });

  describe("startPolling", () => {
    it("starts polling via coordinator", async () => {
      const { pollingCoordinator } = await import("$lib/services");

      serverStore.startPolling();

      expect(pollingCoordinator.start).toHaveBeenCalledWith("servers");
    });

    it("only starts once even if called multiple times", async () => {
      const { pollingCoordinator } = await import("$lib/services");

      serverStore.startPolling();
      serverStore.startPolling();
      serverStore.startPolling();

      expect(pollingCoordinator.start).toHaveBeenCalledTimes(1);
    });
  });

  describe("stopPolling", () => {
    it("stops polling via coordinator", async () => {
      const { pollingCoordinator } = await import("$lib/services");

      serverStore.startPolling();
      serverStore.stopPolling();

      expect(pollingCoordinator.stop).toHaveBeenCalledWith("servers");
    });

    it("allows startPolling to be called again after stop", async () => {
      const { pollingCoordinator } = await import("$lib/services");

      serverStore.startPolling();
      serverStore.stopPolling();
      serverStore.startPolling();

      expect(pollingCoordinator.start).toHaveBeenCalledTimes(2);
    });
  });
});
