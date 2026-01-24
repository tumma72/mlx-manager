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
        start: vi.fn().mockResolvedValue({ pid: 12345, port: 10240 }),
        stop: vi.fn().mockResolvedValue({ stopped: true }),
        restart: vi.fn().mockResolvedValue({ pid: 12345 }),
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
      vi.mocked(serversApi.start).mockResolvedValue({ pid: 12345, port: 10240 });

      await serverStore.start(42);

      expect(serverStore.isStarting(42)).toBe(true);
    });

    it("clears any existing failure", async () => {
      const { servers: serversApi } = await import("$api");
      vi.mocked(serversApi.start).mockResolvedValue({ pid: 12345, port: 10240 });

      // Set up a failure first
      serverStore.markStartupFailed(42, "Previous error");
      expect(serverStore.isFailed(42)).toBe(true);

      await serverStore.start(42);

      expect(serverStore.isFailed(42)).toBe(false);
    });

    it("calls API to start server", async () => {
      const { servers: serversApi } = await import("$api");
      vi.mocked(serversApi.start).mockResolvedValue({ pid: 12345, port: 10240 });

      await serverStore.start(42);

      expect(serversApi.start).toHaveBeenCalledWith(42);
    });
  });

  describe("stop", () => {
    it("removes profile from startingProfiles", async () => {
      const { servers: serversApi } = await import("$api");
      vi.mocked(serversApi.start).mockResolvedValue({ pid: 12345, port: 10240 });
      vi.mocked(serversApi.stop).mockResolvedValue({ stopped: true });

      await serverStore.start(42);
      expect(serverStore.isStarting(42)).toBe(true);

      await serverStore.stop(42);
      expect(serverStore.isStarting(42)).toBe(false);
    });

    it("clears any existing failure", async () => {
      const { servers: serversApi } = await import("$api");
      vi.mocked(serversApi.stop).mockResolvedValue({ stopped: true });

      serverStore.markStartupFailed(42, "Previous error");
      expect(serverStore.isFailed(42)).toBe(true);

      await serverStore.stop(42);

      expect(serverStore.isFailed(42)).toBe(false);
    });

    it("calls API to stop server", async () => {
      const { servers: serversApi } = await import("$api");
      vi.mocked(serversApi.stop).mockResolvedValue({ stopped: true });

      await serverStore.stop(42);

      expect(serversApi.stop).toHaveBeenCalledWith(42, false);
    });

    it("calls API with force flag when specified", async () => {
      const { servers: serversApi } = await import("$api");
      vi.mocked(serversApi.stop).mockResolvedValue({ stopped: true });

      await serverStore.stop(42, true);

      expect(serversApi.stop).toHaveBeenCalledWith(42, true);
    });

    it("triggers refresh after stop", async () => {
      const { servers: serversApi } = await import("$api");
      const { pollingCoordinator } = await import("$lib/services");
      vi.mocked(serversApi.stop).mockResolvedValue({ stopped: true });

      await serverStore.stop(42);

      expect(pollingCoordinator.refresh).toHaveBeenCalledWith("servers");
    });
  });

  describe("restart", () => {
    it("adds profile to startingProfiles", async () => {
      const { servers: serversApi } = await import("$api");
      vi.mocked(serversApi.restart).mockResolvedValue({ pid: 12345 });

      await serverStore.restart(42);

      expect(serverStore.isStarting(42)).toBe(true);
    });

    it("clears any existing failure", async () => {
      const { servers: serversApi } = await import("$api");
      vi.mocked(serversApi.restart).mockResolvedValue({ pid: 12345 });

      serverStore.markStartupFailed(42, "Previous error");
      expect(serverStore.isFailed(42)).toBe(true);

      await serverStore.restart(42);

      expect(serverStore.isFailed(42)).toBe(false);
    });

    it("calls API to restart server", async () => {
      const { servers: serversApi } = await import("$api");
      vi.mocked(serversApi.restart).mockResolvedValue({ pid: 12345 });

      await serverStore.restart(42);

      expect(serversApi.restart).toHaveBeenCalledWith(42);
    });
  });

  describe("markStartupSuccess", () => {
    it("removes profile from startingProfiles", async () => {
      const { servers: serversApi } = await import("$api");
      vi.mocked(serversApi.start).mockResolvedValue({ pid: 12345, port: 10240 });

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

    it("triggers refresh when state actually changes", async () => {
      const { servers: serversApi } = await import("$api");
      const { pollingCoordinator } = await import("$lib/services");

      // Setup: put profile in starting state
      vi.mocked(serversApi.start).mockResolvedValue({ pid: 12345, port: 10240 });
      await serverStore.start(42);

      // Clear previous refresh calls
      vi.mocked(pollingCoordinator.refresh).mockClear();

      serverStore.markStartupSuccess(42);

      expect(pollingCoordinator.refresh).toHaveBeenCalledWith("servers");
    });

    it("does not trigger refresh when no state change", async () => {
      const { pollingCoordinator } = await import("$lib/services");

      // Clear previous calls
      vi.mocked(pollingCoordinator.refresh).mockClear();

      // Call without profile being in starting/failed/restarting state
      serverStore.markStartupSuccess(42);

      // Should not trigger refresh (no state change)
      expect(pollingCoordinator.refresh).not.toHaveBeenCalled();
    });
  });

  describe("markStartupFailed", () => {
    it("removes profile from startingProfiles", async () => {
      const { servers: serversApi } = await import("$api");
      vi.mocked(serversApi.start).mockResolvedValue({ pid: 12345, port: 10240 });

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
      vi.mocked(serversApi.start).mockResolvedValue({ pid: 12345, port: 10240 });

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
      vi.mocked(serversApi.start).mockResolvedValue({ pid: 12345, port: 10240 });

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

  describe("isRestarting", () => {
    it("returns false initially", () => {
      expect(serverStore.isRestarting(42)).toBe(false);
    });

    it("returns true during restart operation", async () => {
      const { servers: serversApi } = await import("$api");
      // Make restart hang so we can check the intermediate state
      let resolveRestart: (value: { pid: number }) => void;
      vi.mocked(serversApi.restart).mockImplementation(
        () => new Promise((resolve) => { resolveRestart = resolve; })
      );

      // Start the restart but don't await it
      const restartPromise = serverStore.restart(42);

      // During the restart, the profile should be marked as restarting
      expect(serverStore.isRestarting(42)).toBe(true);

      // Complete the restart
      resolveRestart!({ pid: 12345 });
      await restartPromise;

      // After restart completes, it transitions to starting
      expect(serverStore.isRestarting(42)).toBe(false);
      expect(serverStore.isStarting(42)).toBe(true);
    });
  });

  describe("doRefresh (internal refresh logic)", () => {
    // To test doRefresh, we need to capture the refreshFn that was registered
    // with the polling coordinator and call it directly

    async function getRefreshFn(): Promise<() => Promise<void>> {
      const { pollingCoordinator } = await import("$lib/services");
      const registerCall = vi.mocked(pollingCoordinator.register).mock.calls[0];
      return registerCall[1].refreshFn as () => Promise<void>;
    }

    it("sets loading to true on initial load", async () => {
      const { servers: serversApi } = await import("$api");
      let resolveList: (value: RunningServer[]) => void;
      vi.mocked(serversApi.list).mockImplementation(
        () => new Promise((resolve) => { resolveList = resolve; })
      );

      const refreshFn = await getRefreshFn();
      const refreshPromise = refreshFn();

      // During initial load, loading should be true
      expect(serverStore.loading).toBe(true);

      // Complete the request
      resolveList!([]);
      await refreshPromise;

      // After completion, loading should be false
      expect(serverStore.loading).toBe(false);
    });

    it("does not set loading on subsequent refreshes", async () => {
      const { servers: serversApi } = await import("$api");
      vi.mocked(serversApi.list).mockResolvedValue([]);

      const refreshFn = await getRefreshFn();

      // First refresh (initial load)
      await refreshFn();
      expect(serverStore.loading).toBe(false);

      // Second refresh (background poll) - should not set loading
      let resolveList: (value: RunningServer[]) => void;
      vi.mocked(serversApi.list).mockImplementation(
        () => new Promise((resolve) => { resolveList = resolve; })
      );

      const refreshPromise = refreshFn();

      // During background poll, loading should stay false
      expect(serverStore.loading).toBe(false);

      resolveList!([]);
      await refreshPromise;
    });

    it("updates servers array via reconciliation", async () => {
      const { servers: serversApi } = await import("$api");
      const mockServer = createMockServer({ profile_id: 1 });
      vi.mocked(serversApi.list).mockResolvedValue([mockServer]);

      const refreshFn = await getRefreshFn();
      await refreshFn();

      expect(serverStore.servers).toHaveLength(1);
      expect(serverStore.servers[0].profile_id).toBe(1);
    });

    it("clears error on successful refresh", async () => {
      const { servers: serversApi } = await import("$api");

      // First, cause an error
      vi.mocked(serversApi.list).mockRejectedValueOnce(new Error("Network error"));
      const refreshFn = await getRefreshFn();
      await refreshFn();
      expect(serverStore.error).toBe("Network error");

      // Then refresh successfully
      vi.mocked(serversApi.list).mockResolvedValue([]);
      await refreshFn();
      expect(serverStore.error).toBe(null);
    });

    it("sets error message on fetch failure with Error object", async () => {
      const { servers: serversApi } = await import("$api");
      vi.mocked(serversApi.list).mockRejectedValue(new Error("Network error"));

      const refreshFn = await getRefreshFn();
      await refreshFn();

      expect(serverStore.error).toBe("Network error");
    });

    it("sets default error message on fetch failure with non-Error", async () => {
      const { servers: serversApi } = await import("$api");
      vi.mocked(serversApi.list).mockRejectedValue("string error");

      const refreshFn = await getRefreshFn();
      await refreshFn();

      expect(serverStore.error).toBe("Failed to fetch servers");
    });

    it("sets loading false even on error during initial load", async () => {
      const { servers: serversApi } = await import("$api");
      vi.mocked(serversApi.list).mockRejectedValue(new Error("Network error"));

      const refreshFn = await getRefreshFn();
      await refreshFn();

      expect(serverStore.loading).toBe(false);
    });

    it("reconciles servers using custom equality (small uptime drift ignored)", async () => {
      const { servers: serversApi } = await import("$api");

      // Initial server
      const server1 = createMockServer({ profile_id: 1, uptime_seconds: 100 });
      vi.mocked(serversApi.list).mockResolvedValue([server1]);

      const refreshFn = await getRefreshFn();
      await refreshFn();

      // Store the original array reference
      const originalArray = serverStore.servers;
      const originalServer = originalArray[0];

      // Update with small uptime drift (< 2 seconds) - should be considered equal
      const server2 = createMockServer({ profile_id: 1, uptime_seconds: 101 });
      vi.mocked(serversApi.list).mockResolvedValue([server2]);
      await refreshFn();

      // Array should still be the same reference (reconciled in-place)
      expect(serverStore.servers).toBe(originalArray);
      // Server object should not have been replaced (considered equal)
      expect(serverStore.servers[0]).toBe(originalServer);
    });

    it("reconciles servers using custom equality (large uptime drift triggers update)", async () => {
      const { servers: serversApi } = await import("$api");

      // Initial server
      const server1 = createMockServer({ profile_id: 1, uptime_seconds: 100 });
      vi.mocked(serversApi.list).mockResolvedValue([server1]);

      const refreshFn = await getRefreshFn();
      await refreshFn();

      // Store the original uptime
      const originalUptime = serverStore.servers[0].uptime_seconds;
      expect(originalUptime).toBe(100);

      // Update with large uptime drift (>= 2 seconds) - should trigger update
      const server2 = createMockServer({ profile_id: 1, uptime_seconds: 105 });
      vi.mocked(serversApi.list).mockResolvedValue([server2]);
      await refreshFn();

      // Value should have been updated in-place
      expect(serverStore.servers[0].uptime_seconds).toBe(105);
    });

    it("reconciles servers using custom equality (small memory drift ignored)", async () => {
      const { servers: serversApi } = await import("$api");

      // Initial server
      const server1 = createMockServer({ profile_id: 1, memory_mb: 500 });
      vi.mocked(serversApi.list).mockResolvedValue([server1]);

      const refreshFn = await getRefreshFn();
      await refreshFn();

      const originalServer = serverStore.servers[0];

      // Update with small memory drift (< 1 MB) - should be considered equal
      const server2 = createMockServer({ profile_id: 1, memory_mb: 500.5 });
      vi.mocked(serversApi.list).mockResolvedValue([server2]);
      await refreshFn();

      // Server object should not have been replaced (considered equal)
      expect(serverStore.servers[0]).toBe(originalServer);
    });

    it("reconciles servers using custom equality (large memory drift triggers update)", async () => {
      const { servers: serversApi } = await import("$api");

      // Initial server
      const server1 = createMockServer({ profile_id: 1, memory_mb: 500 });
      vi.mocked(serversApi.list).mockResolvedValue([server1]);

      const refreshFn = await getRefreshFn();
      await refreshFn();

      // Store the original memory
      const originalMemory = serverStore.servers[0].memory_mb;
      expect(originalMemory).toBe(500);

      // Update with large memory drift (>= 1 MB) - should trigger update
      const server2 = createMockServer({ profile_id: 1, memory_mb: 502 });
      vi.mocked(serversApi.list).mockResolvedValue([server2]);
      await refreshFn();

      // Value should have been updated in-place
      expect(serverStore.servers[0].memory_mb).toBe(502);
    });

    it("reconciles servers using custom equality (different profile_id triggers update)", async () => {
      const { servers: serversApi } = await import("$api");

      // Initial server
      const server1 = createMockServer({ profile_id: 1 });
      vi.mocked(serversApi.list).mockResolvedValue([server1]);

      const refreshFn = await getRefreshFn();
      await refreshFn();

      // Replace with different server (new key)
      const server2 = createMockServer({ profile_id: 2 });
      vi.mocked(serversApi.list).mockResolvedValue([server2]);
      await refreshFn();

      expect(serverStore.servers).toHaveLength(1);
      expect(serverStore.servers[0].profile_id).toBe(2);
    });

    it("reconciles servers using custom equality (different profile_name triggers update)", async () => {
      const { servers: serversApi } = await import("$api");

      const server1 = createMockServer({ profile_id: 1, profile_name: "Server A" });
      vi.mocked(serversApi.list).mockResolvedValue([server1]);

      const refreshFn = await getRefreshFn();
      await refreshFn();

      expect(serverStore.servers[0].profile_name).toBe("Server A");

      const server2 = createMockServer({ profile_id: 1, profile_name: "Server B" });
      vi.mocked(serversApi.list).mockResolvedValue([server2]);
      await refreshFn();

      // Value should have been updated in-place
      expect(serverStore.servers[0].profile_name).toBe("Server B");
    });

    it("reconciles servers using custom equality (different pid triggers update)", async () => {
      const { servers: serversApi } = await import("$api");

      const server1 = createMockServer({ profile_id: 1, pid: 1000 });
      vi.mocked(serversApi.list).mockResolvedValue([server1]);

      const refreshFn = await getRefreshFn();
      await refreshFn();

      expect(serverStore.servers[0].pid).toBe(1000);

      const server2 = createMockServer({ profile_id: 1, pid: 2000 });
      vi.mocked(serversApi.list).mockResolvedValue([server2]);
      await refreshFn();

      // Value should have been updated in-place
      expect(serverStore.servers[0].pid).toBe(2000);
    });

    it("reconciles servers using custom equality (different port triggers update)", async () => {
      const { servers: serversApi } = await import("$api");

      const server1 = createMockServer({ profile_id: 1, port: 10240 });
      vi.mocked(serversApi.list).mockResolvedValue([server1]);

      const refreshFn = await getRefreshFn();
      await refreshFn();

      expect(serverStore.servers[0].port).toBe(10240);

      const server2 = createMockServer({ profile_id: 1, port: 10241 });
      vi.mocked(serversApi.list).mockResolvedValue([server2]);
      await refreshFn();

      // Value should have been updated in-place
      expect(serverStore.servers[0].port).toBe(10241);
    });

    it("reconciles servers using custom equality (different health_status triggers update)", async () => {
      const { servers: serversApi } = await import("$api");

      const server1 = createMockServer({ profile_id: 1, health_status: "healthy" });
      vi.mocked(serversApi.list).mockResolvedValue([server1]);

      const refreshFn = await getRefreshFn();
      await refreshFn();

      expect(serverStore.servers[0].health_status).toBe("healthy");

      const server2 = createMockServer({ profile_id: 1, health_status: "unhealthy" });
      vi.mocked(serversApi.list).mockResolvedValue([server2]);
      await refreshFn();

      // Value should have been updated in-place
      expect(serverStore.servers[0].health_status).toBe("unhealthy");
    });
  });

  describe("restart state transitions", () => {
    it("marks profile as restarting then starting", async () => {
      const { servers: serversApi } = await import("$api");
      vi.mocked(serversApi.restart).mockResolvedValue({ pid: 12345 });

      await serverStore.restart(42);

      // After restart completes, profile should be in starting state, not restarting
      expect(serverStore.isRestarting(42)).toBe(false);
      expect(serverStore.isStarting(42)).toBe(true);
    });

    it("clears restarting state on markStartupSuccess", async () => {
      const { servers: serversApi } = await import("$api");
      let resolveRestart: (value: { pid: number }) => void;
      vi.mocked(serversApi.restart).mockImplementation(
        () => new Promise((resolve) => { resolveRestart = resolve; })
      );

      const restartPromise = serverStore.restart(42);
      expect(serverStore.isRestarting(42)).toBe(true);

      resolveRestart!({ pid: 12345 });
      await restartPromise;

      // Now in starting state
      expect(serverStore.isStarting(42)).toBe(true);

      // Mark as successful
      serverStore.markStartupSuccess(42);

      expect(serverStore.isRestarting(42)).toBe(false);
      expect(serverStore.isStarting(42)).toBe(false);
    });

    it("clears restarting state on markStartupFailed", async () => {
      const { servers: serversApi } = await import("$api");
      let resolveRestart: (value: { pid: number }) => void;
      vi.mocked(serversApi.restart).mockImplementation(
        () => new Promise((resolve) => { resolveRestart = resolve; })
      );

      const restartPromise = serverStore.restart(42);
      expect(serverStore.isRestarting(42)).toBe(true);

      resolveRestart!({ pid: 12345 });
      await restartPromise;

      // Simulate failure during startup
      serverStore.markStartupFailed(42, "Startup failed");

      expect(serverStore.isRestarting(42)).toBe(false);
      expect(serverStore.isStarting(42)).toBe(false);
      expect(serverStore.isFailed(42)).toBe(true);
    });
  });
});
