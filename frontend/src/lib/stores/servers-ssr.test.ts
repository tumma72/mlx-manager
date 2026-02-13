/**
 * SSR-specific tests for ServerStore
 * These tests run in an environment without window object
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

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

describe("ServerStore SSR", () => {
  let originalWindow: typeof globalThis.window;

  beforeEach(() => {
    // Save original window
    originalWindow = globalThis.window;
    // Delete window to simulate SSR environment
    // @ts-expect-error - deleting window for SSR simulation
    delete globalThis.window;
    vi.resetModules();
  });

  afterEach(() => {
    // Restore window
    globalThis.window = originalWindow;
  });

  it("initializes without window object (SSR fallback)", async () => {
    // Import store in SSR environment
    const { serverStore } = await import("./servers.svelte");

    // Should not throw and should initialize with empty state
    expect(serverStore.servers).toEqual([]);
    expect(serverStore.loading).toBe(false);
    expect(serverStore.error).toBeNull();
  });
});
