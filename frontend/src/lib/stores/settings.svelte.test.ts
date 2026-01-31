import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import type { CloudCredential } from "$lib/api/types";

// Mock the API module before importing the store
vi.mock("$lib/api/client", () => ({
  settings: {
    listProviders: vi.fn(),
  },
}));

// Helper to create mock cloud credential
function createMockCredential(
  overrides: Partial<CloudCredential> = {},
): CloudCredential {
  return {
    id: 1,
    backend_type: "openai",
    base_url: null,
    created_at: "2024-01-01T00:00:00Z",
    ...overrides,
  };
}

describe("SettingsStore", () => {
  let settingsStore: Awaited<
    typeof import("./settings.svelte")
  >["settingsStore"];
  let mockSettingsApi: {
    listProviders: ReturnType<typeof vi.fn>;
  };

  beforeEach(async () => {
    vi.clearAllMocks();

    // Reset modules to get fresh store instance
    vi.resetModules();

    // Re-mock after reset
    mockSettingsApi = {
      listProviders: vi.fn().mockResolvedValue([]),
    };

    vi.doMock("$lib/api/client", () => ({
      settings: mockSettingsApi,
    }));

    // Import fresh store
    const module = await import("./settings.svelte");
    settingsStore = module.settingsStore;
  });

  afterEach(() => {
    vi.resetModules();
  });

  describe("initialization", () => {
    it("starts with empty providers array", () => {
      expect(settingsStore.providers).toEqual([]);
    });

    it("starts with loading false", () => {
      expect(settingsStore.loading).toBe(false);
    });

    it("starts with no error", () => {
      expect(settingsStore.error).toBeNull();
    });

    it("starts with empty configured providers set", () => {
      expect(settingsStore.configuredProviders.size).toBe(0);
    });
  });

  describe("loadProviders", () => {
    it("fetches providers from API", async () => {
      const mockProviders = [
        createMockCredential({ id: 1, backend_type: "openai" }),
        createMockCredential({ id: 2, backend_type: "anthropic" }),
      ];
      mockSettingsApi.listProviders.mockResolvedValue(mockProviders);

      await settingsStore.loadProviders();

      expect(mockSettingsApi.listProviders).toHaveBeenCalled();
      expect(settingsStore.providers).toEqual(mockProviders);
    });

    it("sets loading true during fetch", async () => {
      let loadingDuringFetch = false;
      mockSettingsApi.listProviders.mockImplementation(async () => {
        loadingDuringFetch = settingsStore.loading;
        return [];
      });

      await settingsStore.loadProviders();

      expect(loadingDuringFetch).toBe(true);
    });

    it("sets loading false after fetch completes", async () => {
      mockSettingsApi.listProviders.mockResolvedValue([]);

      await settingsStore.loadProviders();

      expect(settingsStore.loading).toBe(false);
    });

    it("clears error before fetch", async () => {
      mockSettingsApi.listProviders.mockResolvedValue([]);

      // Set an error first
      await settingsStore.loadProviders();
      mockSettingsApi.listProviders.mockRejectedValue(new Error("Network error"));
      await settingsStore.loadProviders();
      expect(settingsStore.error).toBe("Network error");

      // Now fetch successfully - error should be cleared
      mockSettingsApi.listProviders.mockResolvedValue([]);
      await settingsStore.loadProviders();

      expect(settingsStore.error).toBeNull();
    });

    it("sets error on fetch failure", async () => {
      mockSettingsApi.listProviders.mockRejectedValue(
        new Error("Network error"),
      );

      await settingsStore.loadProviders();

      expect(settingsStore.error).toBe("Network error");
    });

    it("sets loading false even on error", async () => {
      mockSettingsApi.listProviders.mockRejectedValue(
        new Error("Network error"),
      );

      await settingsStore.loadProviders();

      expect(settingsStore.loading).toBe(false);
    });

    it("sets generic error message for non-Error exceptions", async () => {
      mockSettingsApi.listProviders.mockRejectedValue("string error");

      await settingsStore.loadProviders();

      expect(settingsStore.error).toBe("Failed to load providers");
    });
  });

  describe("setProviders", () => {
    it("updates providers state", () => {
      const providers = [
        createMockCredential({ id: 1, backend_type: "openai" }),
        createMockCredential({ id: 2, backend_type: "anthropic" }),
      ];

      settingsStore.setProviders(providers);

      expect(settingsStore.providers).toEqual(providers);
    });

    it("replaces existing providers", () => {
      const initial = [createMockCredential({ id: 1, backend_type: "openai" })];
      const updated = [
        createMockCredential({ id: 2, backend_type: "anthropic" }),
      ];

      settingsStore.setProviders(initial);
      expect(settingsStore.providers).toEqual(initial);

      settingsStore.setProviders(updated);
      expect(settingsStore.providers).toEqual(updated);
    });

    it("accepts empty array", () => {
      settingsStore.setProviders([
        createMockCredential({ id: 1, backend_type: "openai" }),
      ]);
      settingsStore.setProviders([]);

      expect(settingsStore.providers).toEqual([]);
    });
  });

  describe("addProvider", () => {
    it("adds new provider to empty list", () => {
      const provider = createMockCredential({ id: 1, backend_type: "openai" });

      settingsStore.addProvider(provider);

      expect(settingsStore.providers).toEqual([provider]);
    });

    it("adds provider to existing list", () => {
      const existing = createMockCredential({ id: 1, backend_type: "openai" });
      const newProvider = createMockCredential({
        id: 2,
        backend_type: "anthropic",
      });

      settingsStore.setProviders([existing]);
      settingsStore.addProvider(newProvider);

      expect(settingsStore.providers).toEqual([existing, newProvider]);
    });

    it("replaces existing provider of same type", () => {
      const original = createMockCredential({
        id: 1,
        backend_type: "openai",
        created_at: "2024-01-01T00:00:00Z",
      });
      const replacement = createMockCredential({
        id: 2,
        backend_type: "openai",
        created_at: "2024-01-02T00:00:00Z",
      });

      settingsStore.setProviders([original]);
      settingsStore.addProvider(replacement);

      expect(settingsStore.providers).toHaveLength(1);
      expect(settingsStore.providers[0]).toEqual(replacement);
    });

    it("removes only matching backend type when replacing", () => {
      const openai = createMockCredential({ id: 1, backend_type: "openai" });
      const anthropic = createMockCredential({
        id: 2,
        backend_type: "anthropic",
      });
      const newOpenai = createMockCredential({
        id: 3,
        backend_type: "openai",
      });

      settingsStore.setProviders([openai, anthropic]);
      settingsStore.addProvider(newOpenai);

      expect(settingsStore.providers).toHaveLength(2);
      expect(settingsStore.providers).toContainEqual(anthropic);
      expect(settingsStore.providers).toContainEqual(newOpenai);
      expect(settingsStore.providers).not.toContainEqual(openai);
    });
  });

  describe("removeProvider", () => {
    it("removes provider by backend type", () => {
      const providers = [
        createMockCredential({ id: 1, backend_type: "openai" }),
        createMockCredential({ id: 2, backend_type: "anthropic" }),
      ];

      settingsStore.setProviders(providers);
      settingsStore.removeProvider("openai");

      expect(settingsStore.providers).toHaveLength(1);
      expect(settingsStore.providers[0].backend_type).toBe("anthropic");
    });

    it("does nothing if provider type not found", () => {
      const providers = [
        createMockCredential({ id: 1, backend_type: "openai" }),
      ];

      settingsStore.setProviders(providers);
      settingsStore.removeProvider("anthropic");

      expect(settingsStore.providers).toEqual(providers);
    });

    it("handles empty providers list", () => {
      settingsStore.setProviders([]);
      settingsStore.removeProvider("openai");

      expect(settingsStore.providers).toEqual([]);
    });

    it("removes all providers of specified type", () => {
      // Edge case: multiple providers of same type (shouldn't happen but test handles it)
      const providers = [
        createMockCredential({ id: 1, backend_type: "openai" }),
        createMockCredential({ id: 2, backend_type: "openai" }),
        createMockCredential({ id: 3, backend_type: "anthropic" }),
      ];

      settingsStore.setProviders(providers);
      settingsStore.removeProvider("openai");

      expect(settingsStore.providers).toHaveLength(1);
      expect(settingsStore.providers[0].backend_type).toBe("anthropic");
    });
  });

  describe("reset", () => {
    it("clears providers", () => {
      settingsStore.setProviders([
        createMockCredential({ id: 1, backend_type: "openai" }),
      ]);

      settingsStore.reset();

      expect(settingsStore.providers).toEqual([]);
    });

    it("resets loading to false", async () => {
      mockSettingsApi.listProviders.mockImplementation(
        () =>
          new Promise(() => {
            /* never resolves */
          }),
      );

      void settingsStore.loadProviders();
      // Wait a tick for loading to be set to true
      await new Promise((resolve) => setTimeout(resolve, 0));

      settingsStore.reset();

      expect(settingsStore.loading).toBe(false);

      // Clean up
      vi.clearAllMocks();
    });

    it("clears error", async () => {
      mockSettingsApi.listProviders.mockRejectedValue(
        new Error("Network error"),
      );
      await settingsStore.loadProviders();

      settingsStore.reset();

      expect(settingsStore.error).toBeNull();
    });

    it("resets all state properties at once", async () => {
      mockSettingsApi.listProviders.mockRejectedValue(
        new Error("Network error"),
      );
      await settingsStore.loadProviders();
      settingsStore.setProviders([
        createMockCredential({ id: 1, backend_type: "openai" }),
      ]);

      settingsStore.reset();

      expect(settingsStore.providers).toEqual([]);
      expect(settingsStore.loading).toBe(false);
      expect(settingsStore.error).toBeNull();
    });
  });

  describe("configuredProviders (derived)", () => {
    it("returns empty set when no providers", () => {
      expect(settingsStore.configuredProviders.size).toBe(0);
    });

    it("returns set of configured backend types", () => {
      const providers = [
        createMockCredential({ id: 1, backend_type: "openai" }),
        createMockCredential({ id: 2, backend_type: "anthropic" }),
      ];

      settingsStore.setProviders(providers);

      expect(settingsStore.configuredProviders.size).toBe(2);
      expect(settingsStore.configuredProviders.has("openai")).toBe(true);
      expect(settingsStore.configuredProviders.has("anthropic")).toBe(true);
    });

    it("deduplicates backend types", () => {
      // Multiple credentials of same type (edge case)
      const providers = [
        createMockCredential({ id: 1, backend_type: "openai" }),
        createMockCredential({ id: 2, backend_type: "openai" }),
      ];

      settingsStore.setProviders(providers);

      expect(settingsStore.configuredProviders.size).toBe(1);
      expect(settingsStore.configuredProviders.has("openai")).toBe(true);
    });

    it("updates when providers change", () => {
      settingsStore.setProviders([
        createMockCredential({ id: 1, backend_type: "openai" }),
      ]);
      expect(settingsStore.configuredProviders.has("openai")).toBe(true);
      expect(settingsStore.configuredProviders.has("anthropic")).toBe(false);

      settingsStore.addProvider(
        createMockCredential({ id: 2, backend_type: "anthropic" }),
      );

      expect(settingsStore.configuredProviders.has("openai")).toBe(true);
      expect(settingsStore.configuredProviders.has("anthropic")).toBe(true);
    });
  });

  describe("isProviderConfigured", () => {
    it("returns false for unconfigured provider", () => {
      expect(settingsStore.isProviderConfigured("openai")).toBe(false);
    });

    it("returns true for configured provider", () => {
      settingsStore.setProviders([
        createMockCredential({ id: 1, backend_type: "openai" }),
      ]);

      expect(settingsStore.isProviderConfigured("openai")).toBe(true);
    });

    it("returns false after provider is removed", () => {
      settingsStore.setProviders([
        createMockCredential({ id: 1, backend_type: "openai" }),
      ]);
      expect(settingsStore.isProviderConfigured("openai")).toBe(true);

      settingsStore.removeProvider("openai");

      expect(settingsStore.isProviderConfigured("openai")).toBe(false);
    });

    it("checks all backend types correctly", () => {
      const providers = [
        createMockCredential({ id: 1, backend_type: "openai" }),
        createMockCredential({ id: 2, backend_type: "anthropic" }),
      ];
      settingsStore.setProviders(providers);

      expect(settingsStore.isProviderConfigured("openai")).toBe(true);
      expect(settingsStore.isProviderConfigured("anthropic")).toBe(true);
      expect(settingsStore.isProviderConfigured("local")).toBe(false);
    });
  });

  describe("hasAnyCloudProvider", () => {
    it("returns false when no providers configured", () => {
      expect(settingsStore.hasAnyCloudProvider()).toBe(false);
    });

    it("returns false when only local provider configured", () => {
      settingsStore.setProviders([
        createMockCredential({ id: 1, backend_type: "local" }),
      ]);

      expect(settingsStore.hasAnyCloudProvider()).toBe(false);
    });

    it("returns true when openai provider configured", () => {
      settingsStore.setProviders([
        createMockCredential({ id: 1, backend_type: "openai" }),
      ]);

      expect(settingsStore.hasAnyCloudProvider()).toBe(true);
    });

    it("returns true when anthropic provider configured", () => {
      settingsStore.setProviders([
        createMockCredential({ id: 1, backend_type: "anthropic" }),
      ]);

      expect(settingsStore.hasAnyCloudProvider()).toBe(true);
    });

    it("returns true when mix of local and cloud providers", () => {
      settingsStore.setProviders([
        createMockCredential({ id: 1, backend_type: "local" }),
        createMockCredential({ id: 2, backend_type: "openai" }),
      ]);

      expect(settingsStore.hasAnyCloudProvider()).toBe(true);
    });

    it("returns true when multiple cloud providers", () => {
      settingsStore.setProviders([
        createMockCredential({ id: 1, backend_type: "openai" }),
        createMockCredential({ id: 2, backend_type: "anthropic" }),
      ]);

      expect(settingsStore.hasAnyCloudProvider()).toBe(true);
    });

    it("returns false after all cloud providers removed", () => {
      settingsStore.setProviders([
        createMockCredential({ id: 1, backend_type: "openai" }),
        createMockCredential({ id: 2, backend_type: "local" }),
      ]);
      expect(settingsStore.hasAnyCloudProvider()).toBe(true);

      settingsStore.removeProvider("openai");

      expect(settingsStore.hasAnyCloudProvider()).toBe(false);
    });
  });

  describe("integration scenarios", () => {
    it("handles complete workflow: load -> add -> remove -> reset", async () => {
      // Load initial providers
      const initial = [createMockCredential({ id: 1, backend_type: "openai" })];
      mockSettingsApi.listProviders.mockResolvedValue(initial);
      await settingsStore.loadProviders();
      expect(settingsStore.providers).toEqual(initial);
      expect(settingsStore.isProviderConfigured("openai")).toBe(true);

      // Add new provider
      const newProvider = createMockCredential({
        id: 2,
        backend_type: "anthropic",
      });
      settingsStore.addProvider(newProvider);
      expect(settingsStore.providers).toHaveLength(2);
      expect(settingsStore.hasAnyCloudProvider()).toBe(true);

      // Remove one provider
      settingsStore.removeProvider("openai");
      expect(settingsStore.providers).toHaveLength(1);
      expect(settingsStore.isProviderConfigured("openai")).toBe(false);
      expect(settingsStore.isProviderConfigured("anthropic")).toBe(true);

      // Reset everything
      settingsStore.reset();
      expect(settingsStore.providers).toEqual([]);
      expect(settingsStore.hasAnyCloudProvider()).toBe(false);
    });

    it("handles error recovery workflow", async () => {
      // Initial error
      mockSettingsApi.listProviders.mockRejectedValue(
        new Error("Network error"),
      );
      await settingsStore.loadProviders();
      expect(settingsStore.error).toBe("Network error");
      expect(settingsStore.providers).toEqual([]);

      // Successful retry
      const providers = [
        createMockCredential({ id: 1, backend_type: "openai" }),
      ];
      mockSettingsApi.listProviders.mockResolvedValue(providers);
      await settingsStore.loadProviders();
      expect(settingsStore.error).toBeNull();
      expect(settingsStore.providers).toEqual(providers);
    });

    it("handles rapid successive updates", () => {
      const provider1 = createMockCredential({ id: 1, backend_type: "openai" });
      const provider2 = createMockCredential({
        id: 2,
        backend_type: "anthropic",
      });
      const provider3 = createMockCredential({ id: 3, backend_type: "local" });

      // Rapid adds
      settingsStore.addProvider(provider1);
      settingsStore.addProvider(provider2);
      settingsStore.addProvider(provider3);

      expect(settingsStore.providers).toHaveLength(3);
      expect(settingsStore.configuredProviders.size).toBe(3);

      // Rapid removes
      settingsStore.removeProvider("openai");
      settingsStore.removeProvider("anthropic");

      expect(settingsStore.providers).toHaveLength(1);
      expect(settingsStore.providers[0].backend_type).toBe("local");
    });

    it("handles provider replacement correctly", () => {
      // Initial setup
      const original = createMockCredential({
        id: 1,
        backend_type: "openai",
        base_url: null,
        created_at: "2024-01-01T00:00:00Z",
      });
      settingsStore.addProvider(original);

      // Replace with updated credentials
      const updated = createMockCredential({
        id: 2,
        backend_type: "openai",
        base_url: "https://custom.openai.com",
        created_at: "2024-01-02T00:00:00Z",
      });
      settingsStore.addProvider(updated);

      // Should only have one openai provider with updated details
      expect(settingsStore.providers).toHaveLength(1);
      expect(settingsStore.providers[0].id).toBe(2);
      expect(settingsStore.providers[0].base_url).toBe(
        "https://custom.openai.com",
      );
    });
  });
});
