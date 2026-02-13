import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import type { ExecutionProfile } from "$lib/api/types";

// Mock the API module before importing the store
vi.mock("$api", () => ({
  profiles: {
    list: vi.fn(),
    create: vi.fn(),
    update: vi.fn(),
    delete: vi.fn(),
    duplicate: vi.fn(),
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

// Helper to create a mock profile
function createMockProfile(
  overrides: Partial<ExecutionProfile> = {},
): ExecutionProfile {
  return {
    id: 1,
    name: "Test Profile",
    description: "A test profile",
    model_id: 1,
    model_repo_id: "mlx-community/test-model",
    model_type: "lm",
    profile_type: "inference",
    auto_start: false,
    launchd_installed: false,
    inference: { temperature: 0.7, max_tokens: 4096, top_p: 1.0 },
    context: { context_length: 4096, system_prompt: null, enable_tool_injection: false },
    audio: null,
    created_at: "2024-01-01T00:00:00",
    updated_at: "2024-01-01T00:00:00",
    ...overrides,
  };
}

describe("ProfileStore", () => {
  let profileStore: Awaited<typeof import("./profiles.svelte")>["profileStore"];
  let mockProfilesApi: {
    list: ReturnType<typeof vi.fn>;
    create: ReturnType<typeof vi.fn>;
    update: ReturnType<typeof vi.fn>;
    delete: ReturnType<typeof vi.fn>;
    duplicate: ReturnType<typeof vi.fn>;
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
    mockProfilesApi = {
      list: vi.fn().mockResolvedValue([]),
      create: vi.fn(),
      update: vi.fn(),
      delete: vi.fn(),
      duplicate: vi.fn(),
    };

    mockPollingCoordinator = {
      register: vi.fn(),
      start: vi.fn(),
      stop: vi.fn(),
      refresh: vi.fn().mockResolvedValue(undefined),
    };

    vi.doMock("$api", () => ({
      profiles: mockProfilesApi,
    }));

    vi.doMock("$lib/services", () => ({
      pollingCoordinator: mockPollingCoordinator,
    }));

    // Import fresh store
    const module = await import("./profiles.svelte");
    profileStore = module.profileStore;
  });

  afterEach(() => {
    vi.resetModules();
  });

  describe("initialization", () => {
    it("registers with polling coordinator on creation", () => {
      expect(mockPollingCoordinator.register).toHaveBeenCalledWith(
        "profiles",
        expect.objectContaining({
          interval: 10000,
          minInterval: 1000,
        }),
      );
    });

    it("starts with empty profiles array", () => {
      expect(profileStore.profiles).toEqual([]);
    });

    it("starts with loading false", () => {
      expect(profileStore.loading).toBe(false);
    });

    it("starts with no error", () => {
      expect(profileStore.error).toBeNull();
    });
  });

  describe("refresh", () => {
    it("delegates to polling coordinator", async () => {
      await profileStore.refresh();

      expect(mockPollingCoordinator.refresh).toHaveBeenCalledWith("profiles");
    });
  });

  describe("startPolling", () => {
    it("starts polling via coordinator", () => {
      profileStore.startPolling();

      expect(mockPollingCoordinator.start).toHaveBeenCalledWith("profiles");
    });

    it("only starts polling once", () => {
      profileStore.startPolling();
      profileStore.startPolling();
      profileStore.startPolling();

      expect(mockPollingCoordinator.start).toHaveBeenCalledTimes(1);
    });
  });

  describe("stopPolling", () => {
    it("stops polling via coordinator", () => {
      profileStore.startPolling();
      profileStore.stopPolling();

      expect(mockPollingCoordinator.stop).toHaveBeenCalledWith("profiles");
    });

    it("allows starting polling again after stop", () => {
      profileStore.startPolling();
      profileStore.stopPolling();
      profileStore.startPolling();

      expect(mockPollingCoordinator.start).toHaveBeenCalledTimes(2);
    });
  });

  describe("create", () => {
    it("calls API and refreshes", async () => {
      const newProfile = createMockProfile({ id: 2, name: "New Profile" });
      mockProfilesApi.create.mockResolvedValue(newProfile);

      const result = await profileStore.create({
        name: "New Profile",
        model_id: 1,
      });

      expect(mockProfilesApi.create).toHaveBeenCalledWith({
        name: "New Profile",
        model_id: 1,
      });
      expect(mockPollingCoordinator.refresh).toHaveBeenCalledWith("profiles");
      expect(result).toEqual(newProfile);
    });
  });

  describe("update", () => {
    it("calls API and refreshes", async () => {
      const updatedProfile = createMockProfile({ name: "Updated Name" });
      mockProfilesApi.update.mockResolvedValue(updatedProfile);

      const result = await profileStore.update(1, { name: "Updated Name" });

      expect(mockProfilesApi.update).toHaveBeenCalledWith(1, {
        name: "Updated Name",
      });
      expect(mockPollingCoordinator.refresh).toHaveBeenCalledWith("profiles");
      expect(result).toEqual(updatedProfile);
    });
  });

  describe("delete", () => {
    it("calls API and refreshes", async () => {
      mockProfilesApi.delete.mockResolvedValue(undefined);

      await profileStore.delete(1);

      expect(mockProfilesApi.delete).toHaveBeenCalledWith(1);
      expect(mockPollingCoordinator.refresh).toHaveBeenCalledWith("profiles");
    });
  });

  describe("duplicate", () => {
    it("calls API and refreshes", async () => {
      const duplicatedProfile = createMockProfile({
        id: 2,
        name: "Profile Copy",
      });
      mockProfilesApi.duplicate.mockResolvedValue(duplicatedProfile);

      const result = await profileStore.duplicate(1, "Profile Copy");

      expect(mockProfilesApi.duplicate).toHaveBeenCalledWith(1, "Profile Copy");
      expect(mockPollingCoordinator.refresh).toHaveBeenCalledWith("profiles");
      expect(result).toEqual(duplicatedProfile);
    });
  });

  describe("getProfile", () => {
    it("returns profile by id", async () => {
      const profile1 = createMockProfile({ id: 1, name: "First" });
      const profile2 = createMockProfile({ id: 2, name: "Second" });
      profileStore.profiles.push(profile1, profile2);

      const found = profileStore.getProfile(2);

      expect(found).toEqual(profile2);
    });

    it("returns undefined for non-existent id", () => {
      const profile = createMockProfile({ id: 1 });
      profileStore.profiles.push(profile);

      const found = profileStore.getProfile(999);

      expect(found).toBeUndefined();
    });
  });

  describe("doRefresh (internal)", () => {
    it("fetches profiles and updates state", async () => {
      const profiles = [
        createMockProfile({ id: 1, name: "Profile 1" }),
        createMockProfile({ id: 2, name: "Profile 2" }),
      ];
      mockProfilesApi.list.mockResolvedValue(profiles);

      // Get the registered refresh function
      const registerCall = mockPollingCoordinator.register.mock.calls[0];
      const refreshFn = registerCall[1].refreshFn;

      // Call it directly
      await refreshFn();

      expect(mockProfilesApi.list).toHaveBeenCalled();
      expect(profileStore.profiles).toHaveLength(2);
      expect(profileStore.profiles[0].name).toBe("Profile 1");
    });

    it("sets loading true on initial load", async () => {
      mockProfilesApi.list.mockResolvedValue([]);

      const registerCall = mockPollingCoordinator.register.mock.calls[0];
      const refreshFn = registerCall[1].refreshFn;

      let loadingDuringFetch = false;
      mockProfilesApi.list.mockImplementation(async () => {
        loadingDuringFetch = profileStore.loading;
        return [];
      });

      await refreshFn();

      expect(loadingDuringFetch).toBe(true);
      expect(profileStore.loading).toBe(false);
    });

    it("sets error on fetch failure", async () => {
      mockProfilesApi.list.mockRejectedValue(new Error("Network error"));

      const registerCall = mockPollingCoordinator.register.mock.calls[0];
      const refreshFn = registerCall[1].refreshFn;

      await refreshFn();

      expect(profileStore.error).toBe("Network error");
    });

    it("sets generic error for non-Error exceptions", async () => {
      mockProfilesApi.list.mockRejectedValue("string error");

      const registerCall = mockPollingCoordinator.register.mock.calls[0];
      const refreshFn = registerCall[1].refreshFn;

      await refreshFn();

      expect(profileStore.error).toBe("Failed to fetch profiles");
    });

    it("clears error on successful refresh", async () => {
      // First fail
      mockProfilesApi.list.mockRejectedValue(new Error("Error"));
      const registerCall = mockPollingCoordinator.register.mock.calls[0];
      const refreshFn = registerCall[1].refreshFn;
      await refreshFn();
      expect(profileStore.error).toBe("Error");

      // Then succeed
      mockProfilesApi.list.mockResolvedValue([]);
      await refreshFn();
      expect(profileStore.error).toBeNull();
    });

    it("does not toggle loading on subsequent polls", async () => {
      mockProfilesApi.list.mockResolvedValue([]);

      const registerCall = mockPollingCoordinator.register.mock.calls[0];
      const refreshFn = registerCall[1].refreshFn;

      // First call (initial load)
      await refreshFn();

      // Second call (background poll)
      let loadingDuringPoll = false;
      mockProfilesApi.list.mockImplementation(async () => {
        loadingDuringPoll = profileStore.loading;
        return [];
      });

      await refreshFn();

      // Loading should NOT be true during background poll
      expect(loadingDuringPoll).toBe(false);
    });

    it("reconciles profiles in place when data matches", async () => {
      const profile = createMockProfile({ id: 1, name: "Test" });
      mockProfilesApi.list.mockResolvedValue([profile]);

      const registerCall = mockPollingCoordinator.register.mock.calls[0];
      const refreshFn = registerCall[1].refreshFn;

      // Initial load
      await refreshFn();
      const originalProfile = profileStore.profiles[0];

      // Same data again
      mockProfilesApi.list.mockResolvedValue([
        createMockProfile({ id: 1, name: "Test" }),
      ]);
      await refreshFn();

      // Should be the same object (reconciled in place)
      expect(profileStore.profiles[0]).toBe(originalProfile);
    });

    it("updates profile properties in-place when name changes", async () => {
      mockProfilesApi.list.mockResolvedValue([
        createMockProfile({ id: 1, name: "Original" }),
      ]);

      const registerCall = mockPollingCoordinator.register.mock.calls[0];
      const refreshFn = registerCall[1].refreshFn;

      await refreshFn();
      const originalProfile = profileStore.profiles[0];

      // Name changed
      mockProfilesApi.list.mockResolvedValue([
        createMockProfile({ id: 1, name: "Changed" }),
      ]);
      await refreshFn();

      // Object is updated in-place (same reference, updated properties)
      expect(profileStore.profiles[0]).toBe(originalProfile);
      expect(profileStore.profiles[0].name).toBe("Changed");
    });

    it("updates profile when description changes", async () => {
      mockProfilesApi.list.mockResolvedValue([
        createMockProfile({ id: 1, description: "Original" }),
      ]);

      const registerCall = mockPollingCoordinator.register.mock.calls[0];
      const refreshFn = registerCall[1].refreshFn;

      await refreshFn();

      mockProfilesApi.list.mockResolvedValue([
        createMockProfile({ id: 1, description: "Changed" }),
      ]);
      await refreshFn();

      expect(profileStore.profiles[0].description).toBe("Changed");
    });

    it("updates profile when temperature changes", async () => {
      mockProfilesApi.list.mockResolvedValue([
        createMockProfile({ id: 1, inference: { temperature: 0.7, max_tokens: 4096, top_p: 1.0 } }),
      ]);

      const registerCall = mockPollingCoordinator.register.mock.calls[0];
      const refreshFn = registerCall[1].refreshFn;

      await refreshFn();

      mockProfilesApi.list.mockResolvedValue([
        createMockProfile({ id: 1, inference: { temperature: 0.5, max_tokens: 4096, top_p: 1.0 } }),
      ]);
      await refreshFn();

      expect(profileStore.profiles[0].inference?.temperature).toBe(0.5);
    });

    it("updates profile when model_type changes", async () => {
      mockProfilesApi.list.mockResolvedValue([
        createMockProfile({ id: 1, model_type: "lm" }),
      ]);

      const registerCall = mockPollingCoordinator.register.mock.calls[0];
      const refreshFn = registerCall[1].refreshFn;

      await refreshFn();

      mockProfilesApi.list.mockResolvedValue([
        createMockProfile({ id: 1, model_type: "multimodal" }),
      ]);
      await refreshFn();

      expect(profileStore.profiles[0].model_type).toBe("multimodal");
    });

    it("updates profile when auto_start changes", async () => {
      mockProfilesApi.list.mockResolvedValue([
        createMockProfile({ id: 1, auto_start: false }),
      ]);

      const registerCall = mockPollingCoordinator.register.mock.calls[0];
      const refreshFn = registerCall[1].refreshFn;

      await refreshFn();

      mockProfilesApi.list.mockResolvedValue([
        createMockProfile({ id: 1, auto_start: true }),
      ]);
      await refreshFn();

      expect(profileStore.profiles[0].auto_start).toBe(true);
    });

    it("updates profile when launchd_installed changes", async () => {
      mockProfilesApi.list.mockResolvedValue([
        createMockProfile({ id: 1, launchd_installed: false }),
      ]);

      const registerCall = mockPollingCoordinator.register.mock.calls[0];
      const refreshFn = registerCall[1].refreshFn;

      await refreshFn();

      mockProfilesApi.list.mockResolvedValue([
        createMockProfile({ id: 1, launchd_installed: true }),
      ]);
      await refreshFn();

      expect(profileStore.profiles[0].launchd_installed).toBe(true);
    });
  });
});
