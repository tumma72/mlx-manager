import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import type { User } from "$lib/api/types";

// Helper to create mock user
function createMockUser(overrides: Partial<User> = {}): User {
  return {
    id: 1,
    email: "test@example.com",
    is_admin: false,
    status: "approved",
    ...overrides,
  };
}

describe("AuthStore", () => {
  let authStore: Awaited<typeof import("./auth.svelte")>["authStore"];
  let mockLocalStorage: { [key: string]: string };

  beforeEach(async () => {
    vi.clearAllMocks();

    // Mock localStorage
    mockLocalStorage = {};
    const localStorageMock = {
      getItem: vi.fn((key: string) => mockLocalStorage[key] || null),
      setItem: vi.fn((key: string, value: string) => {
        mockLocalStorage[key] = value;
      }),
      removeItem: vi.fn((key: string) => {
        delete mockLocalStorage[key];
      }),
      clear: vi.fn(() => {
        mockLocalStorage = {};
      }),
    };
    Object.defineProperty(window, "localStorage", {
      value: localStorageMock,
      writable: true,
    });

    // Reset modules to get fresh store instance
    vi.resetModules();

    // Import fresh store
    const module = await import("./auth.svelte");
    authStore = module.authStore;
  });

  afterEach(() => {
    vi.resetModules();
  });

  describe("initial state", () => {
    it("starts with null token", () => {
      expect(authStore.token).toBeNull();
    });

    it("starts with null user", () => {
      expect(authStore.user).toBeNull();
    });

    it("isAuthenticated is false when no token/user", () => {
      expect(authStore.isAuthenticated).toBe(false);
    });

    it("isAdmin is false when no user", () => {
      expect(authStore.isAdmin).toBe(false);
    });
  });

  describe("initialize", () => {
    it("loads token and user from localStorage", async () => {
      const mockUser = createMockUser();
      mockLocalStorage["mlx_auth_token"] = "test-token";
      mockLocalStorage["mlx_auth_user"] = JSON.stringify(mockUser);

      // Reset and reimport to trigger initialization
      vi.resetModules();
      const module = await import("./auth.svelte");
      authStore = module.authStore;

      // Wait for auto-initialize
      await vi.waitFor(() => {
        expect(authStore.loading).toBe(false);
      });

      expect(authStore.token).toBe("test-token");
      expect(authStore.user).toEqual(mockUser);
      expect(authStore.isAuthenticated).toBe(true);
    });

    it("handles missing localStorage data gracefully", async () => {
      // No data in localStorage
      vi.resetModules();
      const module = await import("./auth.svelte");
      authStore = module.authStore;

      await vi.waitFor(() => {
        expect(authStore.loading).toBe(false);
      });

      expect(authStore.token).toBeNull();
      expect(authStore.user).toBeNull();
      expect(authStore.isAuthenticated).toBe(false);
    });

    it("handles invalid JSON in localStorage", async () => {
      mockLocalStorage["mlx_auth_token"] = "test-token";
      mockLocalStorage["mlx_auth_user"] = "invalid-json";

      vi.resetModules();
      const module = await import("./auth.svelte");
      authStore = module.authStore;

      await vi.waitFor(() => {
        expect(authStore.loading).toBe(false);
      });

      // Should reset to null on parse error
      expect(authStore.token).toBeNull();
      expect(authStore.user).toBeNull();
    });

    it("sets loading to false after initialization", async () => {
      vi.resetModules();
      const module = await import("./auth.svelte");
      authStore = module.authStore;

      await vi.waitFor(() => {
        expect(authStore.loading).toBe(false);
      });
    });
  });

  describe("setAuth", () => {
    it("sets token and user state", () => {
      const mockUser = createMockUser();
      authStore.setAuth("new-token", mockUser);

      expect(authStore.token).toBe("new-token");
      expect(authStore.user).toEqual(mockUser);
      expect(authStore.isAuthenticated).toBe(true);
    });

    it("persists token to localStorage", () => {
      const mockUser = createMockUser();
      authStore.setAuth("persist-token", mockUser);

      expect(localStorage.setItem).toHaveBeenCalledWith(
        "mlx_auth_token",
        "persist-token"
      );
    });

    it("persists user to localStorage as JSON", () => {
      const mockUser = createMockUser();
      authStore.setAuth("token", mockUser);

      expect(localStorage.setItem).toHaveBeenCalledWith(
        "mlx_auth_user",
        JSON.stringify(mockUser)
      );
    });
  });

  describe("clearAuth", () => {
    it("clears token and user state", () => {
      const mockUser = createMockUser();
      authStore.setAuth("token", mockUser);

      authStore.clearAuth();

      expect(authStore.token).toBeNull();
      expect(authStore.user).toBeNull();
      expect(authStore.isAuthenticated).toBe(false);
    });

    it("removes token from localStorage", () => {
      authStore.clearAuth();

      expect(localStorage.removeItem).toHaveBeenCalledWith("mlx_auth_token");
    });

    it("removes user from localStorage", () => {
      authStore.clearAuth();

      expect(localStorage.removeItem).toHaveBeenCalledWith("mlx_auth_user");
    });
  });

  describe("updateUser", () => {
    it("updates user state", () => {
      const initialUser = createMockUser({ email: "old@example.com" });
      authStore.setAuth("token", initialUser);

      const updatedUser = createMockUser({ email: "new@example.com" });
      authStore.updateUser(updatedUser);

      expect(authStore.user).toEqual(updatedUser);
    });

    it("persists updated user to localStorage", () => {
      const initialUser = createMockUser();
      authStore.setAuth("token", initialUser);
      vi.clearAllMocks();

      const updatedUser = createMockUser({ email: "updated@example.com" });
      authStore.updateUser(updatedUser);

      expect(localStorage.setItem).toHaveBeenCalledWith(
        "mlx_auth_user",
        JSON.stringify(updatedUser)
      );
    });

    it("preserves token when updating user", () => {
      const initialUser = createMockUser();
      authStore.setAuth("original-token", initialUser);

      const updatedUser = createMockUser({ email: "new@example.com" });
      authStore.updateUser(updatedUser);

      expect(authStore.token).toBe("original-token");
    });
  });

  describe("isAdmin", () => {
    it("returns true for admin users", () => {
      const adminUser = createMockUser({ is_admin: true });
      authStore.setAuth("token", adminUser);

      expect(authStore.isAdmin).toBe(true);
    });

    it("returns false for non-admin users", () => {
      const regularUser = createMockUser({ is_admin: false });
      authStore.setAuth("token", regularUser);

      expect(authStore.isAdmin).toBe(false);
    });

    it("returns false when no user", () => {
      expect(authStore.isAdmin).toBe(false);
    });
  });

  describe("isAuthenticated", () => {
    it("returns true when both token and user exist", () => {
      const mockUser = createMockUser();
      authStore.setAuth("token", mockUser);

      expect(authStore.isAuthenticated).toBe(true);
    });

    it("returns false when token is null", () => {
      authStore.token = null;
      authStore.user = createMockUser();

      expect(authStore.isAuthenticated).toBe(false);
    });

    it("returns false when user is null", () => {
      authStore.token = "token";
      authStore.user = null;

      expect(authStore.isAuthenticated).toBe(false);
    });

    it("returns false when both are null", () => {
      authStore.token = null;
      authStore.user = null;

      expect(authStore.isAuthenticated).toBe(false);
    });
  });
});
