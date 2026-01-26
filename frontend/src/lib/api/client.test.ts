import { describe, it, expect, vi, beforeEach } from "vitest";
import { auth, profiles, models, servers, system, ApiError } from "./client";

// Mock fetch
const mockFetch = vi.fn();
global.fetch = mockFetch;

// Default headers that are always sent
const defaultHeaders = { headers: { "Content-Type": "application/json" } };

function mockResponse(data: unknown, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(data),
    text: () => Promise.resolve(JSON.stringify(data)),
  };
}

function mockErrorResponse(detail: string, status = 400) {
  return {
    ok: false,
    status,
    json: () => Promise.resolve({ detail }),
    text: () => Promise.resolve(JSON.stringify({ detail })),
  };
}

beforeEach(() => {
  mockFetch.mockReset();
});

describe("auth API", () => {
  describe("register", () => {
    it("registers a new user", async () => {
      const mockUser = {
        id: 1,
        email: "test@example.com",
        is_admin: false,
        status: "pending",
      };
      mockFetch.mockResolvedValueOnce(mockResponse(mockUser, 201));

      const result = await auth.register({
        email: "test@example.com",
        password: "password123",
      });

      expect(mockFetch).toHaveBeenCalledWith("/api/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: "test@example.com",
          password: "password123",
        }),
      });
      expect(result).toEqual(mockUser);
    });

    it("throws ApiError on duplicate email", async () => {
      mockFetch.mockResolvedValueOnce(
        mockErrorResponse("Email already registered", 409),
      );

      await expect(
        auth.register({ email: "test@example.com", password: "password123" }),
      ).rejects.toThrow(ApiError);
    });
  });

  describe("login", () => {
    it("logs in with email and password", async () => {
      const mockToken = { access_token: "jwt-token", token_type: "bearer" };
      mockFetch.mockResolvedValueOnce(mockResponse(mockToken));

      const result = await auth.login("test@example.com", "password123");

      expect(mockFetch).toHaveBeenCalledWith("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: expect.any(URLSearchParams),
      });
      expect(result).toEqual(mockToken);
    });

    it("sends form data with username field", async () => {
      mockFetch.mockResolvedValueOnce(
        mockResponse({ access_token: "token", token_type: "bearer" }),
      );

      await auth.login("test@example.com", "password123");

      const [, options] = mockFetch.mock.calls[0];
      const body = options.body as URLSearchParams;
      expect(body.get("username")).toBe("test@example.com");
      expect(body.get("password")).toBe("password123");
    });

    it("throws ApiError on invalid credentials", async () => {
      mockFetch.mockResolvedValueOnce(
        mockErrorResponse("Incorrect email or password", 401),
      );

      await expect(auth.login("test@example.com", "wrong")).rejects.toThrow(
        ApiError,
      );
    });
  });

  describe("me", () => {
    it("fetches current user info", async () => {
      const mockUser = {
        id: 1,
        email: "test@example.com",
        is_admin: true,
        status: "approved",
      };
      mockFetch.mockResolvedValueOnce(mockResponse(mockUser));

      const result = await auth.me();

      expect(mockFetch).toHaveBeenCalledWith(
        "/api/auth/me",
        expect.objectContaining(defaultHeaders),
      );
      expect(result).toEqual(mockUser);
    });
  });

  describe("listUsers", () => {
    it("lists all users (admin only)", async () => {
      const mockUsers = [
        {
          id: 1,
          email: "admin@example.com",
          is_admin: true,
          status: "approved",
        },
        {
          id: 2,
          email: "user@example.com",
          is_admin: false,
          status: "pending",
        },
      ];
      mockFetch.mockResolvedValueOnce(mockResponse(mockUsers));

      const result = await auth.listUsers();

      expect(mockFetch).toHaveBeenCalledWith(
        "/api/auth/users",
        expect.objectContaining(defaultHeaders),
      );
      expect(result).toEqual(mockUsers);
    });
  });

  describe("getPendingCount", () => {
    it("gets count of pending users", async () => {
      mockFetch.mockResolvedValueOnce(mockResponse({ count: 5 }));

      const result = await auth.getPendingCount();

      expect(mockFetch).toHaveBeenCalledWith(
        "/api/auth/users/pending/count",
        expect.objectContaining(defaultHeaders),
      );
      expect(result).toEqual({ count: 5 });
    });
  });

  describe("updateUser", () => {
    it("updates user status", async () => {
      const updatedUser = {
        id: 2,
        email: "user@example.com",
        is_admin: false,
        status: "approved",
      };
      mockFetch.mockResolvedValueOnce(mockResponse(updatedUser));

      const result = await auth.updateUser(2, { status: "approved" });

      expect(mockFetch).toHaveBeenCalledWith("/api/auth/users/2", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: "approved" }),
      });
      expect(result).toEqual(updatedUser);
    });

    it("updates user admin status", async () => {
      const updatedUser = {
        id: 2,
        email: "user@example.com",
        is_admin: true,
        status: "approved",
      };
      mockFetch.mockResolvedValueOnce(mockResponse(updatedUser));

      const result = await auth.updateUser(2, { is_admin: true });

      expect(mockFetch).toHaveBeenCalledWith("/api/auth/users/2", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ is_admin: true }),
      });
      expect(result).toEqual(updatedUser);
    });
  });

  describe("deleteUser", () => {
    it("deletes a user", async () => {
      mockFetch.mockResolvedValueOnce({ ok: true, status: 204 });

      await auth.deleteUser(2);

      expect(mockFetch).toHaveBeenCalledWith("/api/auth/users/2", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
      });
    });
  });

  describe("resetPassword", () => {
    it("resets user password", async () => {
      mockFetch.mockResolvedValueOnce(
        mockResponse({ message: "Password reset successfully" }),
      );

      const result = await auth.resetPassword(2, "newpassword123");

      expect(mockFetch).toHaveBeenCalledWith(
        "/api/auth/users/2/reset-password",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ password: "newpassword123" }),
        },
      );
      expect(result).toEqual({ message: "Password reset successfully" });
    });
  });
});

describe("profiles API", () => {
  describe("list", () => {
    it("fetches all profiles", async () => {
      const mockProfiles = [
        { id: 1, name: "Test Profile", port: 10240 },
        { id: 2, name: "Another Profile", port: 10241 },
      ];
      mockFetch.mockResolvedValueOnce(mockResponse(mockProfiles));

      const result = await profiles.list();

      expect(mockFetch).toHaveBeenCalledWith(
        "/api/profiles",
        expect.objectContaining(defaultHeaders),
      );
      expect(result).toEqual(mockProfiles);
    });
  });

  describe("get", () => {
    it("fetches a single profile by id", async () => {
      const mockProfile = { id: 1, name: "Test Profile", port: 10240 };
      mockFetch.mockResolvedValueOnce(mockResponse(mockProfile));

      const result = await profiles.get(1);

      expect(mockFetch).toHaveBeenCalledWith(
        "/api/profiles/1",
        expect.objectContaining(defaultHeaders),
      );
      expect(result).toEqual(mockProfile);
    });

    it("throws ApiError on 404", async () => {
      mockFetch.mockResolvedValueOnce(
        mockErrorResponse("Profile not found", 404),
      );

      await expect(profiles.get(999)).rejects.toThrow(ApiError);
    });
  });

  describe("create", () => {
    it("creates a new profile", async () => {
      const newProfile = {
        name: "New Profile",
        model_path: "mlx-community/test",
        port: 10240,
      };
      const createdProfile = { id: 1, ...newProfile };
      mockFetch.mockResolvedValueOnce(mockResponse(createdProfile, 201));

      const result = await profiles.create(newProfile);

      expect(mockFetch).toHaveBeenCalledWith("/api/profiles", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(newProfile),
      });
      expect(result).toEqual(createdProfile);
    });
  });

  describe("update", () => {
    it("updates an existing profile", async () => {
      const updates = { name: "Updated Profile" };
      const updatedProfile = { id: 1, name: "Updated Profile", port: 10240 };
      mockFetch.mockResolvedValueOnce(mockResponse(updatedProfile));

      const result = await profiles.update(1, updates);

      expect(mockFetch).toHaveBeenCalledWith("/api/profiles/1", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(updates),
      });
      expect(result).toEqual(updatedProfile);
    });
  });

  describe("delete", () => {
    it("deletes a profile", async () => {
      mockFetch.mockResolvedValueOnce({ ok: true, status: 204 });

      await profiles.delete(1);

      expect(mockFetch).toHaveBeenCalledWith(
        "/api/profiles/1",
        expect.objectContaining({
          method: "DELETE",
          ...defaultHeaders,
        }),
      );
    });
  });

  describe("getNextPort", () => {
    it("returns next available port", async () => {
      mockFetch.mockResolvedValueOnce(mockResponse({ port: 10242 }));

      const result = await profiles.getNextPort();

      expect(mockFetch).toHaveBeenCalledWith(
        "/api/profiles/next-port",
        expect.objectContaining(defaultHeaders),
      );
      expect(result).toEqual({ port: 10242 });
    });
  });

  describe("duplicate", () => {
    it("duplicates a profile with new name", async () => {
      const duplicatedProfile = { id: 2, name: "Copy of Profile", port: 10241 };
      mockFetch.mockResolvedValueOnce(mockResponse(duplicatedProfile, 201));

      const result = await profiles.duplicate(1, "Copy of Profile");

      expect(mockFetch).toHaveBeenCalledWith(
        "/api/profiles/1/duplicate?new_name=Copy%20of%20Profile",
        expect.objectContaining({
          method: "POST",
          ...defaultHeaders,
        }),
      );
      expect(result).toEqual(duplicatedProfile);
    });
  });
});

describe("models API", () => {
  describe("search", () => {
    it("searches models with query", async () => {
      const mockModels = [{ model_id: "mlx-community/test", downloads: 1000 }];
      mockFetch.mockResolvedValueOnce(mockResponse(mockModels));

      const result = await models.search("test");

      expect(mockFetch).toHaveBeenCalledWith(
        "/api/models/search?query=test&limit=20",
        expect.objectContaining(defaultHeaders),
      );
      expect(result).toEqual(mockModels);
    });

    it("includes max_size_gb when provided", async () => {
      mockFetch.mockResolvedValueOnce(mockResponse([]));

      await models.search("test", 50, 10);

      expect(mockFetch).toHaveBeenCalledWith(
        "/api/models/search?query=test&limit=10&max_size_gb=50",
        expect.objectContaining(defaultHeaders),
      );
    });
  });

  describe("listLocal", () => {
    it("lists locally downloaded models", async () => {
      const mockModels = [
        { model_id: "mlx-community/test", local_path: "/path/to/model" },
      ];
      mockFetch.mockResolvedValueOnce(mockResponse(mockModels));

      const result = await models.listLocal();

      expect(mockFetch).toHaveBeenCalledWith(
        "/api/models/local",
        expect.objectContaining(defaultHeaders),
      );
      expect(result).toEqual(mockModels);
    });
  });

  describe("startDownload", () => {
    it("starts a model download", async () => {
      mockFetch.mockResolvedValueOnce(mockResponse({ task_id: "abc123" }));

      const result = await models.startDownload("mlx-community/test");

      expect(mockFetch).toHaveBeenCalledWith("/api/models/download", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: "mlx-community/test" }),
      });
      expect(result).toEqual({ task_id: "abc123" });
    });
  });

  describe("getActiveDownloads", () => {
    it("gets active downloads", async () => {
      const activeDownloads = [
        {
          task_id: "abc123",
          model_id: "mlx-community/test",
          status: "downloading",
          progress: 50,
        },
      ];
      mockFetch.mockResolvedValueOnce(mockResponse(activeDownloads));

      const result = await models.getActiveDownloads();

      expect(mockFetch).toHaveBeenCalledWith(
        "/api/models/downloads/active",
        expect.objectContaining(defaultHeaders),
      );
      expect(result).toEqual(activeDownloads);
    });
  });

  describe("delete", () => {
    it("deletes a model", async () => {
      mockFetch.mockResolvedValueOnce({ ok: true, status: 204 });

      await models.delete("mlx-community/test-model");

      // Note: No URL encoding - backend uses {model_id:path} which expects literal slashes
      expect(mockFetch).toHaveBeenCalledWith(
        "/api/models/mlx-community/test-model",
        expect.objectContaining({
          method: "DELETE",
          ...defaultHeaders,
        }),
      );
    });
  });

  describe("detectOptions", () => {
    it("detects model options", async () => {
      const detectionInfo = { chat_template: "default", model_type: "llama" };
      mockFetch.mockResolvedValueOnce(mockResponse(detectionInfo));

      const result = await models.detectOptions("mlx-community/test-model");

      // Note: No URL encoding - backend uses {model_id:path} which expects literal slashes
      expect(mockFetch).toHaveBeenCalledWith(
        "/api/models/detect-options/mlx-community/test-model",
        expect.objectContaining(defaultHeaders),
      );
      expect(result).toEqual(detectionInfo);
    });
  });

  describe("getAvailableParsers", () => {
    it("gets available parsers", async () => {
      const parsers = { parsers: ["default", "llama", "mistral"] };
      mockFetch.mockResolvedValueOnce(mockResponse(parsers));

      const result = await models.getAvailableParsers();

      expect(mockFetch).toHaveBeenCalledWith(
        "/api/models/available-parsers",
        expect.objectContaining(defaultHeaders),
      );
      expect(result).toEqual(parsers);
    });
  });
});

describe("servers API", () => {
  describe("list", () => {
    it("lists running servers", async () => {
      const mockServers = [
        { profile_id: 1, pid: 12345, health_status: "healthy" },
      ];
      mockFetch.mockResolvedValueOnce(mockResponse(mockServers));

      const result = await servers.list();

      expect(mockFetch).toHaveBeenCalledWith(
        "/api/servers",
        expect.objectContaining(defaultHeaders),
      );
      expect(result).toEqual(mockServers);
    });
  });

  describe("start", () => {
    it("starts a server", async () => {
      mockFetch.mockResolvedValueOnce(
        mockResponse({ pid: 12345, port: 10240 }),
      );

      const result = await servers.start(1);

      expect(mockFetch).toHaveBeenCalledWith(
        "/api/servers/1/start",
        expect.objectContaining({
          method: "POST",
          ...defaultHeaders,
        }),
      );
      expect(result).toEqual({ pid: 12345, port: 10240 });
    });
  });

  describe("stop", () => {
    it("stops a server gracefully", async () => {
      mockFetch.mockResolvedValueOnce(mockResponse({ stopped: true }));

      const result = await servers.stop(1);

      expect(mockFetch).toHaveBeenCalledWith(
        "/api/servers/1/stop",
        expect.objectContaining({
          method: "POST",
          ...defaultHeaders,
        }),
      );
      expect(result).toEqual({ stopped: true });
    });

    it("force stops a server", async () => {
      mockFetch.mockResolvedValueOnce(mockResponse({ stopped: true }));

      await servers.stop(1, true);

      expect(mockFetch).toHaveBeenCalledWith(
        "/api/servers/1/stop?force=true",
        expect.objectContaining({
          method: "POST",
          ...defaultHeaders,
        }),
      );
    });
  });

  describe("restart", () => {
    it("restarts a server", async () => {
      mockFetch.mockResolvedValueOnce(mockResponse({ pid: 12346 }));

      const result = await servers.restart(1);

      expect(mockFetch).toHaveBeenCalledWith(
        "/api/servers/1/restart",
        expect.objectContaining({
          method: "POST",
          ...defaultHeaders,
        }),
      );
      expect(result).toEqual({ pid: 12346 });
    });
  });

  describe("health", () => {
    it("checks server health", async () => {
      const healthStatus = { status: "healthy", response_time_ms: 45 };
      mockFetch.mockResolvedValueOnce(mockResponse(healthStatus));

      const result = await servers.health(1);

      expect(mockFetch).toHaveBeenCalledWith(
        "/api/servers/1/health",
        expect.objectContaining(defaultHeaders),
      );
      expect(result).toEqual(healthStatus);
    });
  });

  describe("status", () => {
    it("gets server status", async () => {
      const serverStatus = { running: true, pid: 12345, health: "healthy" };
      mockFetch.mockResolvedValueOnce(mockResponse(serverStatus));

      const result = await servers.status(1);

      expect(mockFetch).toHaveBeenCalledWith(
        "/api/servers/1/status",
        expect.objectContaining(defaultHeaders),
      );
      expect(result).toEqual(serverStatus);
    });
  });
});

describe("system API", () => {
  describe("memory", () => {
    it("returns memory info", async () => {
      const memoryInfo = { total_gb: 128, available_gb: 64 };
      mockFetch.mockResolvedValueOnce(mockResponse(memoryInfo));

      const result = await system.memory();

      expect(mockFetch).toHaveBeenCalledWith(
        "/api/system/memory",
        expect.objectContaining(defaultHeaders),
      );
      expect(result).toEqual(memoryInfo);
    });
  });

  describe("info", () => {
    it("returns system info", async () => {
      const sysInfo = { os_version: "Darwin", chip: "Apple M4 Max" };
      mockFetch.mockResolvedValueOnce(mockResponse(sysInfo));

      const result = await system.info();

      expect(mockFetch).toHaveBeenCalledWith(
        "/api/system/info",
        expect.objectContaining(defaultHeaders),
      );
      expect(result).toEqual(sysInfo);
    });
  });

  describe("parserOptions", () => {
    it("returns parser options", async () => {
      const parserOpts = { available_parsers: ["default", "custom"] };
      mockFetch.mockResolvedValueOnce(mockResponse(parserOpts));

      const result = await system.parserOptions();

      expect(mockFetch).toHaveBeenCalledWith(
        "/api/system/parser-options",
        expect.objectContaining(defaultHeaders),
      );
      expect(result).toEqual(parserOpts);
    });
  });

  describe("launchd", () => {
    it("installs launchd service", async () => {
      const installResult = {
        plist_path: "/path/to/plist",
        label: "com.mlx-manager.test",
      };
      mockFetch.mockResolvedValueOnce(mockResponse(installResult));

      const result = await system.launchd.install(1);

      expect(mockFetch).toHaveBeenCalledWith(
        "/api/system/launchd/install/1",
        expect.objectContaining({
          method: "POST",
          ...defaultHeaders,
        }),
      );
      expect(result).toEqual(installResult);
    });

    it("uninstalls launchd service", async () => {
      mockFetch.mockResolvedValueOnce({ ok: true, status: 204 });

      await system.launchd.uninstall(1);

      expect(mockFetch).toHaveBeenCalledWith(
        "/api/system/launchd/uninstall/1",
        expect.objectContaining({
          method: "POST",
          ...defaultHeaders,
        }),
      );
    });

    it("gets launchd status", async () => {
      const status = {
        installed: true,
        running: true,
        label: "com.mlx-manager.test",
      };
      mockFetch.mockResolvedValueOnce(mockResponse(status));

      const result = await system.launchd.status(1);

      expect(mockFetch).toHaveBeenCalledWith(
        "/api/system/launchd/status/1",
        expect.objectContaining(defaultHeaders),
      );
      expect(result).toEqual(status);
    });
  });
});

describe("ApiError", () => {
  it("includes status code and message", async () => {
    mockFetch.mockResolvedValueOnce(
      mockErrorResponse("Something went wrong", 500),
    );

    try {
      await profiles.list();
    } catch (error) {
      expect(error).toBeInstanceOf(ApiError);
      expect((error as ApiError).status).toBe(500);
      expect((error as ApiError).message).toBe("Something went wrong");
    }
  });

  it("handles validation errors with array detail", async () => {
    const validationErrors = {
      detail: [
        { loc: ["body", "name"], msg: "field required" },
        { loc: ["body", "port"], msg: "value is not a valid integer" },
      ],
    };
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 422,
      json: () => Promise.resolve(validationErrors),
      text: () => Promise.resolve(JSON.stringify(validationErrors)),
    });

    try {
      await profiles.create({ name: "", model_path: "test", port: 0 });
    } catch (error) {
      expect(error).toBeInstanceOf(ApiError);
      expect((error as ApiError).status).toBe(422);
      expect((error as ApiError).message).toBe(
        "name: field required, port: value is not a valid integer",
      );
    }
  });

  it("handles validation errors with missing loc", async () => {
    const validationErrors = {
      detail: [{ msg: "some error" }],
    };
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 422,
      json: () => Promise.resolve(validationErrors),
      text: () => Promise.resolve(JSON.stringify(validationErrors)),
    });

    try {
      await profiles.create({ name: "", model_path: "test", port: 0 });
    } catch (error) {
      expect(error).toBeInstanceOf(ApiError);
      expect((error as ApiError).message).toBe("field: some error");
    }
  });

  it("handles validation errors with missing msg", async () => {
    const validationErrors = {
      detail: [{ loc: ["body", "name"] }],
    };
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 422,
      json: () => Promise.resolve(validationErrors),
      text: () => Promise.resolve(JSON.stringify(validationErrors)),
    });

    try {
      await profiles.create({ name: "", model_path: "test", port: 0 });
    } catch (error) {
      expect(error).toBeInstanceOf(ApiError);
      expect((error as ApiError).message).toBe("name: validation error");
    }
  });

  it("handles non-JSON error response", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      json: () => Promise.reject(new Error("Not JSON")),
      text: () => Promise.resolve("Internal Server Error"),
    });

    try {
      await profiles.list();
    } catch (error) {
      expect(error).toBeInstanceOf(ApiError);
      expect((error as ApiError).status).toBe(500);
      expect((error as ApiError).message).toBe("Internal Server Error");
    }
  });

  it("handles error with message field instead of detail", async () => {
    const errorResponse = { message: "Resource not found" };
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 404,
      json: () => Promise.resolve(errorResponse),
      text: () => Promise.resolve(JSON.stringify(errorResponse)),
    });

    try {
      await profiles.get(999);
    } catch (error) {
      expect(error).toBeInstanceOf(ApiError);
      expect((error as ApiError).message).toBe("Resource not found");
    }
  });

  it("falls back to raw text when JSON has no detail or message", async () => {
    const errorResponse = { error: "unknown" };
    const rawText = JSON.stringify(errorResponse);
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      json: () => Promise.resolve(errorResponse),
      text: () => Promise.resolve(rawText),
    });

    try {
      await profiles.list();
    } catch (error) {
      expect(error).toBeInstanceOf(ApiError);
      expect((error as ApiError).message).toBe(rawText);
    }
  });

  it("clears auth and throws on 401 response", async () => {
    // Mock window.location
    const originalLocation = window.location;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    delete (window as any).location;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (window as any).location = { href: "" };

    // Mock authStore
    const { authStore } = await import("$lib/stores");
    const clearAuthSpy = vi.spyOn(authStore, "clearAuth");

    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 401,
      json: () => Promise.resolve({ detail: "Not authenticated" }),
      text: () =>
        Promise.resolve(JSON.stringify({ detail: "Not authenticated" })),
    });

    try {
      await profiles.list();
    } catch (error) {
      expect(error).toBeInstanceOf(ApiError);
      expect((error as ApiError).status).toBe(401);
      expect((error as ApiError).message).toBe("Session expired");
      expect(clearAuthSpy).toHaveBeenCalled();
      expect(window.location.href).toBe("/login");
    }

    // Restore
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (window as any).location = originalLocation;
    clearAuthSpy.mockRestore();
  });
});
