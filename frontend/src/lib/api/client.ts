// API Client for MLX Model Manager

import type {
  ServerProfile,
  ServerProfileCreate,
  ServerProfileUpdate,
  RunningServer,
  ModelSearchResult,
  LocalModel,
  SystemMemory,
  SystemInfo,
  HealthStatus,
  LaunchdStatus,
  ServerStatus,
  ModelDetectionInfo,
  ParserOptions,
  User,
  Token,
  UserUpdate,
  ModelCharacteristics,
  ToolDefinition,
  CloudCredential,
  CloudCredentialCreate,
  BackendMapping,
  BackendMappingCreate,
  BackendMappingUpdate,
  ServerPoolConfig,
  ServerPoolConfigUpdate,
  RuleTestResult,
  BackendType,
} from "./types";
import { authStore } from "$lib/stores";

const API_BASE = "/api";

class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/**
 * Get headers with auth token if available.
 */
function getAuthHeaders(): HeadersInit {
  const headers: HeadersInit = { "Content-Type": "application/json" };
  if (authStore.token) {
    headers["Authorization"] = `Bearer ${authStore.token}`;
  }
  return headers;
}

async function handleResponse<T>(response: Response): Promise<T> {
  // Handle 401 - clear auth and redirect to login
  if (response.status === 401) {
    authStore.clearAuth();
    if (typeof window !== "undefined") {
      window.location.href = "/login";
    }
    throw new ApiError(401, "Session expired");
  }

  if (!response.ok) {
    const text = await response.text();
    let message = text;
    try {
      const json = JSON.parse(text);
      // Handle FastAPI validation errors where detail is an array
      if (Array.isArray(json.detail)) {
        // Format validation errors: "field: error message"
        message = json.detail
          .map((err: { loc?: string[]; msg?: string }) => {
            const field = err.loc?.slice(-1)[0] || "field";
            return `${field}: ${err.msg || "validation error"}`;
          })
          .join(", ");
      } else {
        message = json.detail || json.message || text;
      }
    } catch {
      // Use text as-is
    }
    throw new ApiError(response.status, message);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return response.json();
}

// Auth API
export const auth = {
  register: async (data: {
    email: string;
    password: string;
  }): Promise<User> => {
    const res = await fetch(`${API_BASE}/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    return handleResponse(res);
  },

  login: async (email: string, password: string): Promise<Token> => {
    const formData = new URLSearchParams();
    formData.append("username", email); // OAuth2 form uses "username"
    formData.append("password", password);

    const res = await fetch(`${API_BASE}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: formData,
    });
    return handleResponse(res);
  },

  me: async (): Promise<User> => {
    const res = await fetch(`${API_BASE}/auth/me`, {
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },

  listUsers: async (): Promise<User[]> => {
    const res = await fetch(`${API_BASE}/auth/users`, {
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },

  getPendingCount: async (): Promise<{ count: number }> => {
    const res = await fetch(`${API_BASE}/auth/users/pending/count`, {
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },

  updateUser: async (userId: number, data: UserUpdate): Promise<User> => {
    const res = await fetch(`${API_BASE}/auth/users/${userId}`, {
      method: "PUT",
      headers: getAuthHeaders(),
      body: JSON.stringify(data),
    });
    return handleResponse(res);
  },

  deleteUser: async (userId: number): Promise<void> => {
    const res = await fetch(`${API_BASE}/auth/users/${userId}`, {
      method: "DELETE",
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },

  resetPassword: async (
    userId: number,
    password: string,
  ): Promise<{ message: string }> => {
    const res = await fetch(`${API_BASE}/auth/users/${userId}/reset-password`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({ password }),
    });
    return handleResponse(res);
  },
};

// Profiles API
export const profiles = {
  list: async (): Promise<ServerProfile[]> => {
    const res = await fetch(`${API_BASE}/profiles`, {
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },

  get: async (id: number): Promise<ServerProfile> => {
    const res = await fetch(`${API_BASE}/profiles/${id}`, {
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },

  create: async (data: ServerProfileCreate): Promise<ServerProfile> => {
    const res = await fetch(`${API_BASE}/profiles`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify(data),
    });
    return handleResponse(res);
  },

  update: async (
    id: number,
    data: ServerProfileUpdate,
  ): Promise<ServerProfile> => {
    const res = await fetch(`${API_BASE}/profiles/${id}`, {
      method: "PUT",
      headers: getAuthHeaders(),
      body: JSON.stringify(data),
    });
    return handleResponse(res);
  },

  delete: async (id: number): Promise<void> => {
    const res = await fetch(`${API_BASE}/profiles/${id}`, {
      method: "DELETE",
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },

  duplicate: async (id: number, newName: string): Promise<ServerProfile> => {
    const res = await fetch(
      `${API_BASE}/profiles/${id}/duplicate?new_name=${encodeURIComponent(newName)}`,
      {
        method: "POST",
        headers: getAuthHeaders(),
      },
    );
    return handleResponse(res);
  },

  getNextPort: async (): Promise<{ port: number }> => {
    const res = await fetch(`${API_BASE}/profiles/next-port`, {
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },
};

// Active download info from backend
export interface ActiveDownload {
  task_id: string;
  model_id: string;
  status: string;
  progress: number;
  downloaded_bytes: number;
  total_bytes: number;
}

// Models API
export const models = {
  search: async (
    query: string,
    maxSizeGb?: number,
    limit = 20,
  ): Promise<ModelSearchResult[]> => {
    const params = new URLSearchParams({ query, limit: limit.toString() });
    if (maxSizeGb !== undefined) {
      params.set("max_size_gb", maxSizeGb.toString());
    }
    const res = await fetch(`${API_BASE}/models/search?${params}`, {
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },

  listLocal: async (): Promise<LocalModel[]> => {
    const res = await fetch(`${API_BASE}/models/local`, {
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },

  startDownload: async (modelId: string): Promise<{ task_id: string }> => {
    const res = await fetch(`${API_BASE}/models/download`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({ model_id: modelId }),
    });
    return handleResponse(res);
  },

  getActiveDownloads: async (): Promise<ActiveDownload[]> => {
    const res = await fetch(`${API_BASE}/models/downloads/active`, {
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },

  delete: async (modelId: string): Promise<void> => {
    // Note: Don't use encodeURIComponent - backend uses {model_id:path}
    const res = await fetch(`${API_BASE}/models/${modelId}`, {
      method: "DELETE",
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },

  detectOptions: async (modelId: string): Promise<ModelDetectionInfo> => {
    // Note: Don't use encodeURIComponent - backend uses {model_id:path}
    const res = await fetch(`${API_BASE}/models/detect-options/${modelId}`, {
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },

  getAvailableParsers: async (): Promise<{ parsers: string[] }> => {
    const res = await fetch(`${API_BASE}/models/available-parsers`, {
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },

  getConfig: async (
    modelId: string,
    tags?: string[],
  ): Promise<ModelCharacteristics> => {
    // Note: Don't use encodeURIComponent here - the backend uses {model_id:path}
    // which expects slashes to be literal path segments, not URL-encoded
    let url = `${API_BASE}/models/config/${modelId}`;
    if (tags && tags.length > 0) {
      url += `?tags=${encodeURIComponent(tags.join(","))}`;
    }
    const res = await fetch(url, {
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },
};

// Servers API
export const servers = {
  list: async (): Promise<RunningServer[]> => {
    const res = await fetch(`${API_BASE}/servers`, {
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },

  start: async (profileId: number): Promise<{ pid: number; port: number }> => {
    const res = await fetch(`${API_BASE}/servers/${profileId}/start`, {
      method: "POST",
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },

  stop: async (
    profileId: number,
    force = false,
  ): Promise<{ stopped: boolean }> => {
    const params = force ? "?force=true" : "";
    const res = await fetch(`${API_BASE}/servers/${profileId}/stop${params}`, {
      method: "POST",
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },

  restart: async (profileId: number): Promise<{ pid: number }> => {
    const res = await fetch(`${API_BASE}/servers/${profileId}/restart`, {
      method: "POST",
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },

  health: async (profileId: number): Promise<HealthStatus> => {
    const res = await fetch(`${API_BASE}/servers/${profileId}/health`, {
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },

  status: async (profileId: number): Promise<ServerStatus> => {
    const res = await fetch(`${API_BASE}/servers/${profileId}/status`, {
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },
};

// System API
export const system = {
  memory: async (): Promise<SystemMemory> => {
    const res = await fetch(`${API_BASE}/system/memory`, {
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },

  info: async (): Promise<SystemInfo> => {
    const res = await fetch(`${API_BASE}/system/info`, {
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },

  parserOptions: async (): Promise<ParserOptions> => {
    const res = await fetch(`${API_BASE}/system/parser-options`, {
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },

  launchd: {
    install: async (
      profileId: number,
    ): Promise<{ plist_path: string; label: string }> => {
      const res = await fetch(
        `${API_BASE}/system/launchd/install/${profileId}`,
        {
          method: "POST",
          headers: getAuthHeaders(),
        },
      );
      return handleResponse(res);
    },

    uninstall: async (profileId: number): Promise<void> => {
      const res = await fetch(
        `${API_BASE}/system/launchd/uninstall/${profileId}`,
        {
          method: "POST",
          headers: getAuthHeaders(),
        },
      );
      return handleResponse(res);
    },

    status: async (profileId: number): Promise<LaunchdStatus> => {
      const res = await fetch(
        `${API_BASE}/system/launchd/status/${profileId}`,
        {
          headers: getAuthHeaders(),
        },
      );
      return handleResponse(res);
    },
  },
};

// MCP Tools API
export const mcp = {
  listTools: async (): Promise<ToolDefinition[]> => {
    const res = await fetch(`${API_BASE}/mcp/tools`, {
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },

  executeTool: async (
    name: string,
    args: Record<string, unknown>,
  ): Promise<Record<string, unknown>> => {
    const res = await fetch(`${API_BASE}/mcp/execute`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({ name, arguments: args }),
    });
    return handleResponse(res);
  },
};

// Settings API
export const settings = {
  // Providers
  listProviders: async (): Promise<CloudCredential[]> => {
    const res = await fetch(`${API_BASE}/settings/providers`, {
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },

  createProvider: async (
    data: CloudCredentialCreate,
  ): Promise<CloudCredential> => {
    const res = await fetch(`${API_BASE}/settings/providers`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify(data),
    });
    return handleResponse(res);
  },

  deleteProvider: async (backendType: BackendType): Promise<void> => {
    const res = await fetch(`${API_BASE}/settings/providers/${backendType}`, {
      method: "DELETE",
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },

  testProvider: async (
    backendType: BackendType,
  ): Promise<{ success: boolean }> => {
    const res = await fetch(
      `${API_BASE}/settings/providers/${backendType}/test`,
      {
        method: "POST",
        headers: getAuthHeaders(),
      },
    );
    return handleResponse(res);
  },

  // Rules
  listRules: async (): Promise<BackendMapping[]> => {
    const res = await fetch(`${API_BASE}/settings/rules`, {
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },

  createRule: async (data: BackendMappingCreate): Promise<BackendMapping> => {
    const res = await fetch(`${API_BASE}/settings/rules`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify(data),
    });
    return handleResponse(res);
  },

  updateRule: async (
    id: number,
    data: BackendMappingUpdate,
  ): Promise<BackendMapping> => {
    const res = await fetch(`${API_BASE}/settings/rules/${id}`, {
      method: "PUT",
      headers: getAuthHeaders(),
      body: JSON.stringify(data),
    });
    return handleResponse(res);
  },

  deleteRule: async (id: number): Promise<void> => {
    const res = await fetch(`${API_BASE}/settings/rules/${id}`, {
      method: "DELETE",
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },

  updateRulePriorities: async (
    priorities: { id: number; priority: number }[],
  ): Promise<void> => {
    const res = await fetch(`${API_BASE}/settings/rules/priorities`, {
      method: "PUT",
      headers: getAuthHeaders(),
      body: JSON.stringify(priorities),
    });
    return handleResponse(res);
  },

  testRule: async (modelName: string): Promise<RuleTestResult> => {
    const res = await fetch(`${API_BASE}/settings/rules/test`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({ model_name: modelName }),
    });
    return handleResponse(res);
  },

  // Pool config
  getPoolConfig: async (): Promise<ServerPoolConfig> => {
    const res = await fetch(`${API_BASE}/settings/pool`, {
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },

  updatePoolConfig: async (
    data: ServerPoolConfigUpdate,
  ): Promise<ServerPoolConfig> => {
    const res = await fetch(`${API_BASE}/settings/pool`, {
      method: "PUT",
      headers: getAuthHeaders(),
      body: JSON.stringify(data),
    });
    return handleResponse(res);
  },
};

export { ApiError };
