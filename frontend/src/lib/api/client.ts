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
	LaunchdStatus
} from './types';

const API_BASE = '/api';

class ApiError extends Error {
	constructor(
		public status: number,
		message: string
	) {
		super(message);
		this.name = 'ApiError';
	}
}

async function handleResponse<T>(response: Response): Promise<T> {
	if (!response.ok) {
		const text = await response.text();
		let message = text;
		try {
			const json = JSON.parse(text);
			message = json.detail || json.message || text;
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

// Profiles API
export const profiles = {
	list: async (): Promise<ServerProfile[]> => {
		const res = await fetch(`${API_BASE}/profiles`);
		return handleResponse(res);
	},

	get: async (id: number): Promise<ServerProfile> => {
		const res = await fetch(`${API_BASE}/profiles/${id}`);
		return handleResponse(res);
	},

	create: async (data: ServerProfileCreate): Promise<ServerProfile> => {
		const res = await fetch(`${API_BASE}/profiles`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(data)
		});
		return handleResponse(res);
	},

	update: async (id: number, data: ServerProfileUpdate): Promise<ServerProfile> => {
		const res = await fetch(`${API_BASE}/profiles/${id}`, {
			method: 'PUT',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(data)
		});
		return handleResponse(res);
	},

	delete: async (id: number): Promise<void> => {
		const res = await fetch(`${API_BASE}/profiles/${id}`, { method: 'DELETE' });
		return handleResponse(res);
	},

	duplicate: async (id: number, newName: string): Promise<ServerProfile> => {
		const res = await fetch(`${API_BASE}/profiles/${id}/duplicate?new_name=${encodeURIComponent(newName)}`, {
			method: 'POST'
		});
		return handleResponse(res);
	},

	getNextPort: async (): Promise<{ port: number }> => {
		const res = await fetch(`${API_BASE}/profiles/next-port`);
		return handleResponse(res);
	}
};

// Models API
export const models = {
	search: async (query: string, maxSizeGb?: number, limit = 20): Promise<ModelSearchResult[]> => {
		const params = new URLSearchParams({ query, limit: limit.toString() });
		if (maxSizeGb !== undefined) {
			params.set('max_size_gb', maxSizeGb.toString());
		}
		const res = await fetch(`${API_BASE}/models/search?${params}`);
		return handleResponse(res);
	},

	listLocal: async (): Promise<LocalModel[]> => {
		const res = await fetch(`${API_BASE}/models/local`);
		return handleResponse(res);
	},

	startDownload: async (modelId: string): Promise<{ task_id: string }> => {
		const res = await fetch(`${API_BASE}/models/download`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ model_id: modelId })
		});
		return handleResponse(res);
	},

	delete: async (modelId: string): Promise<void> => {
		const res = await fetch(`${API_BASE}/models/${encodeURIComponent(modelId)}`, {
			method: 'DELETE'
		});
		return handleResponse(res);
	}
};

// Servers API
export const servers = {
	list: async (): Promise<RunningServer[]> => {
		const res = await fetch(`${API_BASE}/servers`);
		return handleResponse(res);
	},

	start: async (profileId: number): Promise<{ pid: number; port: number }> => {
		const res = await fetch(`${API_BASE}/servers/${profileId}/start`, { method: 'POST' });
		return handleResponse(res);
	},

	stop: async (profileId: number, force = false): Promise<{ stopped: boolean }> => {
		const params = force ? '?force=true' : '';
		const res = await fetch(`${API_BASE}/servers/${profileId}/stop${params}`, { method: 'POST' });
		return handleResponse(res);
	},

	restart: async (profileId: number): Promise<{ pid: number }> => {
		const res = await fetch(`${API_BASE}/servers/${profileId}/restart`, { method: 'POST' });
		return handleResponse(res);
	},

	health: async (profileId: number): Promise<HealthStatus> => {
		const res = await fetch(`${API_BASE}/servers/${profileId}/health`);
		return handleResponse(res);
	}
};

// System API
export const system = {
	memory: async (): Promise<SystemMemory> => {
		const res = await fetch(`${API_BASE}/system/memory`);
		return handleResponse(res);
	},

	info: async (): Promise<SystemInfo> => {
		const res = await fetch(`${API_BASE}/system/info`);
		return handleResponse(res);
	},

	launchd: {
		install: async (profileId: number): Promise<{ plist_path: string; label: string }> => {
			const res = await fetch(`${API_BASE}/system/launchd/install/${profileId}`, {
				method: 'POST'
			});
			return handleResponse(res);
		},

		uninstall: async (profileId: number): Promise<void> => {
			const res = await fetch(`${API_BASE}/system/launchd/uninstall/${profileId}`, {
				method: 'POST'
			});
			return handleResponse(res);
		},

		status: async (profileId: number): Promise<LaunchdStatus> => {
			const res = await fetch(`${API_BASE}/system/launchd/status/${profileId}`);
			return handleResponse(res);
		}
	}
};

export { ApiError };
