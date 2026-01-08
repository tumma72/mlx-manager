import { describe, it, expect, vi, beforeEach } from 'vitest';
import { profiles, models, servers, system, ApiError } from './client';

// Mock fetch
const mockFetch = vi.fn();
global.fetch = mockFetch;

function mockResponse(data: unknown, status = 200) {
	return {
		ok: status >= 200 && status < 300,
		status,
		json: () => Promise.resolve(data),
		text: () => Promise.resolve(JSON.stringify(data))
	};
}

function mockErrorResponse(detail: string, status = 400) {
	return {
		ok: false,
		status,
		json: () => Promise.resolve({ detail }),
		text: () => Promise.resolve(JSON.stringify({ detail }))
	};
}

beforeEach(() => {
	mockFetch.mockReset();
});

describe('profiles API', () => {
	describe('list', () => {
		it('fetches all profiles', async () => {
			const mockProfiles = [
				{ id: 1, name: 'Test Profile', port: 10240 },
				{ id: 2, name: 'Another Profile', port: 10241 }
			];
			mockFetch.mockResolvedValueOnce(mockResponse(mockProfiles));

			const result = await profiles.list();

			expect(mockFetch).toHaveBeenCalledWith('/api/profiles');
			expect(result).toEqual(mockProfiles);
		});
	});

	describe('get', () => {
		it('fetches a single profile by id', async () => {
			const mockProfile = { id: 1, name: 'Test Profile', port: 10240 };
			mockFetch.mockResolvedValueOnce(mockResponse(mockProfile));

			const result = await profiles.get(1);

			expect(mockFetch).toHaveBeenCalledWith('/api/profiles/1');
			expect(result).toEqual(mockProfile);
		});

		it('throws ApiError on 404', async () => {
			mockFetch.mockResolvedValueOnce(mockErrorResponse('Profile not found', 404));

			await expect(profiles.get(999)).rejects.toThrow(ApiError);
		});
	});

	describe('create', () => {
		it('creates a new profile', async () => {
			const newProfile = { name: 'New Profile', model_path: 'mlx-community/test', port: 10240 };
			const createdProfile = { id: 1, ...newProfile };
			mockFetch.mockResolvedValueOnce(mockResponse(createdProfile, 201));

			const result = await profiles.create(newProfile);

			expect(mockFetch).toHaveBeenCalledWith('/api/profiles', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(newProfile)
			});
			expect(result).toEqual(createdProfile);
		});
	});

	describe('update', () => {
		it('updates an existing profile', async () => {
			const updates = { name: 'Updated Profile' };
			const updatedProfile = { id: 1, name: 'Updated Profile', port: 10240 };
			mockFetch.mockResolvedValueOnce(mockResponse(updatedProfile));

			const result = await profiles.update(1, updates);

			expect(mockFetch).toHaveBeenCalledWith('/api/profiles/1', {
				method: 'PUT',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(updates)
			});
			expect(result).toEqual(updatedProfile);
		});
	});

	describe('delete', () => {
		it('deletes a profile', async () => {
			mockFetch.mockResolvedValueOnce({ ok: true, status: 204 });

			await profiles.delete(1);

			expect(mockFetch).toHaveBeenCalledWith('/api/profiles/1', { method: 'DELETE' });
		});
	});

	describe('getNextPort', () => {
		it('returns next available port', async () => {
			mockFetch.mockResolvedValueOnce(mockResponse({ port: 10242 }));

			const result = await profiles.getNextPort();

			expect(mockFetch).toHaveBeenCalledWith('/api/profiles/next-port');
			expect(result).toEqual({ port: 10242 });
		});
	});
});

describe('models API', () => {
	describe('search', () => {
		it('searches models with query', async () => {
			const mockModels = [{ model_id: 'mlx-community/test', downloads: 1000 }];
			mockFetch.mockResolvedValueOnce(mockResponse(mockModels));

			const result = await models.search('test');

			expect(mockFetch).toHaveBeenCalledWith('/api/models/search?query=test&limit=20');
			expect(result).toEqual(mockModels);
		});

		it('includes max_size_gb when provided', async () => {
			mockFetch.mockResolvedValueOnce(mockResponse([]));

			await models.search('test', 50, 10);

			expect(mockFetch).toHaveBeenCalledWith(
				'/api/models/search?query=test&limit=10&max_size_gb=50'
			);
		});
	});

	describe('listLocal', () => {
		it('lists locally downloaded models', async () => {
			const mockModels = [{ model_id: 'mlx-community/test', local_path: '/path/to/model' }];
			mockFetch.mockResolvedValueOnce(mockResponse(mockModels));

			const result = await models.listLocal();

			expect(mockFetch).toHaveBeenCalledWith('/api/models/local');
			expect(result).toEqual(mockModels);
		});
	});

	describe('startDownload', () => {
		it('starts a model download', async () => {
			mockFetch.mockResolvedValueOnce(mockResponse({ task_id: 'abc123' }));

			const result = await models.startDownload('mlx-community/test');

			expect(mockFetch).toHaveBeenCalledWith('/api/models/download', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ model_id: 'mlx-community/test' })
			});
			expect(result).toEqual({ task_id: 'abc123' });
		});
	});
});

describe('servers API', () => {
	describe('list', () => {
		it('lists running servers', async () => {
			const mockServers = [{ profile_id: 1, pid: 12345, health_status: 'healthy' }];
			mockFetch.mockResolvedValueOnce(mockResponse(mockServers));

			const result = await servers.list();

			expect(mockFetch).toHaveBeenCalledWith('/api/servers');
			expect(result).toEqual(mockServers);
		});
	});

	describe('start', () => {
		it('starts a server', async () => {
			mockFetch.mockResolvedValueOnce(mockResponse({ pid: 12345, port: 10240 }));

			const result = await servers.start(1);

			expect(mockFetch).toHaveBeenCalledWith('/api/servers/1/start', { method: 'POST' });
			expect(result).toEqual({ pid: 12345, port: 10240 });
		});
	});

	describe('stop', () => {
		it('stops a server gracefully', async () => {
			mockFetch.mockResolvedValueOnce(mockResponse({ stopped: true }));

			const result = await servers.stop(1);

			expect(mockFetch).toHaveBeenCalledWith('/api/servers/1/stop', { method: 'POST' });
			expect(result).toEqual({ stopped: true });
		});

		it('force stops a server', async () => {
			mockFetch.mockResolvedValueOnce(mockResponse({ stopped: true }));

			await servers.stop(1, true);

			expect(mockFetch).toHaveBeenCalledWith('/api/servers/1/stop?force=true', {
				method: 'POST'
			});
		});
	});

	describe('restart', () => {
		it('restarts a server', async () => {
			mockFetch.mockResolvedValueOnce(mockResponse({ pid: 12346 }));

			const result = await servers.restart(1);

			expect(mockFetch).toHaveBeenCalledWith('/api/servers/1/restart', { method: 'POST' });
			expect(result).toEqual({ pid: 12346 });
		});
	});

	describe('health', () => {
		it('checks server health', async () => {
			const healthStatus = { status: 'healthy', response_time_ms: 45 };
			mockFetch.mockResolvedValueOnce(mockResponse(healthStatus));

			const result = await servers.health(1);

			expect(mockFetch).toHaveBeenCalledWith('/api/servers/1/health');
			expect(result).toEqual(healthStatus);
		});
	});
});

describe('system API', () => {
	describe('memory', () => {
		it('returns memory info', async () => {
			const memoryInfo = { total_gb: 128, available_gb: 64 };
			mockFetch.mockResolvedValueOnce(mockResponse(memoryInfo));

			const result = await system.memory();

			expect(mockFetch).toHaveBeenCalledWith('/api/system/memory');
			expect(result).toEqual(memoryInfo);
		});
	});

	describe('info', () => {
		it('returns system info', async () => {
			const sysInfo = { os_version: 'Darwin', chip: 'Apple M4 Max' };
			mockFetch.mockResolvedValueOnce(mockResponse(sysInfo));

			const result = await system.info();

			expect(mockFetch).toHaveBeenCalledWith('/api/system/info');
			expect(result).toEqual(sysInfo);
		});
	});

	describe('launchd', () => {
		it('installs launchd service', async () => {
			const installResult = { plist_path: '/path/to/plist', label: 'com.mlx-manager.test' };
			mockFetch.mockResolvedValueOnce(mockResponse(installResult));

			const result = await system.launchd.install(1);

			expect(mockFetch).toHaveBeenCalledWith('/api/system/launchd/install/1', {
				method: 'POST'
			});
			expect(result).toEqual(installResult);
		});

		it('uninstalls launchd service', async () => {
			mockFetch.mockResolvedValueOnce({ ok: true, status: 204 });

			await system.launchd.uninstall(1);

			expect(mockFetch).toHaveBeenCalledWith('/api/system/launchd/uninstall/1', {
				method: 'POST'
			});
		});

		it('gets launchd status', async () => {
			const status = { installed: true, running: true, label: 'com.mlx-manager.test' };
			mockFetch.mockResolvedValueOnce(mockResponse(status));

			const result = await system.launchd.status(1);

			expect(mockFetch).toHaveBeenCalledWith('/api/system/launchd/status/1');
			expect(result).toEqual(status);
		});
	});
});

describe('ApiError', () => {
	it('includes status code and message', async () => {
		mockFetch.mockResolvedValueOnce(mockErrorResponse('Something went wrong', 500));

		try {
			await profiles.list();
		} catch (error) {
			expect(error).toBeInstanceOf(ApiError);
			expect((error as ApiError).status).toBe(500);
			expect((error as ApiError).message).toBe('Something went wrong');
		}
	});
});
