// Profile state management using Svelte 5 runes

import { profiles as profilesApi } from '$api';
import type { ServerProfile, ServerProfileCreate, ServerProfileUpdate } from '$api';

class ProfileStore {
	profiles = $state<ServerProfile[]>([]);
	loading = $state(false);
	error = $state<string | null>(null);

	async refresh() {
		this.loading = true;
		this.error = null;
		try {
			this.profiles = await profilesApi.list();
		} catch (e) {
			this.error = e instanceof Error ? e.message : 'Failed to fetch profiles';
		} finally {
			this.loading = false;
		}
	}

	async create(data: ServerProfileCreate): Promise<ServerProfile> {
		const profile = await profilesApi.create(data);
		await this.refresh();
		return profile;
	}

	async update(id: number, data: ServerProfileUpdate): Promise<ServerProfile> {
		const profile = await profilesApi.update(id, data);
		await this.refresh();
		return profile;
	}

	async delete(id: number): Promise<void> {
		await profilesApi.delete(id);
		await this.refresh();
	}

	async duplicate(id: number, newName: string): Promise<ServerProfile> {
		const profile = await profilesApi.duplicate(id, newName);
		await this.refresh();
		return profile;
	}

	async getNextPort(): Promise<number> {
		const result = await profilesApi.getNextPort();
		return result.port;
	}

	getProfile(id: number): ServerProfile | undefined {
		return this.profiles.find((p) => p.id === id);
	}
}

export const profileStore = new ProfileStore();
