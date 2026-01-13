// Server state management using Svelte 5 runes

import { servers as serversApi } from "$api";
import type { RunningServer } from "$api";

class ServerStore {
  servers = $state<RunningServer[]>([]);
  loading = $state(false);
  error = $state<string | null>(null);

  async refresh() {
    this.loading = true;
    this.error = null;
    try {
      this.servers = await serversApi.list();
    } catch (e) {
      this.error = e instanceof Error ? e.message : "Failed to fetch servers";
    } finally {
      this.loading = false;
    }
  }

  async start(profileId: number) {
    await serversApi.start(profileId);
    await this.refresh();
  }

  async stop(profileId: number, force = false) {
    await serversApi.stop(profileId, force);
    await this.refresh();
  }

  async restart(profileId: number) {
    await serversApi.restart(profileId);
    await this.refresh();
  }

  isRunning(profileId: number): boolean {
    return this.servers.some((s) => s.profile_id === profileId);
  }

  getServer(profileId: number): RunningServer | undefined {
    return this.servers.find((s) => s.profile_id === profileId);
  }
}

export const serverStore = new ServerStore();
