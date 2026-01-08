// System state management using Svelte 5 runes

import { system as systemApi } from "$api";
import type { SystemMemory, SystemInfo } from "$api";

class SystemStore {
  memory = $state<SystemMemory | null>(null);
  info = $state<SystemInfo | null>(null);
  loading = $state(false);
  error = $state<string | null>(null);

  async refreshMemory() {
    try {
      this.memory = await systemApi.memory();
    } catch (e) {
      console.error("Failed to fetch memory:", e);
    }
  }

  async refreshInfo() {
    this.loading = true;
    try {
      this.info = await systemApi.info();
    } catch (e) {
      this.error =
        e instanceof Error ? e.message : "Failed to fetch system info";
    } finally {
      this.loading = false;
    }
  }

  async refresh() {
    await Promise.all([this.refreshMemory(), this.refreshInfo()]);
  }
}

export const systemStore = new SystemStore();
