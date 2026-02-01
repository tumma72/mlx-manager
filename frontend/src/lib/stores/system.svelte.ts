/**
 * System state management using Svelte 5 runes.
 *
 * Refactored to use:
 * - Shallow comparison before updates (prevents unnecessary re-renders)
 * - Polling coordinator for memory polling
 */

import { system as systemApi } from "$api";
import type { SystemMemory, SystemInfo } from "$api";
import { pollingCoordinator } from "$lib/services";

/**
 * Check if two SystemMemory objects are equal.
 */
function memoryEqual(a: SystemMemory | null, b: SystemMemory): boolean {
  if (a === null) return false;
  return (
    a.total_gb === b.total_gb &&
    a.available_gb === b.available_gb &&
    a.used_gb === b.used_gb &&
    a.percent_used === b.percent_used &&
    a.mlx_recommended_gb === b.mlx_recommended_gb
  );
}

/**
 * Check if two SystemInfo objects are equal.
 */
function infoEqual(a: SystemInfo | null, b: SystemInfo): boolean {
  if (a === null) return false;
  return (
    a.os_version === b.os_version &&
    a.chip === b.chip &&
    a.memory_gb === b.memory_gb &&
    a.python_version === b.python_version &&
    a.mlx_version === b.mlx_version
  );
}

class SystemStore {
  memory = $state<SystemMemory | null>(null);
  info = $state<SystemInfo | null>(null);
  loading = $state(false);
  error = $state<string | null>(null);

  constructor() {
    // Register memory polling with coordinator
    pollingCoordinator.register("system-memory", {
      interval: 30000, // 30 seconds
      minInterval: 5000, // Throttle: max 1 request per 5 seconds
      refreshFn: () => this.doRefreshMemory(),
    });
  }

  /**
   * Internal memory refresh - called by polling coordinator.
   * Only updates if data actually changed.
   */
  private async doRefreshMemory() {
    try {
      const newMemory = await systemApi.memory();

      // Only update if actually changed
      if (!memoryEqual(this.memory, newMemory)) {
        this.memory = newMemory;
      }
    } catch (e) {
      console.error("Failed to fetch memory:", e);
    }
  }

  /**
   * Public memory refresh - uses coordinator for deduplication.
   */
  async refreshMemory() {
    return pollingCoordinator.refresh("system-memory");
  }

  /**
   * Start automatic memory polling.
   */
  startMemoryPolling() {
    pollingCoordinator.start("system-memory");
  }

  /**
   * Stop automatic memory polling.
   */
  stopMemoryPolling() {
    pollingCoordinator.stop("system-memory");
  }

  /**
   * Refresh system info (not polled automatically - called on demand).
   * Only updates if data actually changed.
   */
  async refreshInfo() {
    this.loading = true;
    try {
      const newInfo = await systemApi.info();

      // Only update if actually changed
      if (!infoEqual(this.info, newInfo)) {
        this.info = newInfo;
      }
    } catch (e) {
      this.error =
        e instanceof Error ? e.message : "Failed to fetch system info";
    } finally {
      this.loading = false;
    }
  }

  /**
   * Refresh both memory and info.
   */
  async refresh() {
    await Promise.all([this.refreshMemory(), this.refreshInfo()]);
  }
}

export const systemStore = new SystemStore();
