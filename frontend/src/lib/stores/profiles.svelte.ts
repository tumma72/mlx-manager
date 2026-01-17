/**
 * Profile state management using Svelte 5 runes.
 *
 * Refactored to use:
 * - In-place array reconciliation (prevents unnecessary re-renders)
 * - Polling coordinator for centralized refresh management
 */

import { profiles as profilesApi } from "$api";
import type {
  ServerProfile,
  ServerProfileCreate,
  ServerProfileUpdate,
} from "$api";
import { reconcileArray } from "$lib/utils/reconcile";
import { pollingCoordinator } from "$lib/services";

/**
 * Custom equality function for ServerProfile.
 * Compares all fields that affect the UI.
 */
function profilesEqual(a: ServerProfile, b: ServerProfile): boolean {
  return (
    a.id === b.id &&
    a.name === b.name &&
    a.description === b.description &&
    a.model_path === b.model_path &&
    a.model_type === b.model_type &&
    a.port === b.port &&
    a.host === b.host &&
    a.context_length === b.context_length &&
    a.max_concurrency === b.max_concurrency &&
    a.queue_timeout === b.queue_timeout &&
    a.queue_size === b.queue_size &&
    a.tool_call_parser === b.tool_call_parser &&
    a.reasoning_parser === b.reasoning_parser &&
    a.enable_auto_tool_choice === b.enable_auto_tool_choice &&
    a.trust_remote_code === b.trust_remote_code &&
    a.chat_template_file === b.chat_template_file &&
    a.log_level === b.log_level &&
    a.log_file === b.log_file &&
    a.no_log_file === b.no_log_file &&
    a.auto_start === b.auto_start &&
    a.launchd_installed === b.launchd_installed &&
    a.created_at === b.created_at &&
    a.updated_at === b.updated_at
  );
}

class ProfileStore {
  // Single array instance - mutated in-place by reconcileArray
  profiles = $state<ServerProfile[]>([]);
  loading = $state(false);
  error = $state<string | null>(null);

  // Track if polling has been initialized
  private pollingInitialized = false;

  // Track if initial load has completed (prevents loading flicker on subsequent polls)
  private initialLoadComplete = false;

  constructor() {
    // Register with polling coordinator
    pollingCoordinator.register("profiles", {
      interval: 10000, // 10 seconds (profiles change less frequently)
      minInterval: 1000, // Throttle: max 1 request per second
      refreshFn: () => this.doRefresh(),
    });
  }

  /**
   * Internal refresh - called by polling coordinator.
   * Uses reconcileArray to update in-place.
   *
   * IMPORTANT: Only sets loading=true on initial load (before first successful fetch).
   * Background polls should not toggle loading state, as this triggers
   * unnecessary re-renders across all components watching the store.
   */
  private async doRefresh() {
    // Only show loading on initial load, not background polls
    const isInitialLoad = !this.initialLoadComplete;
    if (isInitialLoad) {
      this.loading = true;
    }

    try {
      const newProfiles = await profilesApi.list();

      // Reconcile instead of replace - only updates what changed
      reconcileArray(this.profiles, newProfiles, {
        getKey: (p) => p.id,
        isEqual: profilesEqual,
      });

      // Mark initial load complete on first successful fetch
      this.initialLoadComplete = true;

      // Clear any previous error on successful refresh
      if (this.error) {
        this.error = null;
      }
    } catch (e) {
      this.error = e instanceof Error ? e.message : "Failed to fetch profiles";
    } finally {
      if (isInitialLoad) {
        this.loading = false;
      }
    }
  }

  /**
   * Public refresh - uses coordinator for deduplication.
   */
  async refresh() {
    return pollingCoordinator.refresh("profiles");
  }

  /**
   * Start automatic polling - call from root layout.
   */
  startPolling() {
    if (!this.pollingInitialized) {
      pollingCoordinator.start("profiles");
      this.pollingInitialized = true;
    }
  }

  /**
   * Stop automatic polling.
   */
  stopPolling() {
    pollingCoordinator.stop("profiles");
    this.pollingInitialized = false;
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
