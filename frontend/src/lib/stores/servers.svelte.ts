/**
 * Server state management using Svelte 5 runes.
 *
 * Refactored to use:
 * - In-place array reconciliation (prevents unnecessary re-renders)
 * - Svelte 5 reactive Map/Set for proper change detection
 * - Polling coordinator for centralized refresh management
 * - HMR state preservation for development experience
 */

import { servers as serversApi } from "$api";
import type { RunningServer } from "$api";
import { reconcileArray } from "$lib/utils/reconcile";
import { pollingCoordinator } from "$lib/services";
import { SvelteMap, SvelteSet } from "svelte/reactivity";

export interface FailedServer {
  error: string;
  details: string | null;
  detailsOpen: boolean;
}

/**
 * Custom equality function for RunningServer.
 * Allows small differences in frequently-changing fields (uptime, memory)
 * to avoid constant re-renders.
 */
function serversEqual(a: RunningServer, b: RunningServer): boolean {
  return (
    a.profile_id === b.profile_id &&
    a.profile_name === b.profile_name &&
    a.pid === b.pid &&
    a.port === b.port &&
    a.health_status === b.health_status &&
    // Allow 2-second drift in uptime to reduce update frequency
    Math.abs(a.uptime_seconds - b.uptime_seconds) < 2 &&
    // Allow 1MB drift in memory to reduce update frequency
    Math.abs(a.memory_mb - b.memory_mb) < 1
  );
}

// HMR state preservation - stores state outside the class instance
// so it survives module reloads during development
interface HmrState {
  servers: RunningServer[];
  startingProfiles: SvelteSet<number>;
  failedProfiles: SvelteMap<number, FailedServer>;
  restartingProfiles: SvelteSet<number>;
  // Track which profiles have active polling loops (prevents duplicates)
  pollingProfiles: Set<number>;
}

// Initialize HMR state container if it doesn't exist
const hmrKey = "__serverStoreHmrState";
function getOrCreateHmrState(): HmrState {
  if (typeof window !== "undefined") {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const win = window as any;
    if (!win[hmrKey]) {
      win[hmrKey] = {
        servers: [],
        startingProfiles: new SvelteSet<number>(),
        failedProfiles: new SvelteMap<number, FailedServer>(),
        restartingProfiles: new SvelteSet<number>(),
        pollingProfiles: new Set<number>(),
      };
    }
    return win[hmrKey];
  }
  // SSR fallback
  return {
    servers: [],
    startingProfiles: new SvelteSet<number>(),
    failedProfiles: new SvelteMap<number, FailedServer>(),
    restartingProfiles: new SvelteSet<number>(),
    pollingProfiles: new Set<number>(),
  };
}

// Get the preserved state (same instance across HMR)
const hmrState = getOrCreateHmrState();

class ServerStore {
  // Single array instance - mutated in-place by reconcileArray
  // Preserved across HMR via window state
  servers = $state<RunningServer[]>(hmrState.servers);
  loading = $state(false);
  error = $state<string | null>(null);

  // Use SvelteSet/SvelteMap for proper reactivity with .has()/.get() methods
  // These are preserved across HMR
  startingProfiles = hmrState.startingProfiles;
  failedProfiles = hmrState.failedProfiles;
  restartingProfiles = hmrState.restartingProfiles;

  // Track which profiles have active polling loops (prevents duplicate loops)
  // This is NOT reactive - just used for coordination
  private pollingProfiles = hmrState.pollingProfiles;

  // Track if polling has been initialized
  private pollingInitialized = false;

  // Track if initial load has completed (prevents loading flicker on subsequent polls)
  private initialLoadComplete = false;

  constructor() {
    // Register with polling coordinator
    pollingCoordinator.register("servers", {
      interval: 5000, // 5 seconds
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
      const newServers = await serversApi.list();

      // Reconcile instead of replace - only updates what changed
      reconcileArray(this.servers, newServers, {
        getKey: (s) => s.profile_id,
        isEqual: serversEqual,
      });

      // Mark initial load complete on first successful fetch
      this.initialLoadComplete = true;

      // Clear any previous error on successful refresh
      if (this.error) {
        this.error = null;
      }
    } catch (e) {
      this.error = e instanceof Error ? e.message : "Failed to fetch servers";
    } finally {
      if (isInitialLoad) {
        this.loading = false;
      }
    }
  }

  /**
   * Public refresh - uses coordinator for deduplication.
   * Multiple callers get the same promise if a refresh is in-flight.
   */
  async refresh() {
    return pollingCoordinator.refresh("servers");
  }

  /**
   * Start automatic polling - call from root layout.
   */
  startPolling() {
    if (!this.pollingInitialized) {
      pollingCoordinator.start("servers");
      this.pollingInitialized = true;
    }
  }

  /**
   * Stop automatic polling.
   */
  stopPolling() {
    pollingCoordinator.stop("servers");
    this.pollingInitialized = false;
  }

  async start(profileId: number) {
    // Direct mutation - Svelte 5 proxies detect this
    this.startingProfiles.add(profileId);
    this.failedProfiles.delete(profileId);
    await serversApi.start(profileId);
    // Don't refresh yet - let ProfileCard handle health checking
  }

  async stop(profileId: number, force = false) {
    // Direct mutation
    this.startingProfiles.delete(profileId);
    this.failedProfiles.delete(profileId);
    await serversApi.stop(profileId, force);
    await this.refresh();
  }

  async restart(profileId: number) {
    // Direct mutation - mark as restarting (keeps tile mounted)
    this.restartingProfiles.add(profileId);
    this.failedProfiles.delete(profileId);
    await serversApi.restart(profileId);
    // After backend confirms stop, transition to starting state
    this.restartingProfiles.delete(profileId);
    this.startingProfiles.add(profileId);
    // Don't refresh yet - let ProfileCard handle health checking
  }

  /**
   * Mark startup as complete with success.
   */
  markStartupSuccess(profileId: number) {
    // Direct mutation
    this.startingProfiles.delete(profileId);
    this.failedProfiles.delete(profileId);
    this.restartingProfiles.delete(profileId);
    // Use coordinator for deduped refresh
    this.refresh();
  }

  /**
   * Mark startup as failed with error details.
   */
  markStartupFailed(
    profileId: number,
    error: string,
    details: string | null = null,
  ) {
    console.log(
      `[ServerStore] markStartupFailed called for profile ${profileId}`,
    );
    console.log(`[ServerStore] Error: ${error}`);
    console.log(
      `[ServerStore] Before - startingProfiles.has(${profileId}):`,
      this.startingProfiles.has(profileId),
    );
    console.log(
      `[ServerStore] Before - failedProfiles.has(${profileId}):`,
      this.failedProfiles.has(profileId),
    );

    // Direct mutation on SvelteSet/SvelteMap
    this.startingProfiles.delete(profileId);
    this.restartingProfiles.delete(profileId);

    // Preserve detailsOpen state if already failed
    const existing = this.failedProfiles.get(profileId);
    const detailsOpen = existing?.detailsOpen ?? false;
    this.failedProfiles.set(profileId, { error, details, detailsOpen });

    console.log(
      `[ServerStore] After - startingProfiles.has(${profileId}):`,
      this.startingProfiles.has(profileId),
    );
    console.log(
      `[ServerStore] After - failedProfiles.has(${profileId}):`,
      this.failedProfiles.has(profileId),
    );
    console.log(
      `[ServerStore] failedProfiles size:`,
      this.failedProfiles.size,
    );
    console.log(
      `[ServerStore] failedProfiles.get(${profileId}):`,
      this.failedProfiles.get(profileId),
    );

    // Use coordinator for deduped refresh
    this.refresh();
  }

  /**
   * Toggle error details open/closed state.
   */
  toggleDetailsOpen(profileId: number) {
    const failure = this.failedProfiles.get(profileId);
    if (failure) {
      // Direct mutation - update in-place
      this.failedProfiles.set(profileId, {
        ...failure,
        detailsOpen: !failure.detailsOpen,
      });
    }
  }

  /**
   * Clear failure state (e.g., when user dismisses error).
   */
  clearFailure(profileId: number) {
    // Direct mutation
    this.failedProfiles.delete(profileId);
  }

  /**
   * A server is "running" only if it's in the backend server list
   * AND not still starting (waiting for model to load).
   */
  isRunning(profileId: number): boolean {
    return (
      this.servers.some((s) => s.profile_id === profileId) &&
      !this.startingProfiles.has(profileId)
    );
  }

  /**
   * Check if a profile is currently starting (waiting for model to load).
   */
  isStarting(profileId: number): boolean {
    return this.startingProfiles.has(profileId);
  }

  /**
   * Check if a profile is currently restarting.
   */
  isRestarting(profileId: number): boolean {
    return this.restartingProfiles.has(profileId);
  }

  /**
   * Check if a profile failed to start.
   */
  isFailed(profileId: number): boolean {
    return this.failedProfiles.has(profileId);
  }

  /**
   * Get failure details for a profile.
   */
  getFailure(profileId: number): FailedServer | undefined {
    return this.failedProfiles.get(profileId);
  }

  /**
   * Get the running server info for a profile.
   */
  getServer(profileId: number): RunningServer | undefined {
    return this.servers.find((s) => s.profile_id === profileId);
  }

  /**
   * Check if a profile has an active polling loop.
   * Used to prevent duplicate polling loops when components are recreated.
   */
  isProfilePolling(profileId: number): boolean {
    return this.pollingProfiles.has(profileId);
  }

  /**
   * Mark a profile as having an active polling loop.
   * Call this when starting a polling loop.
   */
  startProfilePolling(profileId: number): boolean {
    if (this.pollingProfiles.has(profileId)) {
      console.log(
        `[ServerStore] Profile ${profileId} already has active polling, skipping`,
      );
      return false; // Already polling
    }
    this.pollingProfiles.add(profileId);
    console.log(`[ServerStore] Started polling for profile ${profileId}`);
    return true;
  }

  /**
   * Mark a profile as no longer having an active polling loop.
   * Call this when a polling loop ends (success, failure, or stop).
   */
  stopProfilePolling(profileId: number) {
    this.pollingProfiles.delete(profileId);
    console.log(`[ServerStore] Stopped polling for profile ${profileId}`);
  }
}

export const serverStore = new ServerStore();
