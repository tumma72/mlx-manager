/**
 * Polling Coordinator - Centralized polling management for all stores.
 *
 * This singleton service provides:
 * - Request deduplication: If a refresh is in-flight, returns the existing promise
 * - Throttling: Prevents rapid successive calls within minInterval
 * - Tab visibility: Pauses polling when tab is hidden, resumes when visible
 * - Central registration: Stores register their refresh functions
 */

export type PollingKey = "servers" | "profiles" | "system-memory";

interface PollingConfig {
  /** Polling interval in milliseconds */
  interval: number;
  /** Minimum time between requests (throttle) in milliseconds */
  minInterval: number;
  /** The refresh function to call */
  refreshFn: () => Promise<void>;
}

interface PollingState {
  config: PollingConfig;
  intervalId: ReturnType<typeof setInterval> | null;
  paused: boolean;
}

class PollingCoordinator {
  // Track in-flight requests to prevent duplicates
  private inFlight = new Map<PollingKey, Promise<void>>();

  // Last successful refresh timestamp per key
  private lastRefresh = new Map<PollingKey, number>();

  // Registered polling configs and state
  private pollingState = new Map<PollingKey, PollingState>();

  // Tab visibility state
  private isVisible = true;

  // Global pause (e.g., during heavy operations)
  private globalPause = false;

  // Track if we've set up visibility listener
  private visibilityListenerAttached = false;

  constructor() {
    this.setupVisibilityListener();
  }

  /**
   * Set up document visibility change listener.
   * Only runs on client-side.
   */
  private setupVisibilityListener() {
    if (typeof document === "undefined") return;
    if (this.visibilityListenerAttached) return;

    this.isVisible = document.visibilityState === "visible";

    document.addEventListener("visibilitychange", () => {
      const wasVisible = this.isVisible;
      this.isVisible = document.visibilityState === "visible";

      if (this.isVisible && !wasVisible) {
        // Tab became visible - resume polling and do immediate refresh
        this.resumeAll();
      } else if (!this.isVisible && wasVisible) {
        // Tab became hidden - pause all polling
        this.pauseAll();
      }
    });

    this.visibilityListenerAttached = true;
  }

  /**
   * Register a polling configuration for a key.
   * Does not start polling automatically.
   */
  register(key: PollingKey, config: PollingConfig) {
    // If already registered, update config but preserve state
    const existing = this.pollingState.get(key);
    if (existing) {
      existing.config = config;
    } else {
      this.pollingState.set(key, {
        config,
        intervalId: null,
        paused: false,
      });
    }
  }

  /**
   * Start polling for a specific key.
   * If already polling, restarts with current config.
   */
  start(key: PollingKey) {
    const state = this.pollingState.get(key);
    if (!state) {
      console.warn(`[PollingCoordinator] No config registered for ${key}`);
      return;
    }

    // Clear existing interval if any
    this.stop(key);

    // Mark as not paused
    state.paused = false;

    // Do immediate refresh
    this.refresh(key);

    // Set up interval
    state.intervalId = setInterval(() => {
      if (this.shouldPoll(key)) {
        this.refresh(key);
      }
    }, state.config.interval);
  }

  /**
   * Stop polling for a specific key.
   */
  stop(key: PollingKey) {
    const state = this.pollingState.get(key);
    if (state?.intervalId) {
      clearInterval(state.intervalId);
      state.intervalId = null;
    }
  }

  /**
   * Request a refresh for a key.
   * Handles deduplication (returns existing promise if in-flight)
   * and throttling (skips if last refresh was too recent).
   */
  async refresh(key: PollingKey): Promise<void> {
    const state = this.pollingState.get(key);
    if (!state) {
      console.warn(`[PollingCoordinator] No config registered for ${key}`);
      return;
    }

    // Check if request is already in flight (deduplication)
    const existing = this.inFlight.get(key);
    if (existing) {
      return existing;
    }

    // Check throttle - don't refresh if last refresh was too recent
    const lastTime = this.lastRefresh.get(key) ?? 0;
    const now = Date.now();
    if (now - lastTime < state.config.minInterval) {
      return;
    }

    // Create and track the request
    const request = state.config
      .refreshFn()
      .catch((error) => {
        console.error(`[PollingCoordinator] Refresh failed for ${key}:`, error);
      })
      .finally(() => {
        this.inFlight.delete(key);
        this.lastRefresh.set(key, Date.now());
      });

    this.inFlight.set(key, request);
    return request;
  }

  /**
   * Check if we should poll (considers visibility, global pause, key pause)
   */
  private shouldPoll(key: PollingKey): boolean {
    if (!this.isVisible) return false;
    if (this.globalPause) return false;

    const state = this.pollingState.get(key);
    if (state?.paused) return false;

    return true;
  }

  /**
   * Pause all polling (e.g., when tab hidden)
   */
  private pauseAll() {
    for (const [, state] of this.pollingState) {
      if (state.intervalId) {
        clearInterval(state.intervalId);
        state.intervalId = null;
      }
    }
  }

  /**
   * Resume all polling that wasn't manually paused
   */
  private resumeAll() {
    for (const [key, state] of this.pollingState) {
      if (!state.paused && !state.intervalId) {
        // Do immediate refresh when returning to tab
        this.refresh(key);

        // Restart interval
        state.intervalId = setInterval(() => {
          if (this.shouldPoll(key)) {
            this.refresh(key);
          }
        }, state.config.interval);
      }
    }
  }

  /**
   * Check if a specific poll is currently in-flight
   */
  isRefreshing(key: PollingKey): boolean {
    return this.inFlight.has(key);
  }

  /**
   * Get time since last refresh in milliseconds
   */
  getTimeSinceRefresh(key: PollingKey): number {
    const lastTime = this.lastRefresh.get(key);
    return lastTime ? Date.now() - lastTime : Infinity;
  }

  /**
   * Set global pause state
   */
  setGlobalPause(paused: boolean) {
    this.globalPause = paused;
    if (paused) {
      this.pauseAll();
    } else if (this.isVisible) {
      this.resumeAll();
    }
  }

  /**
   * Pause polling for a specific key
   */
  pause(key: PollingKey) {
    const state = this.pollingState.get(key);
    if (state) {
      state.paused = true;
      this.stop(key);
    }
  }

  /**
   * Resume polling for a specific key
   */
  resume(key: PollingKey) {
    const state = this.pollingState.get(key);
    if (state) {
      state.paused = false;
      if (this.isVisible && !this.globalPause) {
        this.start(key);
      }
    }
  }

  /**
   * Check if polling is active for a key
   */
  isPolling(key: PollingKey): boolean {
    const state = this.pollingState.get(key);
    return state?.intervalId != null;
  }

  /**
   * Cleanup - call on app unmount
   */
  destroy() {
    for (const [key] of this.pollingState) {
      this.stop(key);
    }
    this.pollingState.clear();
    this.inFlight.clear();
    this.lastRefresh.clear();
  }
}

// Singleton instance
export const pollingCoordinator = new PollingCoordinator();
