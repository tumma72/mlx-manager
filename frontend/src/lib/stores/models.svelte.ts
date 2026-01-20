/**
 * Model configuration state management using Svelte 5 runes.
 *
 * Lazy-loads model characteristics (config.json) on demand and caches results.
 */

/* eslint-disable svelte/prefer-svelte-reactivity -- Using Map with reassignment for reactivity */
import { models as modelsApi } from "$api";
import type { ModelCharacteristics } from "$api";

export interface ConfigState {
  characteristics: ModelCharacteristics | null;
  loading: boolean;
  error: string | null;
}

class ModelConfigStore {
  private configs = $state<Map<string, ConfigState>>(new Map());

  /**
   * Get cached config state for a model (if any).
   */
  getConfig(modelId: string): ConfigState | undefined {
    return this.configs.get(modelId);
  }

  /**
   * Fetch config for a model (lazy load).
   * Does nothing if already loaded or currently loading.
   */
  async fetchConfig(modelId: string): Promise<void> {
    // Don't refetch if already loaded or loading
    const existing = this.configs.get(modelId);
    if (existing?.characteristics || existing?.loading) return;

    // Set loading state (create new Map for reactivity)
    this.configs = new Map(this.configs).set(modelId, {
      characteristics: null,
      loading: true,
      error: null,
    });

    try {
      const characteristics = await modelsApi.getConfig(modelId);
      this.configs = new Map(this.configs).set(modelId, {
        characteristics,
        loading: false,
        error: null,
      });
    } catch (e) {
      this.configs = new Map(this.configs).set(modelId, {
        characteristics: null,
        loading: false,
        error: e instanceof Error ? e.message : "Failed to load config",
      });
    }
  }

  /**
   * Clear cached config for a specific model.
   */
  clearConfig(modelId: string): void {
    const newMap = new Map(this.configs);
    newMap.delete(modelId);
    this.configs = newMap;
  }

  /**
   * Clear all cached configs.
   */
  clearAll(): void {
    this.configs = new Map();
  }
}

export const modelConfigStore = new ModelConfigStore();
