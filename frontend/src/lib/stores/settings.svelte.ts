/**
 * Settings store for tracking provider configuration state.
 * Uses Svelte 5 runes for reactivity.
 */
import type { BackendType, CloudCredential } from "$lib/api/types";
import { settings } from "$lib/api/client";

interface SettingsState {
	providers: CloudCredential[];
	loading: boolean;
	error: string | null;
}

function createSettingsStore() {
	const state = $state<SettingsState>({
		providers: [],
		loading: false,
		error: null,
	});

	// Derived state for configured provider types
	const configuredProviders = $derived.by(() => {
		return new Set(state.providers.map((p) => p.backend_type));
	});

	return {
		// Getters
		get providers() {
			return state.providers;
		},
		get loading() {
			return state.loading;
		},
		get error() {
			return state.error;
		},
		get configuredProviders() {
			return configuredProviders;
		},

		// Check if a specific provider is configured
		isProviderConfigured(type: BackendType): boolean {
			return configuredProviders.has(type);
		},

		// Check if any cloud provider is configured
		hasAnyCloudProvider(): boolean {
			return state.providers.some((p) => p.backend_type !== "local");
		},

		// Load providers from API
		async loadProviders(): Promise<void> {
			state.loading = true;
			state.error = null;
			try {
				state.providers = await settings.listProviders();
			} catch (e) {
				state.error = e instanceof Error ? e.message : "Failed to load providers";
			} finally {
				state.loading = false;
			}
		},

		// Set providers directly (useful after create/delete operations)
		setProviders(providers: CloudCredential[]): void {
			state.providers = providers;
		},

		// Add a single provider to the list
		addProvider(provider: CloudCredential): void {
			// Remove existing provider of same type if any (replace semantics)
			state.providers = state.providers.filter(
				(p) => p.backend_type !== provider.backend_type
			);
			state.providers = [...state.providers, provider];
		},

		// Remove a provider by backend type
		removeProvider(type: BackendType): void {
			state.providers = state.providers.filter((p) => p.backend_type !== type);
		},

		// Reset state
		reset(): void {
			state.providers = [];
			state.loading = false;
			state.error = null;
		},
	};
}

export const settingsStore = createSettingsStore();
