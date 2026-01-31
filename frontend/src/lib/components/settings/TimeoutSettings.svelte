<script lang="ts">
	import { onMount } from 'svelte';
	import { Save, RotateCcw } from 'lucide-svelte';
	import { Button } from '$lib/components/ui/button';
	import { settings } from '$lib/api/client';
	import type { TimeoutSettings } from '$lib/api/types';

	// State
	let currentSettings = $state<TimeoutSettings | null>(null);
	let loading = $state(true);
	let saving = $state(false);
	let error = $state<string | null>(null);

	// Editable values (in seconds)
	let chatTimeout = $state(900);
	let completionsTimeout = $state(600);
	let embeddingsTimeout = $state(120);

	// Track if values changed
	let hasChanges = $derived(
		currentSettings !== null &&
			(chatTimeout !== currentSettings.chat_seconds ||
				completionsTimeout !== currentSettings.completions_seconds ||
				embeddingsTimeout !== currentSettings.embeddings_seconds)
	);

	// Format seconds to human readable
	function formatSeconds(seconds: number): string {
		if (seconds < 60) return `${seconds}s`;
		if (seconds < 3600) return `${Math.round(seconds / 60)} min`;
		return `${(seconds / 3600).toFixed(1)} hr`;
	}

	// Load settings
	async function loadSettings() {
		loading = true;
		error = null;
		try {
			currentSettings = await settings.getTimeoutSettings();
			chatTimeout = currentSettings.chat_seconds;
			completionsTimeout = currentSettings.completions_seconds;
			embeddingsTimeout = currentSettings.embeddings_seconds;
		} catch (e) {
			error = 'Failed to load timeout settings';
			console.error(e);
		} finally {
			loading = false;
		}
	}

	// Save settings
	async function saveSettings() {
		saving = true;
		error = null;
		try {
			currentSettings = await settings.updateTimeoutSettings({
				chat_seconds: chatTimeout,
				completions_seconds: completionsTimeout,
				embeddings_seconds: embeddingsTimeout
			});
		} catch (e) {
			error = 'Failed to save timeout settings';
			console.error(e);
		} finally {
			saving = false;
		}
	}

	// Reset to defaults
	function resetToDefaults() {
		chatTimeout = 900;
		completionsTimeout = 600;
		embeddingsTimeout = 120;
	}

	onMount(() => {
		loadSettings();
	});
</script>

<div class="space-y-4">
	{#if loading}
		<div class="text-muted-foreground">Loading timeout settings...</div>
	{:else if error}
		<div class="text-red-500">{error}</div>
	{:else}
		<div class="grid gap-4">
			<!-- Chat Completions -->
			<div class="flex items-center gap-4">
				<div class="flex-1">
					<label class="block text-sm font-medium mb-1">
						Chat Completions
						<span class="text-muted-foreground font-normal ml-1"> (/v1/chat/completions) </span>
					</label>
					<div class="flex items-center gap-2">
						<input
							type="range"
							min="60"
							max="7200"
							step="60"
							class="flex-1"
							bind:value={chatTimeout}
						/>
						<input
							type="number"
							min="60"
							max="7200"
							step="60"
							class="w-24 px-2 py-1 border rounded text-sm"
							bind:value={chatTimeout}
						/>
						<span class="text-sm text-muted-foreground w-16">
							{formatSeconds(chatTimeout)}
						</span>
					</div>
					<p class="text-xs text-muted-foreground mt-1">
						Default: 15 minutes. For long conversations and large models.
					</p>
				</div>
			</div>

			<!-- Completions -->
			<div class="flex items-center gap-4">
				<div class="flex-1">
					<label class="block text-sm font-medium mb-1">
						Text Completions
						<span class="text-muted-foreground font-normal ml-1"> (/v1/completions) </span>
					</label>
					<div class="flex items-center gap-2">
						<input
							type="range"
							min="60"
							max="7200"
							step="60"
							class="flex-1"
							bind:value={completionsTimeout}
						/>
						<input
							type="number"
							min="60"
							max="7200"
							step="60"
							class="w-24 px-2 py-1 border rounded text-sm"
							bind:value={completionsTimeout}
						/>
						<span class="text-sm text-muted-foreground w-16">
							{formatSeconds(completionsTimeout)}
						</span>
					</div>
					<p class="text-xs text-muted-foreground mt-1">
						Default: 10 minutes. For legacy completion requests.
					</p>
				</div>
			</div>

			<!-- Embeddings -->
			<div class="flex items-center gap-4">
				<div class="flex-1">
					<label class="block text-sm font-medium mb-1">
						Embeddings
						<span class="text-muted-foreground font-normal ml-1"> (/v1/embeddings) </span>
					</label>
					<div class="flex items-center gap-2">
						<input
							type="range"
							min="30"
							max="600"
							step="30"
							class="flex-1"
							bind:value={embeddingsTimeout}
						/>
						<input
							type="number"
							min="30"
							max="600"
							step="30"
							class="w-24 px-2 py-1 border rounded text-sm"
							bind:value={embeddingsTimeout}
						/>
						<span class="text-sm text-muted-foreground w-16">
							{formatSeconds(embeddingsTimeout)}
						</span>
					</div>
					<p class="text-xs text-muted-foreground mt-1">
						Default: 2 minutes. Embeddings are fast, short timeout is appropriate.
					</p>
				</div>
			</div>
		</div>

		<!-- Actions -->
		<div class="flex items-center gap-2 pt-4 border-t">
			<Button variant="default" size="sm" onclick={saveSettings} disabled={!hasChanges || saving}>
				<Save class="h-4 w-4 mr-1" />
				{saving ? 'Saving...' : 'Save Changes'}
			</Button>
			<Button variant="outline" size="sm" onclick={resetToDefaults} disabled={saving}>
				<RotateCcw class="h-4 w-4 mr-1" />
				Reset to Defaults
			</Button>
			{#if hasChanges}
				<span class="text-sm text-muted-foreground ml-2"> Unsaved changes </span>
			{/if}
		</div>

		<p class="text-xs text-muted-foreground">
			Note: Changes are stored but may require restarting the MLX Server to take effect.
		</p>
	{/if}
</div>
