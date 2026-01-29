<script lang="ts">
	import { settings } from '$lib/api/client';
	import type { BackendType, CloudCredential } from '$lib/api/types';
	import { Button, Input } from '$components/ui';
	import { Check, X, Loader2, RefreshCw, ChevronDown, ChevronUp, Trash2 } from 'lucide-svelte';

	interface Props {
		backendType: BackendType;
		existingCredential: CloudCredential | null;
		onSave: () => void;
		onDelete: () => void;
	}

	let { backendType, existingCredential, onSave, onDelete }: Props = $props();

	let apiKey = $state('');
	let baseUrl = $state(existingCredential?.base_url ?? '');
	let showAdvanced = $state(!!existingCredential?.base_url);
	let testing = $state(false);
	let saving = $state(false);
	let deleting = $state(false);
	let testResult = $state<'success' | 'error' | null>(null);
	let error = $state<string | null>(null);

	// Mask the API key for display
	const maskedKey = $derived(
		apiKey.length > 4 ? `****...${apiKey.slice(-4)}` : apiKey ? '****' : ''
	);

	// Show masked placeholder if credential exists
	const hasCredential = $derived(existingCredential !== null);
	const placeholder = $derived(
		hasCredential ? '****...saved (enter new key to update)' : 'Enter API key'
	);

	// Clear test result when key changes
	$effect(() => {
		if (apiKey) {
			testResult = null;
			error = null;
		}
	});

	async function handleSave() {
		if (!apiKey.trim()) {
			error = 'API key is required';
			return;
		}

		saving = true;
		error = null;
		testResult = null;

		try {
			// Step 1: Save credentials to database
			await settings.createProvider({
				backend_type: backendType,
				api_key: apiKey.trim(),
				base_url: baseUrl.trim() || undefined
			});

			// Step 2: Test the saved credentials
			try {
				await settings.testProvider(backendType);
				testResult = 'success';
			} catch {
				testResult = 'error';
				error = 'Saved but connection test failed - check your API key';
			}

			onSave();
			apiKey = ''; // Clear input after successful save
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to save credentials';
		} finally {
			saving = false;
		}
	}

	async function handleTest() {
		// Test already-saved credentials
		if (!existingCredential) return;

		testing = true;
		error = null;

		try {
			await settings.testProvider(backendType);
			testResult = 'success';
		} catch (e) {
			testResult = 'error';
			error = e instanceof Error ? e.message : 'Connection test failed';
		} finally {
			testing = false;
		}
	}

	async function handleDelete() {
		if (!existingCredential) return;

		deleting = true;
		error = null;

		try {
			await settings.deleteProvider(backendType);
			testResult = null;
			onDelete();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to delete credentials';
		} finally {
			deleting = false;
		}
	}
</script>

<div class="space-y-4">
	<!-- API Key Input -->
	<div class="space-y-2">
		<label for="{backendType}-api-key" class="text-sm font-medium text-foreground">
			API Key
		</label>
		<div class="relative">
			<Input
				id="{backendType}-api-key"
				type="password"
				bind:value={apiKey}
				{placeholder}
				class="pr-24"
			/>
			{#if apiKey}
				<span
					class="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-muted-foreground font-mono"
				>
					{maskedKey}
				</span>
			{/if}
		</div>
	</div>

	<!-- Advanced Settings Toggle -->
	<button
		type="button"
		class="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors"
		onclick={() => (showAdvanced = !showAdvanced)}
	>
		{#if showAdvanced}
			<ChevronUp class="h-4 w-4" />
		{:else}
			<ChevronDown class="h-4 w-4" />
		{/if}
		Advanced Settings
	</button>

	<!-- Base URL (Advanced) -->
	{#if showAdvanced}
		<div class="space-y-2">
			<label for="{backendType}-base-url" class="text-sm font-medium text-foreground">
				Base URL (optional)
			</label>
			<Input
				id="{backendType}-base-url"
				type="text"
				bind:value={baseUrl}
				placeholder={backendType === 'openai'
					? 'https://api.openai.com/v1'
					: 'https://api.anthropic.com'}
			/>
			<p class="text-xs text-muted-foreground">
				Override the default API endpoint (useful for proxies or custom deployments)
			</p>
		</div>
	{/if}

	<!-- Error Display -->
	{#if error}
		<div class="flex items-center gap-2 text-sm text-destructive">
			<X class="h-4 w-4" />
			{error}
		</div>
	{/if}

	<!-- Success Display -->
	{#if testResult === 'success' && !error}
		<div class="flex items-center gap-2 text-sm text-green-600 dark:text-green-400">
			<Check class="h-4 w-4" />
			Connection successful
		</div>
	{/if}

	<!-- Action Buttons -->
	<div class="flex items-center gap-2">
		<Button onclick={handleSave} disabled={saving || !apiKey.trim()} variant="default" size="sm">
			{#if saving}
				<Loader2 class="h-4 w-4 mr-2 animate-spin" />
				Saving...
			{:else}
				Save & Test
			{/if}
		</Button>

		{#if hasCredential}
			<Button onclick={handleTest} disabled={testing} variant="outline" size="sm">
				{#if testing}
					<Loader2 class="h-4 w-4 mr-2 animate-spin" />
					Testing...
				{:else}
					<RefreshCw class="h-4 w-4 mr-2" />
					Test Connection
				{/if}
			</Button>

			<Button onclick={handleDelete} disabled={deleting} variant="destructive" size="sm">
				{#if deleting}
					<Loader2 class="h-4 w-4 mr-2 animate-spin" />
				{:else}
					<Trash2 class="h-4 w-4 mr-2" />
				{/if}
				Delete
			</Button>
		{/if}
	</div>
</div>
