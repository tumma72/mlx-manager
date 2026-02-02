<script lang="ts">
	import { settings } from '$lib/api/client';
	import type { ApiType, BackendType, CloudCredential } from '$lib/api/types';
	import { Button, Input, ConfirmDialog } from '$components/ui';
	import { Check, X, Loader2, RefreshCw, ChevronDown, ChevronUp, Trash2 } from 'lucide-svelte';

	// Known provider configurations
	const PROVIDER_CONFIGS: Record<
		string,
		{ label: string; backendType: BackendType; apiType: ApiType; defaultBaseUrl: string }
	> = {
		openai: {
			label: 'OpenAI',
			backendType: 'openai',
			apiType: 'openai',
			defaultBaseUrl: 'https://api.openai.com'
		},
		anthropic: {
			label: 'Anthropic',
			backendType: 'anthropic',
			apiType: 'anthropic',
			defaultBaseUrl: 'https://api.anthropic.com'
		},
		together: {
			label: 'Together AI',
			backendType: 'together',
			apiType: 'openai',
			defaultBaseUrl: 'https://api.together.xyz'
		},
		groq: {
			label: 'Groq',
			backendType: 'groq',
			apiType: 'openai',
			defaultBaseUrl: 'https://api.groq.com/openai'
		},
		fireworks: {
			label: 'Fireworks AI',
			backendType: 'fireworks',
			apiType: 'openai',
			defaultBaseUrl: 'https://api.fireworks.ai/inference'
		},
		mistral: {
			label: 'Mistral AI',
			backendType: 'mistral',
			apiType: 'openai',
			defaultBaseUrl: 'https://api.mistral.ai'
		},
		deepseek: {
			label: 'DeepSeek',
			backendType: 'deepseek',
			apiType: 'openai',
			defaultBaseUrl: 'https://api.deepseek.com'
		},
		openai_compatible: {
			label: 'Custom (OpenAI-compatible)',
			backendType: 'openai_compatible',
			apiType: 'openai',
			defaultBaseUrl: ''
		},
		anthropic_compatible: {
			label: 'Custom (Anthropic-compatible)',
			backendType: 'anthropic_compatible',
			apiType: 'anthropic',
			defaultBaseUrl: ''
		}
	};

	interface Props {
		backendType: BackendType;
		existingCredential: CloudCredential | null;
		onSave: () => void;
		onDelete: () => void;
	}

	let { backendType, existingCredential, onSave, onDelete }: Props = $props();

	// Get provider config for this backend type
	const providerConfig = $derived(
		Object.values(PROVIDER_CONFIGS).find((p) => p.backendType === backendType)
	);

	let apiKey = $state('');
	let baseUrl = $state('');
	let customName = $state('');
	let showAdvanced = $state(false);
	let testing = $state(false);

	// Sync state with existingCredential prop when it changes
	$effect(() => {
		baseUrl = existingCredential?.base_url ?? providerConfig?.defaultBaseUrl ?? '';
		customName = existingCredential?.name ?? '';
		showAdvanced =
			!!existingCredential?.base_url ||
			backendType === 'openai_compatible' ||
			backendType === 'anthropic_compatible';
	});
	let saving = $state(false);
	let deleting = $state(false);
	let testResult = $state<'success' | 'error' | null>(null);
	let error = $state<string | null>(null);

	// Delete confirmation dialog state
	let deleteDialogOpen = $state(false);

	// Mask the API key for display
	const maskedKey = $derived(
		apiKey.length > 4 ? `****...${apiKey.slice(-4)}` : apiKey ? '****' : ''
	);

	// Show masked placeholder if credential exists
	const hasCredential = $derived(existingCredential !== null);
	const placeholder = $derived(
		hasCredential ? '****...saved (enter new key to update)' : 'Enter API key'
	);

	// Check if base URL is required (custom providers)
	const isCustomProvider = $derived(
		backendType === 'openai_compatible' || backendType === 'anthropic_compatible'
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

		if (isCustomProvider && !baseUrl.trim()) {
			error = 'Base URL is required for custom providers';
			return;
		}

		saving = true;
		error = null;
		testResult = null;

		try {
			// Step 1: Save credentials to database
			await settings.createProvider({
				backend_type: backendType,
				api_type: providerConfig?.apiType ?? 'openai',
				name: customName.trim() || providerConfig?.label || backendType,
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

	// Request deletion - show confirmation dialog
	function requestDelete() {
		deleteDialogOpen = true;
	}

	async function confirmDelete() {
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
	<!-- Provider Name (for custom providers) -->
	{#if isCustomProvider}
		<div class="space-y-2">
			<label for="{backendType}-name" class="text-sm font-medium text-foreground">
				Provider Name
			</label>
			<Input
				id="{backendType}-name"
				type="text"
				bind:value={customName}
				placeholder="e.g., My OpenAI Proxy"
			/>
			<p class="text-xs text-muted-foreground">A display name for this provider</p>
		</div>
	{/if}

	<!-- API Key Input -->
	<div class="space-y-2">
		<label for="{backendType}-api-key" class="text-sm font-medium text-foreground"> API Key </label>
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

	<!-- Advanced Settings Toggle (not for custom providers - they always show base_url) -->
	{#if !isCustomProvider}
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
	{/if}

	<!-- Base URL (Advanced or always for custom) -->
	{#if showAdvanced || isCustomProvider}
		<div class="space-y-2">
			<label for="{backendType}-base-url" class="text-sm font-medium text-foreground">
				Base URL {isCustomProvider ? '' : '(optional)'}
			</label>
			<Input
				id="{backendType}-base-url"
				type="text"
				bind:value={baseUrl}
				placeholder={providerConfig?.defaultBaseUrl || 'https://api.example.com'}
			/>
			<p class="text-xs text-muted-foreground">
				{#if isCustomProvider}
					The base URL for the API endpoint
				{:else}
					Override the default API endpoint (useful for proxies or custom deployments)
				{/if}
			</p>
		</div>
	{/if}

	<!-- API Type Info (for custom providers) -->
	{#if isCustomProvider}
		<div class="text-sm text-muted-foreground">
			API Protocol: <span class="font-medium text-foreground"
				>{providerConfig?.apiType === 'anthropic' ? 'Anthropic' : 'OpenAI'}-compatible</span
			>
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

			<Button onclick={requestDelete} disabled={deleting} variant="destructive" size="sm">
				{#if deleting}
					<Loader2 class="h-4 w-4 mr-2 animate-spin" />
				{:else}
					<Trash2 class="h-4 w-4 mr-2" />
				{/if}
				Delete
			</Button>
		{/if}
	</div>

	<ConfirmDialog
		bind:open={deleteDialogOpen}
		title="Delete Provider Credentials"
		description="Are you sure you want to delete the {providerConfig?.label ?? backendType} credentials? You will need to re-enter your API key to use this provider again."
		confirmLabel="Delete"
		variant="destructive"
		onConfirm={confirmDelete}
		onCancel={() => {}}
	/>
</div>
