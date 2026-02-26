<script lang="ts">
	import { onMount } from 'svelte';
	import { Save, Trash2, RefreshCw, Loader2, Check, X } from 'lucide-svelte';
	import { Button, Input } from '$components/ui';
	import { settings } from '$lib/api/client';

	let configured = $state(false);
	let loading = $state(true);
	let saving = $state(false);
	let testing = $state(false);
	let deleting = $state(false);
	let error = $state<string | null>(null);
	let testResult = $state<{ success: boolean; username?: string } | null>(null);

	let tokenInput = $state('');

	const placeholder = $derived(
		configured ? '****...saved (enter new token to update)' : 'hf_xxxxxxxxxxxxxxxxxxxxxxxxx'
	);

	// Clear feedback when input changes
	$effect(() => {
		if (tokenInput) {
			testResult = null;
			error = null;
		}
	});

	async function loadStatus() {
		loading = true;
		error = null;
		try {
			const status = await settings.getHuggingFaceStatus();
			configured = status.configured;
		} catch (e) {
			error = 'Failed to load HuggingFace token status';
			console.error(e);
		} finally {
			loading = false;
		}
	}

	async function handleSave() {
		if (!tokenInput.trim()) {
			error = 'Token is required';
			return;
		}

		saving = true;
		error = null;
		testResult = null;

		try {
			await settings.saveHuggingFaceToken(tokenInput.trim());
			configured = true;
			tokenInput = '';

			// Auto-test after save
			try {
				const result = await settings.testHuggingFaceToken();
				testResult = result;
			} catch {
				testResult = null;
				error = 'Saved but connection test failed - check your token';
			}
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to save token';
		} finally {
			saving = false;
		}
	}

	async function handleTest() {
		testing = true;
		error = null;
		testResult = null;

		try {
			const result = await settings.testHuggingFaceToken();
			testResult = result;
		} catch (e) {
			testResult = null;
			error = e instanceof Error ? e.message : 'Connection test failed';
		} finally {
			testing = false;
		}
	}

	async function handleDelete() {
		deleting = true;
		error = null;
		testResult = null;

		try {
			await settings.deleteHuggingFaceToken();
			configured = false;
			tokenInput = '';
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to remove token';
		} finally {
			deleting = false;
		}
	}

	onMount(() => {
		loadStatus();
	});
</script>

<div class="space-y-4">
	{#if loading}
		<div class="text-muted-foreground">Loading HuggingFace settings...</div>
	{:else}
		<!-- Status -->
		<div class="flex items-center gap-2 text-sm">
			<span class="font-medium">Status:</span>
			{#if configured}
				<span class="flex items-center gap-1 text-green-600 dark:text-green-400">
					<Check class="h-4 w-4" />
					Token configured
				</span>
			{:else}
				<span class="text-muted-foreground">No token configured (using public access)</span>
			{/if}
		</div>

		<!-- Token Input -->
		<div class="space-y-2">
			<label for="hf-token" class="text-sm font-medium text-foreground">
				Access Token
			</label>
			<Input
				id="hf-token"
				type="password"
				bind:value={tokenInput}
				{placeholder}
			/>
			<p class="text-xs text-muted-foreground">
				Get your token from
				<a
					href="https://huggingface.co/settings/tokens"
					target="_blank"
					rel="noopener noreferrer"
					class="underline hover:text-foreground"
				>
					huggingface.co/settings/tokens
				</a>. A read-only token is sufficient for downloading models.
			</p>
		</div>

		<!-- Error Display -->
		{#if error}
			<div class="flex items-center gap-2 text-sm text-destructive">
				<X class="h-4 w-4" />
				{error}
			</div>
		{/if}

		<!-- Success Display -->
		{#if testResult?.success}
			<div class="flex items-center gap-2 text-sm text-green-600 dark:text-green-400">
				<Check class="h-4 w-4" />
				Authenticated as <span class="font-medium">{testResult.username}</span>
			</div>
		{/if}

		<!-- Action Buttons -->
		<div class="flex items-center gap-2">
			<Button
				onclick={handleSave}
				disabled={saving || !tokenInput.trim()}
				variant="default"
				size="sm"
			>
				{#if saving}
					<Loader2 class="h-4 w-4 mr-2 animate-spin" />
					Saving...
				{:else}
					<Save class="h-4 w-4 mr-2" />
					Save & Test
				{/if}
			</Button>

			{#if configured}
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
					Remove
				</Button>
			{/if}
		</div>
	{/if}
</div>
