<script lang="ts">
	import type { ServerProfile } from '$api';
	import { servers as serversApi, servers } from '$api';
	import { serverStore } from '$stores';
	import { Card, Button, Badge } from '$components/ui';
	import { Loader2, Square, AlertCircle, Clipboard, Check } from 'lucide-svelte';

	// Timeout for model loading (2 minutes)
	const MODEL_LOAD_TIMEOUT_MS = 120_000;
	const POLL_INTERVAL_MS = 3_000; // Poll every 3 seconds during startup
	const INITIAL_HEALTH_DELAY_MS = 5_000; // Wait 5s after PID confirmation before first health check

	interface Props {
		profile: ServerProfile;
	}

	let { profile }: Props = $props();

	let stopping = $state(false);
	let startTime = $state<number | null>(null);
	let pollTimeoutId = $state<ReturnType<typeof setTimeout> | null>(null);
	let isPolling = $state(false);

	// Derived state from store
	const isFailed = $derived(serverStore.isFailed(profile.id));
	const failure = $derived(serverStore.getFailure(profile.id));

	// Local state for error details collapse
	let localDetailsOpen = $state(false);
	let lastFailureId = $state<number | null>(null);
	let isUpdatingDetails = false;

	// Sync local details state from store when failure first appears
	$effect(() => {
		const currentFailure = serverStore.getFailure(profile.id);
		if (currentFailure && lastFailureId !== profile.id) {
			lastFailureId = profile.id;
			isUpdatingDetails = true;
			localDetailsOpen = currentFailure.detailsOpen;
			requestAnimationFrame(() => {
				isUpdatingDetails = false;
			});
		} else if (!currentFailure) {
			lastFailureId = null;
			localDetailsOpen = false;
		}
	});

	function handleDetailsToggle() {
		if (!isUpdatingDetails) {
			serverStore.toggleDetailsOpen(profile.id);
		}
	}

	// Copy error details to clipboard
	let copySuccess = $state(false);
	async function copyErrorToClipboard() {
		if (failure?.details) {
			try {
				await navigator.clipboard.writeText(failure.details);
				copySuccess = true;
				setTimeout(() => {
					copySuccess = false;
				}, 2000);
			} catch (e) {
				console.error('Failed to copy to clipboard:', e);
			}
		}
	}

	// On mount: start polling if profile is starting and not already polling
	$effect(() => {
		// Start polling when component mounts for a starting profile
		if (serverStore.isStarting(profile.id) && !isPolling && !serverStore.isProfilePolling(profile.id)) {
			if (serverStore.startProfilePolling(profile.id)) {
				isPolling = true;
				startTime = Date.now();
				pollServerStatus();
			}
		}

		// Cleanup on unmount
		return () => {
			if (pollTimeoutId) {
				clearTimeout(pollTimeoutId);
				pollTimeoutId = null;
			}
			// Don't stop profile polling here - it might be picked up by a re-mounted tile
		};
	});

	async function pollServerStatus() {
		if (!isPolling) {
			serverStore.stopProfilePolling(profile.id);
			return;
		}

		// Check for timeout
		if (startTime && Date.now() - startTime > MODEL_LOAD_TIMEOUT_MS) {
			serverStore.markStartupFailed(
				profile.id,
				'Model loading timed out after 2 minutes. Check server logs for details.'
			);
			isPolling = false;
			pollTimeoutId = null;
			serverStore.stopProfilePolling(profile.id);
			return;
		}

		try {
			const status = await serversApi.status(profile.id);

			if (!status.running) {
				if (status.failed) {
					serverStore.markStartupFailed(
						profile.id,
						'Server crashed while loading model',
						status.error_message || 'Server process exited unexpectedly'
					);
				} else {
					serverStore.markStartupSuccess(profile.id);
				}
				isPolling = false;
				pollTimeoutId = null;
				serverStore.stopProfilePolling(profile.id);
				return;
			}

			// Server is running (PID confirmed), check if model is loaded via backend health API
			try {
				const healthStatus = await servers.health(profile.id);
				if (healthStatus.status === 'healthy' && healthStatus.model_loaded) {
					serverStore.markStartupSuccess(profile.id);
					isPolling = false;
					pollTimeoutId = null;
					serverStore.stopProfilePolling(profile.id);
					return;
				}
			} catch {
				// Health check failed, keep polling
			}

			pollTimeoutId = setTimeout(pollServerStatus, POLL_INTERVAL_MS);
		} catch {
			pollTimeoutId = setTimeout(pollServerStatus, POLL_INTERVAL_MS);
		}
	}

	async function handleStop() {
		stopping = true;
		startTime = null;
		isPolling = false;
		if (pollTimeoutId) {
			clearTimeout(pollTimeoutId);
			pollTimeoutId = null;
		}
		serverStore.stopProfilePolling(profile.id);

		try {
			await serverStore.stop(profile.id);
		} catch (e) {
			const errorMsg = e instanceof Error ? e.message : 'Failed to stop server';
			serverStore.markStartupFailed(profile.id, errorMsg);
		} finally {
			stopping = false;
		}
	}

	function handleDismiss() {
		serverStore.clearFailure(profile.id);
	}

	async function handleRetry() {
		serverStore.clearFailure(profile.id);
		startTime = Date.now();

		if (pollTimeoutId) {
			clearTimeout(pollTimeoutId);
			pollTimeoutId = null;
		}

		if (!serverStore.startProfilePolling(profile.id)) {
			return;
		}

		isPolling = true;

		try {
			await serverStore.start(profile.id);
			pollServerStatus();
		} catch (e) {
			const errorMsg = e instanceof Error ? e.message : 'Failed to start server';
			serverStore.markStartupFailed(profile.id, errorMsg);
			startTime = null;
			isPolling = false;
			serverStore.stopProfilePolling(profile.id);
		}
	}
</script>

<Card class="p-4">
	<!-- Header Row: Badge, Name, Actions -->
	<div class="mb-3 flex items-start justify-between gap-2">
		<div class="min-w-0 flex-1">
			<div class="mb-1 flex items-center gap-2">
				{#if isFailed}
					<Badge variant="destructive" class="shrink-0">Error</Badge>
				{:else}
					<Badge variant="warning" class="shrink-0">Starting</Badge>
				{/if}
				<h3 class="truncate font-semibold">{profile.name}</h3>
			</div>
			<p class="truncate text-sm text-muted-foreground">{profile.model_path}</p>
		</div>

		<div class="flex shrink-0 gap-1">
			{#if isFailed}
				<Button variant="outline" size="sm" onclick={handleRetry} title="Retry">
					Retry
				</Button>
				<Button variant="outline" size="sm" onclick={handleDismiss} title="Dismiss">
					Dismiss
				</Button>
			{:else}
				<Button
					variant="outline"
					size="sm"
					onclick={handleStop}
					disabled={stopping}
					title="Cancel Start"
				>
					{#if stopping}
						<Loader2 class="mr-1 h-4 w-4 animate-spin" />
					{:else}
						<Square class="mr-1 h-4 w-4" />
					{/if}
					Cancel
				</Button>
			{/if}
		</div>
	</div>

	<!-- Status Message -->
	{#if !isFailed}
		<div class="flex items-center gap-2 text-sm text-muted-foreground">
			<Loader2 class="h-4 w-4 animate-spin" />
			<span>Loading model... this may take a minute</span>
		</div>
	{/if}

	<!-- Error Details -->
	{#if failure}
		<div
			class="rounded-lg border border-red-200 bg-red-50 p-3 dark:border-red-900 dark:bg-red-950/50"
		>
			<div class="flex items-start gap-2">
				<AlertCircle class="mt-0.5 h-4 w-4 flex-shrink-0 text-red-500 dark:text-red-400" />
				<div class="min-w-0 flex-1">
					<p class="text-sm font-medium text-red-600 dark:text-red-400">{failure.error}</p>
					{#if failure.details}
						<details class="mt-2" bind:open={localDetailsOpen} ontoggle={handleDetailsToggle}>
							<summary
								class="cursor-pointer text-xs text-red-500 hover:underline dark:text-red-400"
							>
								Show server log
							</summary>
							<div class="relative mt-2">
								<button
									onclick={copyErrorToClipboard}
									class="absolute right-2 top-2 rounded bg-red-200 p-1.5 transition-colors hover:bg-red-300 dark:bg-red-800 dark:hover:bg-red-700"
									title="Copy to clipboard"
								>
									{#if copySuccess}
										<Check class="h-3.5 w-3.5 text-green-600 dark:text-green-400" />
									{:else}
										<Clipboard class="h-3.5 w-3.5 text-red-600 dark:text-red-300" />
									{/if}
								</button>
								<pre
									class="max-h-48 overflow-x-auto whitespace-pre-wrap rounded bg-red-100 p-2 pr-10 font-mono text-xs text-red-600 dark:bg-red-900/50 dark:text-red-300">{failure.details}</pre>
							</div>
						</details>
					{/if}
				</div>
			</div>
		</div>
	{/if}
</Card>
