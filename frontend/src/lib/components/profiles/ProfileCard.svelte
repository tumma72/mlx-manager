<script lang="ts">
	import type { ServerProfile, RunningServer } from '$api';
	import { servers as serversApi } from '$api';
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { serverStore, profileStore } from '$stores';
	import { formatDuration } from '$lib/utils/format';
	import { Card, Button, Badge, ConfirmDialog } from '$components/ui';
	import {
		Play,
		Square,
		RotateCw,
		Edit,
		Trash2,
		Copy,
		Activity,
		Cpu,
		HardDrive,
		MessageSquare,
		Loader2,
		AlertCircle,
		Clipboard,
		Check
	} from 'lucide-svelte';

	// Timeout for model loading (2 minutes)
	const MODEL_LOAD_TIMEOUT_MS = 120_000;
	const POLL_INTERVAL_MS = 2_000; // Poll every 2 seconds during startup

	interface Props {
		profile: ServerProfile;
		server?: RunningServer;
	}

	let { profile, server }: Props = $props();

	let loading = $state(false);
	let startTime = $state<number | null>(null);
	let pollTimeoutId = $state<ReturnType<typeof setTimeout> | null>(null);
	// Track if we're actively polling (independent of store state)
	let isPolling = $state(false);
	// Delete confirmation dialog state
	let showDeleteConfirm = $state(false);

	// Derived state from store (store is source of truth for status)
	const isStarting = $derived(serverStore.isStarting(profile.id));
	const isRunning = $derived(serverStore.isRunning(profile.id));
	const isFailed = $derived(serverStore.isFailed(profile.id));
	const failure = $derived(serverStore.getFailure(profile.id));
	const currentServer = $derived(server || serverStore.getServer(profile.id));

	// Local state for error details collapse - managed locally to prevent infinite loops
	// The store's detailsOpen state was causing a bidirectional binding loop:
	// effect syncs from store → updates DOM → ontoggle fires → updates store → repeat
	let localDetailsOpen = $state(false);

	// Track if we're programmatically updating to prevent toggle handler from firing
	let isUpdatingDetails = false;

	// Sync local details state from store ONLY when failure first appears
	// After that, manage it locally to avoid loops
	let lastFailureId = $state<number | null>(null);
	$effect(() => {
		const currentFailure = serverStore.getFailure(profile.id);
		// Only sync on initial failure or when failure changes (not on every update)
		if (currentFailure && lastFailureId !== profile.id) {
			lastFailureId = profile.id;
			isUpdatingDetails = true;
			localDetailsOpen = currentFailure.detailsOpen;
			// Reset flag after DOM update
			requestAnimationFrame(() => { isUpdatingDetails = false; });
		} else if (!currentFailure) {
			lastFailureId = null;
			localDetailsOpen = false;
		}
	});

	// Handle toggle - only respond to user interaction, not programmatic changes
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
				setTimeout(() => { copySuccess = false; }, 2000);
			} catch (e) {
				console.error('Failed to copy to clipboard:', e);
			}
		}
	}

	// On mount: resume polling if profile is still starting AND no other component is polling
	// Using $effect with empty deps equivalent - runs once on mount
	let mounted = $state(false);
	$effect(() => {
		if (!mounted) {
			mounted = true;
			// Check if we need to resume polling (component recreated while profile starting)
			// Use store's polling tracker to prevent duplicate loops across component recreations
			if (serverStore.isStarting(profile.id) && !serverStore.isProfilePolling(profile.id)) {
				if (serverStore.startProfilePolling(profile.id)) {
					isPolling = true;
					startTime = Date.now();
					pollServerStatus();
				}
			}
		}
	});

	// Cleanup on unmount
	$effect(() => {
		return () => {
			if (pollTimeoutId) {
				clearTimeout(pollTimeoutId);
				pollTimeoutId = null;
			}
			// Note: Don't stop profile polling here - it should continue even if component unmounts
			// The polling state is managed by the store and will be cleared when polling actually stops
		};
	});

	async function pollServerStatus() {
		// Guard: stop if we're no longer supposed to be polling
		if (!isPolling) {
			serverStore.stopProfilePolling(profile.id);
			return;
		}

		// Check for timeout
		if (startTime && Date.now() - startTime > MODEL_LOAD_TIMEOUT_MS) {
			serverStore.markStartupFailed(profile.id, 'Model loading timed out after 2 minutes. Check server logs for details.');
			isPolling = false;
			pollTimeoutId = null;
			serverStore.stopProfilePolling(profile.id);
			return;
		}

		// Check server status from backend
		try {
			const status = await serversApi.status(profile.id);

			if (!status.running) {
				// Server is not running
				if (status.failed) {
					serverStore.markStartupFailed(
						profile.id,
						'Server crashed while loading model',
						status.error_message || 'Server process exited unexpectedly'
					);
				} else {
					// Server stopped gracefully
					serverStore.markStartupSuccess(profile.id);
				}
				isPolling = false;
				pollTimeoutId = null;
				serverStore.stopProfilePolling(profile.id);
				return;
			}

			// Server is running, check if health endpoint reports ready
			// With embedded server, this always returns healthy immediately
			try {
				const health = await serversApi.health(profile.id);
				if (health.status === 'healthy' && health.model_loaded) {
					// Server is ready!
					serverStore.markStartupSuccess(profile.id);
					isPolling = false;
					pollTimeoutId = null;
					serverStore.stopProfilePolling(profile.id);
					return;
				}
			} catch {
				// Health check failed, keep polling
			}

			// Continue polling
			pollTimeoutId = setTimeout(pollServerStatus, POLL_INTERVAL_MS);
		} catch {
			// Status check failed, keep polling
			pollTimeoutId = setTimeout(pollServerStatus, POLL_INTERVAL_MS);
		}
	}

	async function handleStart() {
		loading = true;
		serverStore.clearFailure(profile.id);
		startTime = Date.now();

		// Clear any existing timeout
		if (pollTimeoutId) {
			clearTimeout(pollTimeoutId);
			pollTimeoutId = null;
		}

		// Register polling with the store (prevents duplicate loops)
		if (!serverStore.startProfilePolling(profile.id)) {
			loading = false;
			return;
		}

		isPolling = true;

		try {
			await serverStore.start(profile.id);
			// Start polling for server status
			pollServerStatus();
		} catch (e) {
			const errorMsg = e instanceof Error ? e.message : 'Failed to start server';
			serverStore.markStartupFailed(profile.id, errorMsg);
			startTime = null;
			isPolling = false;
			serverStore.stopProfilePolling(profile.id);
		} finally {
			loading = false;
		}
	}

	async function handleStop() {
		loading = true;
		startTime = null;
		isPolling = false;
		if (pollTimeoutId) {
			clearTimeout(pollTimeoutId);
			pollTimeoutId = null;
		}
		// Stop any polling for this profile
		serverStore.stopProfilePolling(profile.id);

		try {
			await serverStore.stop(profile.id);
		} catch (e) {
			const errorMsg = e instanceof Error ? e.message : 'Failed to stop server';
			serverStore.markStartupFailed(profile.id, errorMsg);
		} finally {
			loading = false;
		}
	}

	async function handleRestart() {
		loading = true;
		serverStore.clearFailure(profile.id);
		startTime = Date.now();

		// Clear any existing timeout
		if (pollTimeoutId) {
			clearTimeout(pollTimeoutId);
			pollTimeoutId = null;
		}

		// Register polling with the store (prevents duplicate loops)
		// For restart, we first stop any existing polling then start fresh
		serverStore.stopProfilePolling(profile.id);
		if (!serverStore.startProfilePolling(profile.id)) {
			loading = false;
			return;
		}

		isPolling = true;

		try {
			await serverStore.restart(profile.id);
			// Start polling for server status
			pollServerStatus();
		} catch (e) {
			const errorMsg = e instanceof Error ? e.message : 'Failed to restart server';
			serverStore.markStartupFailed(profile.id, errorMsg);
			startTime = null;
			isPolling = false;
			serverStore.stopProfilePolling(profile.id);
		} finally {
			loading = false;
		}
	}

	function requestDelete() {
		showDeleteConfirm = true;
	}

	async function confirmDelete() {
		loading = true;
		serverStore.clearFailure(profile.id);
		try {
			if (isRunning) {
				await serverStore.stop(profile.id);
			}
			await profileStore.delete(profile.id);
		} catch (e) {
			const errorMsg = e instanceof Error ? e.message : 'Failed to delete profile';
			serverStore.markStartupFailed(profile.id, errorMsg);
		} finally {
			loading = false;
		}
	}

	async function handleDuplicate() {
		const newName = prompt('Enter name for the duplicate profile:', `${profile.name} (copy)`);
		if (!newName) return;
		loading = true;
		try {
			await profileStore.duplicate(profile.id, newName);
		} catch (e) {
			const errorMsg = e instanceof Error ? e.message : 'Failed to duplicate profile';
			alert(errorMsg); // Simple alert for duplicate errors
		} finally {
			loading = false;
		}
	}

	async function handleChat() {
		const chatUrl = `${resolve('/chat')}?profile=${profile.id}`;
		// eslint-disable-next-line svelte/no-navigation-without-resolve -- query params appended to resolved path
		await goto(chatUrl);
	}
</script>

<Card class="p-4">
	<div class="flex items-start justify-between gap-4">
		<div class="flex-1 min-w-0">
			<div class="flex items-center gap-2 flex-wrap">
				<h3 class="font-semibold text-lg">{profile.name}</h3>
				{#if isFailed}
					<Badge variant="destructive">Error</Badge>
				{:else if isRunning}
					<Badge variant="success">Running</Badge>
				{:else if isStarting}
					<Badge variant="warning">Loading...</Badge>
				{/if}
				{#if profile.launchd_installed}
					<Badge variant="outline">launchd</Badge>
				{/if}
			</div>
			{#if profile.description}
				<p class="text-sm text-muted-foreground mt-1">{profile.description}</p>
			{/if}
			<div class="mt-2 text-sm text-muted-foreground truncate">
				<span class="font-mono">{profile.model_path}</span>
			</div>
		</div>

		<div class="flex shrink-0 gap-1">
			<!-- Server Control Buttons -->
			{#if isRunning}
				<Button variant="outline" size="sm" onclick={handleChat} disabled={loading} title="Chat">
					<MessageSquare class="h-4 w-4" />
				</Button>
				<Button variant="outline" size="sm" onclick={handleRestart} disabled={loading} title="Restart">
					<RotateCw class="h-4 w-4" />
				</Button>
				<Button variant="destructive" size="sm" onclick={handleStop} disabled={loading} title="Stop">
					<Square class="h-4 w-4" />
				</Button>
			{:else if isStarting}
				<Button variant="outline" size="sm" disabled title="Starting...">
					<Loader2 class="h-4 w-4 animate-spin" />
				</Button>
				<Button variant="destructive" size="sm" onclick={handleStop} disabled={loading} title="Cancel">
					<Square class="h-4 w-4" />
				</Button>
			{:else}
				<Button variant="default" size="sm" onclick={handleStart} disabled={loading} title="Start">
					{#if loading}
						<Loader2 class="h-4 w-4 animate-spin" />
					{:else}
						<Play class="h-4 w-4" />
					{/if}
				</Button>
			{/if}

			<!-- Management Buttons (always shown) -->
			<Button variant="outline" size="sm" href={`/profiles/${profile.id}`} title="Edit">
				<Edit class="h-4 w-4" />
			</Button>
			<Button variant="outline" size="sm" onclick={handleDuplicate} disabled={loading} title="Duplicate">
				<Copy class="h-4 w-4" />
			</Button>
			<Button variant="outline" size="sm" onclick={requestDelete} disabled={loading} title="Delete">
				<Trash2 class="h-4 w-4" />
			</Button>
		</div>
	</div>

	<!-- Server Stats (when running) -->
	{#if currentServer && isRunning}
		<div class="mt-4 grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
			<div class="flex items-center gap-2">
				<Cpu class="w-4 h-4 text-muted-foreground" />
				<span class="text-muted-foreground">PID:</span>
				<span class="font-mono">{currentServer.pid}</span>
			</div>

			<div class="flex items-center gap-2">
				<HardDrive class="w-4 h-4 text-muted-foreground" />
				<span class="text-muted-foreground">Memory:</span>
				<span>{currentServer.memory_mb.toFixed(1)} MB</span>
			</div>

			<div class="flex items-center gap-2">
				<Activity class="w-4 h-4 text-muted-foreground" />
				<span class="text-muted-foreground">Uptime:</span>
				<span>{formatDuration(currentServer.uptime_seconds)}</span>
			</div>
		</div>
	{:else}
		<div class="mt-3 flex items-center gap-4 text-sm text-muted-foreground">
			<div class="flex items-center gap-2">
				<span>Type:</span>
				<span>{profile.model_type}</span>
			</div>
		</div>
	{/if}

	{#if failure}
		<div class="mt-3 p-3 bg-red-50 dark:bg-red-950/50 rounded-lg border border-red-200 dark:border-red-900">
			<div class="flex items-start gap-2">
				<AlertCircle class="w-4 h-4 text-red-500 dark:text-red-400 mt-0.5 flex-shrink-0" />
				<div class="flex-1 min-w-0">
					<p class="text-sm text-red-600 dark:text-red-400 font-medium">{failure.error}</p>
					{#if failure.details}
						<details
							class="mt-2"
							bind:open={localDetailsOpen}
							ontoggle={handleDetailsToggle}
						>
							<summary class="text-xs text-red-500 dark:text-red-400 cursor-pointer hover:underline">
								Show server log
							</summary>
							<div class="mt-2 relative">
								<button
									onclick={copyErrorToClipboard}
									class="absolute top-2 right-2 p-1.5 rounded bg-red-200 dark:bg-red-800 hover:bg-red-300 dark:hover:bg-red-700 transition-colors"
									title="Copy to clipboard"
								>
									{#if copySuccess}
										<Check class="w-3.5 h-3.5 text-green-600 dark:text-green-400" />
									{:else}
										<Clipboard class="w-3.5 h-3.5 text-red-600 dark:text-red-300" />
									{/if}
								</button>
								<pre class="text-xs text-red-600 dark:text-red-300 bg-red-100 dark:bg-red-900/50 p-2 pr-10 rounded overflow-x-auto max-h-48 whitespace-pre-wrap font-mono">{failure.details}</pre>
							</div>
						</details>
					{/if}
				</div>
			</div>
		</div>
	{/if}
</Card>

<ConfirmDialog
	bind:open={showDeleteConfirm}
	title="Delete Profile"
	description="Are you sure you want to delete '{profile.name}'? This action cannot be undone."
	confirmLabel="Delete"
	cancelLabel="Cancel"
	variant="destructive"
	onConfirm={confirmDelete}
	onCancel={() => {}}
/>
