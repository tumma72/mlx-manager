<script lang="ts">
	import type { ServerProfile, RunningServer } from '$api';
	import { servers as serversApi } from '$api';
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { serverStore, profileStore } from '$stores';
	import { formatDuration } from '$lib/utils/format';
	import { Card, Button, Badge } from '$components/ui';
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
		AlertCircle
	} from 'lucide-svelte';

	// Timeout for model loading (2 minutes)
	const MODEL_LOAD_TIMEOUT_MS = 120_000;
	const POLL_INTERVAL_MS = 5_000; // Poll every 5 seconds to reduce server load

	interface Props {
		profile: ServerProfile;
		server?: RunningServer;
	}

	let { profile, server }: Props = $props();

	let loading = $state(false);
	let error = $state<string | null>(null);
	let modelReady = $state(false);
	let checkingModel = $state(false);
	let startTime = $state<number | null>(null);
	let serverFailed = $state(false);
	let crashError = $state<string | null>(null);

	const isRunning = $derived(!!server || serverStore.isRunning(profile.id));
	const currentServer = $derived(server || serverStore.getServer(profile.id));

	// Check if model is loaded when server is running
	$effect(() => {
		if (isRunning && !modelReady && !checkingModel) {
			checkModelReady();
		}
		if (!isRunning) {
			modelReady = false;
		}
	});

	async function checkModelReady() {
		checkingModel = true;

		// Check for timeout
		if (startTime && Date.now() - startTime > MODEL_LOAD_TIMEOUT_MS) {
			checkingModel = false;
			serverFailed = true;
			error = 'Model loading timed out after 2 minutes. Check server logs for details.';
			return;
		}

		// Check if server process is still alive
		try {
			const status = await serversApi.status(profile.id);
			if (!status.running) {
				if (status.failed) {
					serverFailed = true;
					crashError = status.error_message || 'Server process exited unexpectedly';
					error = 'Server crashed while loading model';
					checkingModel = false;
					// Refresh server list to update UI
					await serverStore.refresh();
					return;
				}
				// Server stopped gracefully - stop polling
				checkingModel = false;
				return;
			}
		} catch {
			// Status endpoint failed - continue with model check
		}

		// Check if model is loaded
		try {
			const response = await fetch(`http://${profile.host}:${profile.port}/v1/models`);
			if (response.ok) {
				const data = await response.json();
				// Check if any models are loaded
				modelReady = data.data && data.data.length > 0;
			}
		} catch {
			modelReady = false;
		} finally {
			checkingModel = false;
		}

		// Retry if not ready and no error
		if (isRunning && !modelReady && !serverFailed) {
			setTimeout(checkModelReady, POLL_INTERVAL_MS);
		}
	}

	async function handleStart() {
		loading = true;
		error = null;
		modelReady = false;
		serverFailed = false;
		crashError = null;
		startTime = Date.now();
		try {
			await serverStore.start(profile.id);
			// Start checking for model readiness
			checkModelReady();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to start server';
			startTime = null;
		} finally {
			loading = false;
		}
	}

	async function handleStop() {
		loading = true;
		error = null;
		modelReady = false;
		serverFailed = false;
		crashError = null;
		startTime = null;
		try {
			await serverStore.stop(profile.id);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to stop server';
		} finally {
			loading = false;
		}
	}

	async function handleRestart() {
		loading = true;
		error = null;
		modelReady = false;
		serverFailed = false;
		crashError = null;
		startTime = Date.now();
		try {
			await serverStore.restart(profile.id);
			// Start checking for model readiness
			setTimeout(checkModelReady, POLL_INTERVAL_MS);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to restart server';
			startTime = null;
		} finally {
			loading = false;
		}
	}

	async function handleDelete() {
		if (!confirm('Are you sure you want to delete this profile?')) return;
		loading = true;
		error = null;
		try {
			if (isRunning) {
				await serverStore.stop(profile.id);
			}
			await profileStore.delete(profile.id);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to delete profile';
		} finally {
			loading = false;
		}
	}

	async function handleDuplicate() {
		const newName = prompt('Enter name for the duplicate profile:', `${profile.name} (copy)`);
		if (!newName) return;
		loading = true;
		error = null;
		try {
			await profileStore.duplicate(profile.id, newName);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to duplicate profile';
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
				{#if isRunning || serverFailed}
					{#if serverFailed}
						<Badge variant="destructive">Error</Badge>
					{:else if modelReady}
						<Badge variant="success">Ready</Badge>
					{:else}
						<Badge variant="warning">Loading...</Badge>
					{/if}
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

		<div class="flex items-center gap-2 flex-shrink-0">
			<!-- Server Control Buttons -->
			{#if isRunning}
				<Button
					variant="default"
					size="sm"
					onclick={handleChat}
					disabled={loading || !modelReady}
					title={modelReady ? "Chat with model" : "Waiting for model to load..."}
				>
					{#if checkingModel && !modelReady}
						<Loader2 class="w-4 h-4 mr-1 animate-spin" />
					{:else}
						<MessageSquare class="w-4 h-4 mr-1" />
					{/if}
					Chat
				</Button>
				<Button variant="outline" size="icon" onclick={handleStop} disabled={loading} title="Stop">
					<Square class="w-4 h-4" />
				</Button>
				<Button variant="outline" size="icon" onclick={handleRestart} disabled={loading} title="Restart">
					<RotateCw class="w-4 h-4" />
				</Button>
			{:else}
				<Button variant="default" size="sm" onclick={handleStart} disabled={loading} title="Start">
					{#if loading}
						<Loader2 class="w-4 h-4 mr-1 animate-spin" />
					{:else}
						<Play class="w-4 h-4 mr-1" />
					{/if}
					Start
				</Button>
			{/if}

			<!-- Management Buttons (always shown) -->
			<Button variant="outline" size="icon" href={`/profiles/${profile.id}`} title="Edit">
				<Edit class="w-4 h-4" />
			</Button>
			<Button variant="outline" size="icon" onclick={handleDuplicate} disabled={loading} title="Duplicate">
				<Copy class="w-4 h-4" />
			</Button>
			<Button variant="outline" size="icon" onclick={handleDelete} disabled={loading} title="Delete">
				<Trash2 class="w-4 h-4" />
			</Button>
		</div>
	</div>

	<!-- Server Stats (when running) -->
	{#if currentServer}
		<div class="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
			<div class="flex items-center gap-2">
				<Activity class="w-4 h-4 text-muted-foreground" />
				<span class="text-muted-foreground">Port:</span>
				<span class="font-mono">{profile.port}</span>
			</div>

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
				<span class="text-muted-foreground">Uptime:</span>
				<span>{formatDuration(currentServer.uptime_seconds)}</span>
			</div>
		</div>
	{:else}
		<div class="mt-3 flex items-center gap-4 text-sm text-muted-foreground">
			<div class="flex items-center gap-2">
				<Activity class="w-4 h-4" />
				<span>Port:</span>
				<span class="font-mono">{profile.port}</span>
			</div>
			<div class="flex items-center gap-2">
				<span>Type:</span>
				<span>{profile.model_type}</span>
			</div>
		</div>
	{/if}

	{#if error}
		<div class="mt-3 p-3 bg-red-50 dark:bg-red-950/50 rounded-lg border border-red-200 dark:border-red-900">
			<div class="flex items-start gap-2">
				<AlertCircle class="w-4 h-4 text-red-500 mt-0.5 flex-shrink-0" />
				<div class="flex-1 min-w-0">
					<p class="text-sm text-red-600 dark:text-red-400 font-medium">{error}</p>
					{#if crashError}
						<details class="mt-2">
							<summary class="text-xs text-red-500 dark:text-red-400 cursor-pointer hover:underline">
								Show server log
							</summary>
							<pre class="mt-2 text-xs text-red-600 dark:text-red-300 bg-red-100 dark:bg-red-900/50 p-2 rounded overflow-x-auto max-h-48 whitespace-pre-wrap font-mono">{crashError}</pre>
						</details>
					{/if}
				</div>
			</div>
		</div>
	{/if}
</Card>
