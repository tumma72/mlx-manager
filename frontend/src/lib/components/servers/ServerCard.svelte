<script lang="ts">
	import type { ServerProfile, RunningServer } from '$api';
	import { serverStore } from '$stores';
	import { formatDuration } from '$lib/utils/format';
	import { Card, Button } from '$components/ui';
	import {
		Play,
		Square,
		RotateCw,
		Activity,
		Cpu,
		HardDrive
	} from 'lucide-svelte';

	interface Props {
		profile: ServerProfile;
		server?: RunningServer;
	}

	let { profile, server }: Props = $props();

	let loading = $state(false);
	let error = $state<string | null>(null);

	const isRunning = $derived(!!server);

	const healthColor = $derived(() => {
		if (!server) return 'bg-gray-400';
		switch (server.health_status) {
			case 'healthy':
				return 'bg-green-500';
			case 'starting':
				return 'bg-yellow-500';
			case 'unhealthy':
				return 'bg-red-500';
			default:
				return 'bg-gray-400';
		}
	});

	async function handleStart() {
		loading = true;
		error = null;
		try {
			await serverStore.start(profile.id);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to start server';
		} finally {
			loading = false;
		}
	}

	async function handleStop() {
		loading = true;
		error = null;
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
		try {
			await serverStore.restart(profile.id);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to restart server';
		} finally {
			loading = false;
		}
	}
</script>

<Card class="p-4">
	<div class="flex items-start justify-between">
		<div class="flex items-center gap-3">
			<div class={`w-3 h-3 rounded-full ${healthColor()}`}></div>
			<div>
				<h3 class="font-semibold text-lg">{profile.name}</h3>
				{#if profile.description}
					<p class="text-sm text-muted-foreground">{profile.description}</p>
				{/if}
			</div>
		</div>

		<div class="flex gap-2">
			{#if isRunning}
				<Button variant="outline" size="sm" onclick={handleStop} disabled={loading}>
					<Square class="w-4 h-4 mr-1" />
					Stop
				</Button>
				<Button variant="outline" size="sm" onclick={handleRestart} disabled={loading}>
					<RotateCw class="w-4 h-4 mr-1" />
					Restart
				</Button>
			{:else}
				<Button variant="default" size="sm" onclick={handleStart} disabled={loading}>
					<Play class="w-4 h-4 mr-1" />
					Start
				</Button>
			{/if}
		</div>
	</div>

	<div class="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
		<div class="flex items-center gap-2">
			<Activity class="w-4 h-4 text-muted-foreground" />
			<span class="text-muted-foreground">Port:</span>
			<span class="font-mono">{profile.port}</span>
		</div>

		{#if server}
			<div class="flex items-center gap-2">
				<Cpu class="w-4 h-4 text-muted-foreground" />
				<span class="text-muted-foreground">PID:</span>
				<span class="font-mono">{server.pid}</span>
			</div>

			<div class="flex items-center gap-2">
				<HardDrive class="w-4 h-4 text-muted-foreground" />
				<span class="text-muted-foreground">Memory:</span>
				<span>{server.memory_mb.toFixed(1)} MB</span>
			</div>

			<div class="flex items-center gap-2">
				<span class="text-muted-foreground">Uptime:</span>
				<span>{formatDuration(server.uptime_seconds)}</span>
			</div>
		{/if}
	</div>

	<div class="mt-3 text-sm text-muted-foreground truncate">
		<span class="font-mono">{profile.model_path}</span>
	</div>

	{#if error}
		<div class="mt-2 text-sm text-red-500 dark:text-red-400">{error}</div>
	{/if}
</Card>
