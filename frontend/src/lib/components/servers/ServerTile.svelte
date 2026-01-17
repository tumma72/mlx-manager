<script lang="ts">
	import type { RunningServer } from '$api';
	import { serverStore } from '$stores';
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { formatDuration } from '$lib/utils/format';
	import { Card, Button, Badge } from '$components/ui';
	import { MetricGauge } from '$components/servers';
	import { Square, RotateCw, MessageSquare, Loader2, Clock, Server, Hash } from 'lucide-svelte';

	interface Props {
		server: RunningServer;
	}

	let { server }: Props = $props();

	let stopping = $state(false);
	let restarting = $state(false);

	async function handleStop() {
		stopping = true;
		try {
			await serverStore.stop(server.profile_id);
		} finally {
			stopping = false;
		}
	}

	async function handleRestart() {
		restarting = true;
		try {
			await serverStore.restart(server.profile_id);
		} finally {
			restarting = false;
		}
	}

	async function handleChat() {
		await goto(`${resolve('/chat')}?profile=${server.profile_id}`);
	}

	// Derive model path from profile_name (shown in card)
	// The actual model path is stored in the profile, not the running server
	// So we show profile_name prominently and can truncate long paths
	const displayName = $derived(server.profile_name);
</script>

<Card class="p-4">
	<!-- Header Row: Badge, Name, Actions -->
	<div class="mb-3 flex items-start justify-between gap-2">
		<div class="min-w-0 flex-1">
			<div class="mb-1 flex items-center gap-2">
				<Badge variant="success" class="shrink-0">Running</Badge>
				<h3 class="truncate font-semibold">{displayName}</h3>
			</div>
			<p class="truncate text-sm text-muted-foreground">Port {server.port}</p>
		</div>

		<div class="flex shrink-0 gap-1">
			<Button variant="outline" size="sm" onclick={handleChat} title="Open Chat">
				<MessageSquare class="h-4 w-4" />
			</Button>
			<Button
				variant="outline"
				size="sm"
				onclick={handleRestart}
				disabled={restarting || stopping}
				title="Restart Server"
			>
				{#if restarting}
					<Loader2 class="h-4 w-4 animate-spin" />
				{:else}
					<RotateCw class="h-4 w-4" />
				{/if}
			</Button>
			<Button
				variant="destructive"
				size="sm"
				onclick={handleStop}
				disabled={stopping || restarting}
				title="Stop Server"
			>
				{#if stopping}
					<Loader2 class="h-4 w-4 animate-spin" />
				{:else}
					<Square class="h-4 w-4" />
				{/if}
			</Button>
		</div>
	</div>

	<!-- Content Row: Gauges + Stats -->
	<div class="flex items-center gap-6">
		<!-- Metrics Gauges -->
		<div class="flex gap-3">
			<MetricGauge value={server.memory_percent} label="Memory" size="md" />
			<MetricGauge value={server.cpu_percent} label="CPU" size="md" />
		</div>

		<!-- Text Stats -->
		<div class="flex flex-1 flex-col gap-1 text-sm text-muted-foreground">
			<div class="flex items-center gap-2">
				<Clock class="h-3.5 w-3.5" />
				<span>Uptime: {formatDuration(server.uptime_seconds)}</span>
			</div>
			<div class="flex items-center gap-2">
				<Server class="h-3.5 w-3.5" />
				<span>Memory: {server.memory_mb.toFixed(0)} MB</span>
			</div>
			<div class="flex items-center gap-2">
				<Hash class="h-3.5 w-3.5" />
				<span>PID: {server.pid}</span>
			</div>
		</div>
	</div>
</Card>
