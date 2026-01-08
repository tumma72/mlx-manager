<script lang="ts">
	import type { ServerProfile, RunningServer } from '$api';
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
		MessageSquare
	} from 'lucide-svelte';

	interface Props {
		profile: ServerProfile;
		server?: RunningServer;
		showManagementActions?: boolean;
		onChat?: (profile: ServerProfile) => void;
	}

	let { profile, server, showManagementActions = true, onChat }: Props = $props();

	let loading = $state(false);
	let error = $state<string | null>(null);

	const isRunning = $derived(!!server || serverStore.isRunning(profile.id));
	const currentServer = $derived(server || serverStore.getServer(profile.id));

	const healthColor = $derived(() => {
		if (!currentServer) return 'bg-gray-400';
		switch (currentServer.health_status) {
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

	function handleChat() {
		onChat?.(profile);
	}
</script>

<Card class="p-4">
	<div class="flex items-start justify-between gap-4">
		<div class="flex items-start gap-3 flex-1 min-w-0">
			<div class={`w-3 h-3 rounded-full mt-1.5 flex-shrink-0 ${healthColor()}`}></div>
			<div class="flex-1 min-w-0">
				<div class="flex items-center gap-2 flex-wrap">
					<h3 class="font-semibold text-lg">{profile.name}</h3>
					{#if isRunning}
						<Badge variant="success">Running</Badge>
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
		</div>

		<div class="flex items-center gap-2 flex-shrink-0">
			<!-- Server Control Buttons -->
			{#if isRunning}
				{#if onChat}
					<Button variant="default" size="sm" onclick={handleChat} disabled={loading} title="Chat">
						<MessageSquare class="w-4 h-4 mr-1" />
						Chat
					</Button>
				{/if}
				<Button variant="outline" size="sm" onclick={handleStop} disabled={loading} title="Stop">
					<Square class="w-4 h-4" />
				</Button>
				<Button variant="outline" size="sm" onclick={handleRestart} disabled={loading} title="Restart">
					<RotateCw class="w-4 h-4" />
				</Button>
			{:else}
				<Button variant="default" size="sm" onclick={handleStart} disabled={loading} title="Start">
					<Play class="w-4 h-4 mr-1" />
					Start
				</Button>
			{/if}

			<!-- Management Buttons -->
			{#if showManagementActions}
				<Button variant="outline" size="icon" href={`/profiles/${profile.id}`} title="Edit">
					<Edit class="w-4 h-4" />
				</Button>
				<Button variant="outline" size="icon" onclick={handleDuplicate} disabled={loading} title="Duplicate">
					<Copy class="w-4 h-4" />
				</Button>
				<Button variant="outline" size="icon" onclick={handleDelete} disabled={loading} title="Delete">
					<Trash2 class="w-4 h-4" />
				</Button>
			{/if}
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
		<div class="mt-2 text-sm text-red-500">{error}</div>
	{/if}
</Card>
