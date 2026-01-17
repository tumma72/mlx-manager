<script lang="ts">
	import { onMount } from 'svelte';
	import { serverStore, profileStore } from '$stores';
	import { ProfileSelector, ServerTile, StartingTile } from '$components/servers';
	import { Button } from '$components/ui';
	import type { ServerProfile } from '$api';
	import { RefreshCw, Plus } from 'lucide-svelte';
	let refreshing = $state(false);

	// Container-scoped scroll preservation
	let serverListContainer: HTMLElement | undefined = $state();
	let savedScrollTop = 0;

	onMount(() => {
		// Initial data load - polling is handled globally by +layout.svelte
		serverStore.refresh();
		profileStore.refresh();
	});

	// Capture scroll position BEFORE DOM updates
	$effect.pre(() => {
		// Track dependency on server list to run before updates
		void serverStore.servers.length;
		if (serverListContainer) {
			savedScrollTop = serverListContainer.scrollTop;
		}
	});

	// Restore scroll position AFTER DOM updates
	$effect(() => {
		// Track dependency to run after updates
		void serverStore.servers;
		if (serverListContainer && savedScrollTop > 0) {
			// Only restore if significantly different (prevents minor drift)
			if (Math.abs(serverListContainer.scrollTop - savedScrollTop) > 10) {
				serverListContainer.scrollTop = savedScrollTop;
			}
		}
	});

	async function handleRefresh() {
		refreshing = true;
		await Promise.all([serverStore.refresh(), profileStore.refresh()]);
		refreshing = false;
	}

	// Profiles that are starting or failed (show StartingTile)
	const startingOrFailedProfiles = $derived(
		profileStore.profiles.filter(
			(p) => serverStore.isStarting(p.id) || serverStore.isFailed(p.id)
		)
	);

	// Stopped profiles (for the start dropdown) - exclude starting and failed
	const stoppedProfiles = $derived(
		profileStore.profiles.filter(
			(p) =>
				!serverStore.isRunning(p.id) &&
				!serverStore.isStarting(p.id) &&
				!serverStore.isFailed(p.id)
		)
	);

	// Running servers count for the header
	const runningCount = $derived(serverStore.servers.length);

	async function handleStartProfile(profile: ServerProfile) {
		await serverStore.start(profile.id);
	}
</script>

<div class="space-y-6">
	<div class="flex items-center justify-between">
		<h1 class="text-2xl font-bold">Server Dashboard</h1>
		<div class="flex gap-2">
			<Button variant="outline" onclick={handleRefresh} disabled={refreshing}>
				<RefreshCw class="w-4 h-4 mr-2 {refreshing ? 'animate-spin' : ''}" />
				Refresh
			</Button>
			<Button href="/profiles/new">
				<Plus class="w-4 h-4 mr-2" />
				New Profile
			</Button>
		</div>
	</div>

	{#if serverStore.loading && serverStore.servers.length === 0}
		<div class="py-12 text-center text-muted-foreground">Loading servers...</div>
	{:else if serverStore.error}
		<div class="py-12 text-center text-red-500 dark:text-red-400">{serverStore.error}</div>
	{:else}
		<!-- Start Server Section -->
		{#if stoppedProfiles.length > 0}
			<section>
				<h2 class="mb-3 text-lg font-semibold">Start Server</h2>
				<ProfileSelector profiles={stoppedProfiles} onStart={handleStartProfile} />
			</section>
		{/if}

		<!-- Starting/Failed Servers -->
		{#if startingOrFailedProfiles.length > 0}
			<section>
				<div class="grid gap-4">
					{#each startingOrFailedProfiles as profile (profile.id)}
						<StartingTile {profile} />
					{/each}
				</div>
			</section>
		{/if}

		<!-- Running Servers -->
		<section>
			<h2 class="mb-4 text-lg font-semibold">
				Running Servers ({runningCount})
			</h2>
			{#if serverStore.servers.length === 0 && startingOrFailedProfiles.length === 0}
				<div
					class="rounded-lg border bg-white py-8 text-center text-muted-foreground dark:bg-gray-800"
				>
					No servers running. Start a profile to begin.
				</div>
			{:else if serverStore.servers.length === 0}
				<!-- Don't show "no servers" message when something is starting -->
			{:else}
				<div
					bind:this={serverListContainer}
					class="grid gap-4 overflow-auto max-h-[calc(100vh-300px)]"
				>
					{#each serverStore.servers as server (server.profile_id)}
						<ServerTile {server} />
					{/each}
				</div>
			{/if}
		</section>

		{#if profileStore.profiles.length === 0}
			<div class="py-12 text-center">
				<p class="mb-4 text-muted-foreground">No profiles configured yet.</p>
				<Button href="/profiles/new">
					<Plus class="mr-2 h-4 w-4" />
					Create Your First Profile
				</Button>
			</div>
		{/if}
	{/if}
</div>
