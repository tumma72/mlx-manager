<script lang="ts">
	import { onMount } from 'svelte';
	import { serverStore, profileStore } from '$stores';
	import { ProfileCard } from '$components/profiles';
	import { Button } from '$components/ui';
	import { RefreshCw, Plus } from 'lucide-svelte';
	let refreshing = $state(false);

	// Scroll preservation: continuously track scroll position and restore after updates
	let lastScrollY = 0;
	let rafId: number | null = null;

	onMount(() => {
		// Initial data load - polling is handled globally by +layout.svelte
		serverStore.refresh();
		profileStore.refresh();

		// Track scroll position continuously
		function onScroll() {
			lastScrollY = window.scrollY;
		}
		window.addEventListener('scroll', onScroll, { passive: true });

		return () => {
			window.removeEventListener('scroll', onScroll);
			if (rafId) cancelAnimationFrame(rafId);
		};
	});

	async function handleRefresh() {
		refreshing = true;
		await Promise.all([serverStore.refresh(), profileStore.refresh()]);
		refreshing = false;
	}

	// Separate running and stopped profiles
	const runningProfiles = $derived(
		profileStore.profiles.filter((p) => serverStore.isRunning(p.id))
	);

	const stoppedProfiles = $derived(
		profileStore.profiles.filter((p) => !serverStore.isRunning(p.id))
	);

	// Restore scroll position after any store update causes a re-render
	$effect(() => {
		// Track these dependencies to run after updates
		void serverStore.servers;
		void profileStore.profiles;

		// Use double-RAF to ensure DOM is fully updated before restoring scroll
		if (rafId) cancelAnimationFrame(rafId);
		rafId = requestAnimationFrame(() => {
			rafId = requestAnimationFrame(() => {
				if (lastScrollY > 0 && Math.abs(window.scrollY - lastScrollY) > 50) {
					window.scrollTo({ top: lastScrollY, behavior: 'instant' });
				}
			});
		});
	});
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
		<div class="text-center py-12 text-muted-foreground">Loading servers...</div>
	{:else if serverStore.error}
		<div class="text-center py-12 text-red-500 dark:text-red-400">{serverStore.error}</div>
	{:else}
		<!-- Running Servers -->
		<section>
			<h2 class="text-lg font-semibold mb-4">
				Running Servers ({runningProfiles.length})
			</h2>
			{#if runningProfiles.length === 0}
				<div class="text-center py-8 text-muted-foreground bg-white dark:bg-gray-800 rounded-lg border">
					No servers running. Start a profile to begin.
				</div>
			{:else}
				<div class="space-y-4">
					{#each runningProfiles as profile (profile.id)}
						<ProfileCard
							{profile}
							server={serverStore.getServer(profile.id)}
						/>
					{/each}
				</div>
			{/if}
		</section>

		<!-- Stopped Profiles -->
		{#if stoppedProfiles.length > 0}
			<section>
				<h2 class="text-lg font-semibold mb-4">
					Available Profiles ({stoppedProfiles.length})
				</h2>
				<div class="space-y-4">
					{#each stoppedProfiles as profile (profile.id)}
						<ProfileCard {profile} />
					{/each}
				</div>
			</section>
		{/if}

		{#if profileStore.profiles.length === 0}
			<div class="text-center py-12">
				<p class="text-muted-foreground mb-4">No profiles configured yet.</p>
				<Button href="/profiles/new">
					<Plus class="w-4 h-4 mr-2" />
					Create Your First Profile
				</Button>
			</div>
		{/if}
	{/if}
</div>
