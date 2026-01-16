<script lang="ts">
	import { onMount } from 'svelte';
	import { profileStore, serverStore } from '$stores';
	import { ProfileCard } from '$components/profiles';
	import { Card, Button } from '$components/ui';
	import { Plus } from 'lucide-svelte';

	// Scroll preservation: continuously track scroll position and restore after updates
	let lastScrollY = 0;
	let rafId: number | null = null;

	onMount(() => {
		// Initial data load - polling is handled globally by +layout.svelte
		profileStore.refresh();
		serverStore.refresh();

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

	// Restore scroll position after any store update causes a re-render
	$effect(() => {
		void profileStore.profiles;

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
		<h1 class="text-2xl font-bold">Server Profiles</h1>
		<Button href="/profiles/new">
			<Plus class="w-4 h-4 mr-2" />
			New Profile
		</Button>
	</div>

	{#if profileStore.loading}
		<div class="text-center py-12 text-muted-foreground">Loading profiles...</div>
	{:else if profileStore.error}
		<div class="text-center py-12 text-red-500 dark:text-red-400">{profileStore.error}</div>
	{:else if profileStore.profiles.length === 0}
		<Card class="p-12 text-center">
			<p class="text-muted-foreground mb-4">No profiles configured yet.</p>
			<Button href="/profiles/new">
				<Plus class="w-4 h-4 mr-2" />
				Create Your First Profile
			</Button>
		</Card>
	{:else}
		<div class="space-y-4">
			{#each profileStore.profiles as profile (profile.id)}
				<ProfileCard {profile} />
			{/each}
		</div>
	{/if}
</div>
