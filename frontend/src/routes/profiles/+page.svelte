<script lang="ts">
	import { onMount } from 'svelte';
	import { profileStore, serverStore } from '$stores';
	import { ProfileCard } from '$components/profiles';
	import { Card, Button } from '$components/ui';
	import { Plus } from 'lucide-svelte';

	onMount(() => {
		profileStore.refresh();
		serverStore.refresh();

		// Auto-refresh server status every 5 seconds
		const interval = setInterval(() => serverStore.refresh(), 5000);
		return () => clearInterval(interval);
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
		<div class="text-center py-12 text-red-500">{profileStore.error}</div>
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
				<ProfileCard {profile} showManagementActions={true} />
			{/each}
		</div>
	{/if}
</div>
