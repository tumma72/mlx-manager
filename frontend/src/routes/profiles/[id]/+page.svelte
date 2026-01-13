<script lang="ts">
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { page } from '$app/stores';
	import { onMount } from 'svelte';
	import { profiles } from '$api';
	import { profileStore } from '$stores';
	import type { ServerProfile, ServerProfileUpdate } from '$api';
	import { ProfileForm } from '$components/profiles';

	let profile = $state<ServerProfile | null>(null);
	let loading = $state(true);
	let error = $state<string | null>(null);

	const profileId = $derived(parseInt($page.params.id ?? '0', 10));

	onMount(async () => {
		try {
			profile = await profiles.get(profileId);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load profile';
		} finally {
			loading = false;
		}
	});

	async function handleSubmit(data: ServerProfileUpdate) {
		await profileStore.update(profileId, data);
		await goto(resolve('/profiles'));
	}

	async function handleCancel() {
		await goto(resolve('/profiles'));
	}
</script>

<div class="max-w-2xl mx-auto">
	<h1 class="text-2xl font-bold mb-6">Edit Server Profile</h1>

	{#if loading}
		<div class="text-center py-12 text-muted-foreground">Loading profile...</div>
	{:else if error}
		<div class="text-center py-12 text-red-500">{error}</div>
	{:else if profile}
		<ProfileForm {profile} onSubmit={handleSubmit} onCancel={handleCancel} />
	{/if}
</div>
