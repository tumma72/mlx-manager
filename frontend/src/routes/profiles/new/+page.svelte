<script lang="ts">
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { page } from '$app/stores';
	import { onMount } from 'svelte';
	import { profileStore } from '$stores';
	import type { ServerProfileCreate, ServerProfileUpdate } from '$api';
	import { ProfileForm } from '$components/profiles';

	let nextPort = $state(10240);
	let initialModelPath = $state('');

	onMount(async () => {
		// Get next available port
		nextPort = await profileStore.getNextPort();

		// Check for model query param
		const modelParam = $page.url.searchParams.get('model');
		if (modelParam) {
			initialModelPath = modelParam;
		}
	});

	async function handleSubmit(data: ServerProfileCreate | ServerProfileUpdate) {
		await profileStore.create(data as ServerProfileCreate);
		await goto(resolve('/profiles'));
	}

	async function handleCancel() {
		await goto(resolve('/profiles'));
	}
</script>

<div class="max-w-2xl mx-auto">
	<h1 class="text-2xl font-bold mb-6">Create Server Profile</h1>

	<!-- Use keyed block to prevent ProfileForm recreation during polling updates -->
	{#key 'new-profile'}
		<ProfileForm {nextPort} initialModelPath={initialModelPath} onSubmit={handleSubmit} onCancel={handleCancel} />
	{/key}
</div>
