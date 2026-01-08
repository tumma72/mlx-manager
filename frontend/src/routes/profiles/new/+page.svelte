<script lang="ts">
	import { goto } from '$app/navigation';
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
		goto('/profiles');
	}

	function handleCancel() {
		goto('/profiles');
	}
</script>

<div class="max-w-2xl mx-auto">
	<h1 class="text-2xl font-bold mb-6">Create Server Profile</h1>

	<ProfileForm {nextPort} onSubmit={handleSubmit} onCancel={handleCancel} />
</div>
