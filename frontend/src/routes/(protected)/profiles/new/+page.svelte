<script lang="ts">
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { page } from '$app/stores';
	import { onMount } from 'svelte';
	import { profileStore } from '$stores';
	import type { ServerProfileCreate, ServerProfileUpdate } from '$api';
	import { models as modelsApi } from '$api';
	import { ProfileForm } from '$components/profiles';

	let initialModelId = $state<number | undefined>(undefined);

	onMount(async () => {
		// Check for model query param
		const modelParam = $page.url.searchParams.get('model');
		if (modelParam) {
			// Look up model_id from repo_id
			try {
				const downloaded = await modelsApi.listDownloaded();
				const found = downloaded.find((m) => m.repo_id === modelParam);
				if (found) {
					initialModelId = found.id;
				}
			} catch {
				// Ignore - user can select manually
			}
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
		<ProfileForm initialModelId={initialModelId} onSubmit={handleSubmit} onCancel={handleCancel} />
	{/key}
</div>
