<script lang="ts">
	import { onMount } from 'svelte';
	import { profileStore, serverStore } from '$stores';
	import { Card, Button, Badge } from '$components/ui';
	import { Plus, Edit, Trash2, Copy, Play, Square } from 'lucide-svelte';

	onMount(() => {
		profileStore.refresh();
		serverStore.refresh();
	});

	async function handleDelete(id: number) {
		if (!confirm('Are you sure you want to delete this profile?')) return;

		try {
			// Stop server if running
			if (serverStore.isRunning(id)) {
				await serverStore.stop(id);
			}
			await profileStore.delete(id);
		} catch (e) {
			alert(e instanceof Error ? e.message : 'Failed to delete profile');
		}
	}

	async function handleDuplicate(id: number, name: string) {
		const newName = prompt('Enter name for the duplicate profile:', `${name} (copy)`);
		if (!newName) return;

		try {
			await profileStore.duplicate(id, newName);
		} catch (e) {
			alert(e instanceof Error ? e.message : 'Failed to duplicate profile');
		}
	}

	async function handleToggleServer(id: number) {
		try {
			if (serverStore.isRunning(id)) {
				await serverStore.stop(id);
			} else {
				await serverStore.start(id);
			}
		} catch (e) {
			alert(e instanceof Error ? e.message : 'Failed to toggle server');
		}
	}
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
				{@const isRunning = serverStore.isRunning(profile.id)}
				<Card class="p-4">
					<div class="flex items-start justify-between">
						<div class="flex-1">
							<div class="flex items-center gap-2">
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
							<div class="mt-2 text-sm text-muted-foreground">
								<span class="font-mono">{profile.model_path}</span>
								<span class="mx-2">•</span>
								<span>Port {profile.port}</span>
							</div>
						</div>

						<div class="flex items-center gap-2">
							<Button
								variant="outline"
								size="icon"
								onclick={() => handleToggleServer(profile.id)}
								title={isRunning ? 'Stop' : 'Start'}
							>
								{#if isRunning}
									<Square class="w-4 h-4" />
								{:else}
									<Play class="w-4 h-4" />
								{/if}
							</Button>

							<Button
								variant="outline"
								size="icon"
								href={`/profiles/${profile.id}`}
								title="Edit"
							>
								<Edit class="w-4 h-4" />
							</Button>

							<Button
								variant="outline"
								size="icon"
								onclick={() => handleDuplicate(profile.id, profile.name)}
								title="Duplicate"
							>
								<Copy class="w-4 h-4" />
							</Button>

							<Button
								variant="outline"
								size="icon"
								onclick={() => handleDelete(profile.id)}
								title="Delete"
							>
								<Trash2 class="w-4 h-4" />
							</Button>
						</div>
					</div>
				</Card>
			{/each}
		</div>
	{/if}
</div>
