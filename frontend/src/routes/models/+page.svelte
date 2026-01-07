<script lang="ts">
	import { goto } from '$app/navigation';
	import { models } from '$api';
	import type { ModelSearchResult, LocalModel } from '$api';
	import { systemStore } from '$stores';
	import { ModelCard } from '$components/models';
	import { Button, Input, Card, Badge } from '$components/ui';
	import { Search, Filter, HardDrive } from 'lucide-svelte';

	let searchQuery = $state('');
	let searchResults = $state<ModelSearchResult[]>([]);
	let localModels = $state<LocalModel[]>([]);
	let loading = $state(false);
	let error = $state<string | null>(null);

	let filterByMemory = $state(true);
	let showLocalOnly = $state(false);

	// Load local models on mount
	$effect(() => {
		loadLocalModels();
		systemStore.refreshMemory();
	});

	async function loadLocalModels() {
		try {
			localModels = await models.listLocal();
		} catch (e) {
			console.error('Failed to load local models:', e);
		}
	}

	async function handleSearch() {
		if (!searchQuery.trim()) return;

		loading = true;
		error = null;

		try {
			const maxSize = filterByMemory && systemStore.memory ? systemStore.memory.mlx_recommended_gb : undefined;
			searchResults = await models.search(searchQuery, maxSize, 20);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Search failed';
		} finally {
			loading = false;
		}
	}

	function handleUseModel(modelId: string) {
		// Navigate to create profile with this model pre-filled
		goto(`/profiles/new?model=${encodeURIComponent(modelId)}`);
	}

	function handleModelDeleted() {
		loadLocalModels();
	}

	// Filter results based on local-only toggle
	const displayResults = $derived(
		showLocalOnly ? searchResults.filter((m) => m.is_downloaded) : searchResults
	);
</script>

<div class="space-y-6">
	<div class="flex items-center justify-between">
		<h1 class="text-2xl font-bold">Model Browser</h1>
		{#if systemStore.memory}
			<Badge variant="outline" class="text-sm">
				<HardDrive class="w-4 h-4 mr-1" />
				Recommended: &lt;{systemStore.memory.mlx_recommended_gb.toFixed(0)} GB
			</Badge>
		{/if}
	</div>

	<!-- Search -->
	<Card class="p-4">
		<form
			onsubmit={(e) => {
				e.preventDefault();
				handleSearch();
			}}
			class="flex gap-4"
		>
			<div class="flex-1 relative">
				<Search class="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
				<Input
					bind:value={searchQuery}
					placeholder="Search mlx-community models..."
					class="pl-10"
				/>
			</div>
			<Button type="submit" disabled={loading}>
				{loading ? 'Searching...' : 'Search'}
			</Button>
		</form>

		<div class="flex items-center gap-4 mt-4">
			<label class="flex items-center gap-2 text-sm">
				<input type="checkbox" bind:checked={filterByMemory} class="rounded" />
				<span>Fits in memory (&lt;{systemStore.memory?.mlx_recommended_gb.toFixed(0) ?? 80} GB)</span>
			</label>
			<label class="flex items-center gap-2 text-sm">
				<input type="checkbox" bind:checked={showLocalOnly} class="rounded" />
				<span>Downloaded only</span>
			</label>
		</div>
	</Card>

	{#if error}
		<div class="text-center py-8 text-red-500">{error}</div>
	{/if}

	<!-- Search Results -->
	{#if searchResults.length > 0}
		<section>
			<h2 class="text-lg font-semibold mb-4">
				Search Results ({displayResults.length})
			</h2>
			<div class="grid gap-4">
				{#each displayResults as model (model.model_id)}
					<ModelCard {model} onUse={handleUseModel} onDeleted={handleModelDeleted} />
				{/each}
			</div>
		</section>
	{:else if searchQuery && !loading}
		<div class="text-center py-8 text-muted-foreground">
			No models found. Try a different search term.
		</div>
	{/if}

	<!-- Local Models -->
	{#if localModels.length > 0 && !searchQuery}
		<section>
			<h2 class="text-lg font-semibold mb-4">
				Downloaded Models ({localModels.length})
			</h2>
			<div class="grid gap-4">
				{#each localModels as model (model.model_id)}
					<Card class="p-4">
						<div class="flex items-center justify-between">
							<div>
								<h3 class="font-medium">{model.model_id}</h3>
								<p class="text-sm text-muted-foreground">
									{model.size_gb.toFixed(2)} GB
								</p>
							</div>
							<Button size="sm" onclick={() => handleUseModel(model.model_id)}>
								Use
							</Button>
						</div>
					</Card>
				{/each}
			</div>
		</section>
	{:else if !searchQuery}
		<div class="text-center py-12 text-muted-foreground">
			Search for models to download, or browse your local models.
		</div>
	{/if}
</div>
