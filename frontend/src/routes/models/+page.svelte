<script lang="ts">
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { models } from '$api';
	import type { ModelSearchResult, LocalModel } from '$api';
	import { systemStore, downloadsStore } from '$stores';
	import { ModelCard, DownloadProgressTile } from '$components/models';
	import { Button, Input, Card, Badge, ConfirmDialog } from '$components/ui';
	import { Search, HardDrive, Trash2 } from 'lucide-svelte';

	let searchQuery = $state('');
	let searchResults = $state<ModelSearchResult[]>([]);
	let localModels = $state<LocalModel[]>([]);
	let loading = $state(false);
	let error = $state<string | null>(null);

	let filterByMemory = $state(true);
	let showLocalOnly = $state(false);

	// Delete confirmation state
	let showDeleteConfirm = $state(false);
	let modelToDelete = $state<string | null>(null);
	let deleting = $state(false);

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

		// If searching local only, no need for API call - just use client-side filter
		if (showLocalOnly) {
			return;
		}

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

	async function handleUseModel(modelId: string) {
		// Navigate to create profile with this model pre-filled
		const profileUrl = `${resolve('/profiles/new')}?model=${encodeURIComponent(modelId)}`;
		// eslint-disable-next-line svelte/no-navigation-without-resolve -- query params appended to resolved path
		await goto(profileUrl);
	}

	function handleModelDeleted() {
		loadLocalModels();
	}

	function requestDeleteModel(modelId: string) {
		modelToDelete = modelId;
		showDeleteConfirm = true;
	}

	async function confirmDeleteModel() {
		if (!modelToDelete) return;

		deleting = true;
		try {
			await models.delete(modelToDelete);
			await loadLocalModels();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Delete failed';
		} finally {
			deleting = false;
			modelToDelete = null;
		}
	}

	// Filter local models by search query (client-side)
	const filteredLocalModels = $derived(() => {
		if (!searchQuery.trim()) return localModels;
		const query = searchQuery.toLowerCase();
		return localModels.filter((m) => m.model_id.toLowerCase().includes(query));
	});

	// Get active downloads for pinned section (defined before displayResults as it's used there)
	const activeDownloads = $derived(() => {
		return downloadsStore.getAllDownloads().filter(
			(d) => d.status === 'pending' || d.status === 'starting' || d.status === 'downloading'
		);
	});

	// Filter search results (online) based on local-only toggle
	// Also exclude models that are currently being downloaded (they appear in download section)
	const displayResults = $derived(() => {
		const activeIds = new Set(activeDownloads().map((d) => d.model_id));
		const resultsToShow = showLocalOnly ? searchResults.filter((m) => m.is_downloaded) : searchResults;
		return resultsToShow.filter((r) => !activeIds.has(r.model_id));
	});

	// Determine what to show based on mode
	const isLocalSearchMode = $derived(showLocalOnly);
	const hasOnlineResults = $derived(searchResults.length > 0);
</script>

<div class="space-y-6">
	<!-- Sticky Header and Search Section -->
	<div class="sticky top-0 z-20 bg-background pt-2 pb-4 -mx-4 px-4">
		<div class="flex items-center justify-between mb-4">
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
						placeholder={showLocalOnly ? 'Filter downloaded models...' : 'Search mlx-community models...'}
						class="pl-10"
					/>
				</div>
				{#if !showLocalOnly}
					<Button type="submit" disabled={loading}>
						{loading ? 'Searching...' : 'Search'}
					</Button>
				{/if}
			</form>

			<div class="flex items-center gap-4 mt-4">
				{#if !showLocalOnly}
					<label class="flex items-center gap-2 text-sm">
						<input type="checkbox" bind:checked={filterByMemory} class="rounded" />
						<span>Fits in memory (&lt;{systemStore.memory?.mlx_recommended_gb.toFixed(0) ?? 80} GB)</span>
					</label>
				{/if}
				<label class="flex items-center gap-2 text-sm">
					<input type="checkbox" bind:checked={showLocalOnly} class="rounded" />
					<span>Downloaded only</span>
				</label>
			</div>
		</Card>
	</div>

	<!-- Active Downloads Section (below sticky search) -->
	{#if activeDownloads().length > 0}
		<div class="bg-background pb-4 border-b mb-2">
			<h2 class="text-sm font-medium text-muted-foreground mb-2">
				Downloading ({activeDownloads().length})
			</h2>
			<div class="space-y-2">
				{#each activeDownloads() as download (download.model_id)}
					<DownloadProgressTile {download} />
				{/each}
			</div>
		</div>
	{/if}

	{#if error}
		<div class="text-center py-8 text-red-500 dark:text-red-400">{error}</div>
	{/if}

	<!-- Local Search Results (when "Downloaded only" is checked) -->
	{#if isLocalSearchMode}
		<section>
			<h2 class="text-lg font-semibold mb-4">
				Downloaded Models ({filteredLocalModels().length})
			</h2>
			{#if filteredLocalModels().length > 0}
				<div class="grid gap-4">
					{#each filteredLocalModels() as model (model.model_id)}
						<Card class="p-4">
							<div class="flex items-center justify-between">
								<div>
									<h3 class="font-medium">{model.model_id}</h3>
									<p class="text-sm text-muted-foreground">
										{model.size_gb.toFixed(2)} GB
									</p>
								</div>
								<div class="flex gap-2">
									<Button
										variant="outline"
										size="sm"
										onclick={() => requestDeleteModel(model.model_id)}
										disabled={deleting}
									>
										<Trash2 class="w-4 h-4 mr-1" />
										Delete
									</Button>
									<Button size="sm" onclick={() => handleUseModel(model.model_id)}>
										Use
									</Button>
								</div>
							</div>
						</Card>
					{/each}
				</div>
			{:else if searchQuery}
				<div class="text-center py-8 text-muted-foreground">
					No downloaded models match "{searchQuery}".
				</div>
			{:else}
				<div class="text-center py-8 text-muted-foreground">
					No models downloaded yet. Uncheck "Downloaded only" to search HuggingFace.
				</div>
			{/if}
		</section>

	<!-- Online Search Results (when "Downloaded only" is unchecked) -->
	{:else if hasOnlineResults}
		<section>
			<h2 class="text-lg font-semibold mb-4">
				Search Results ({displayResults().length})
			</h2>
			<div class="grid gap-4">
				{#each displayResults() as model (model.model_id)}
					<ModelCard {model} onUse={handleUseModel} onDeleted={handleModelDeleted} />
				{/each}
			</div>
		</section>
	{:else if searchQuery && !loading}
		<div class="text-center py-8 text-muted-foreground">
			No models found. Try a different search term.
		</div>
	{/if}

	<!-- Local Models Section (only show when no search and not in local-only mode) -->
	{#if localModels.length > 0 && !searchQuery && !isLocalSearchMode}
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
							<div class="flex gap-2">
								<Button
									variant="outline"
									size="sm"
									onclick={() => requestDeleteModel(model.model_id)}
									disabled={deleting}
								>
									<Trash2 class="w-4 h-4 mr-1" />
									Delete
								</Button>
								<Button size="sm" onclick={() => handleUseModel(model.model_id)}>
									Use
								</Button>
							</div>
						</div>
					</Card>
				{/each}
			</div>
		</section>
	{:else if !searchQuery && !isLocalSearchMode}
		<div class="text-center py-12 text-muted-foreground">
			Search for models to download, or check "Downloaded only" to filter local models.
		</div>
	{/if}
</div>

<ConfirmDialog
	bind:open={showDeleteConfirm}
	title="Delete Model"
	description={modelToDelete ? `Are you sure you want to delete ${modelToDelete}? This will remove the model from your local cache and free up disk space.` : ''}
	confirmLabel="Delete"
	cancelLabel="Cancel"
	variant="destructive"
	onConfirm={confirmDeleteModel}
	onCancel={() => { modelToDelete = null; }}
/>
