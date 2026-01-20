<script lang="ts">
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { models } from '$api';
	import type { ModelSearchResult, LocalModel, ModelCharacteristics } from '$api';
	import { systemStore, downloadsStore } from '$stores';
	import {
		ModelCard,
		DownloadProgressTile,
		ModelToggle,
		FilterModal,
		FilterChips,
		type FilterState,
		createEmptyFilters
	} from '$components/models';
	import { Button, Input, Card, Badge, ConfirmDialog } from '$components/ui';
	import { Search, HardDrive, Trash2, Filter } from 'lucide-svelte';

	let searchQuery = $state('');
	let searchResults = $state<ModelSearchResult[]>([]);
	let localModels = $state<LocalModel[]>([]);
	let loading = $state(false);
	let error = $state<string | null>(null);

	// Mode toggle: 'local' = My Models, 'online' = HuggingFace search
	let searchMode = $state<'local' | 'online'>('local');

	// Filter state
	let filterByMemory = $state(true);
	let filters = $state<FilterState>(createEmptyFilters());
	let showFilterModal = $state(false);

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
		if (searchMode === 'local') {
			return;
		}

		loading = true;
		error = null;

		try {
			const maxSize =
				filterByMemory && systemStore.memory ? systemStore.memory.mlx_recommended_gb : undefined;
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

	// Check if model characteristics match current filters
	function matchesFilters(characteristics: ModelCharacteristics | null | undefined): boolean {
		if (!characteristics) return true; // Show models without loaded characteristics

		// Architecture filter
		if (filters.architectures.length > 0) {
			if (
				!characteristics.architecture_family ||
				!filters.architectures.includes(characteristics.architecture_family)
			) {
				return false;
			}
		}

		// Multimodal filter
		if (filters.multimodal !== null) {
			if (filters.multimodal && !characteristics.is_multimodal) return false;
			if (!filters.multimodal && characteristics.is_multimodal) return false;
		}

		// Quantization filter
		if (filters.quantization.length > 0) {
			if (
				!characteristics.quantization_bits ||
				!filters.quantization.includes(characteristics.quantization_bits)
			) {
				return false;
			}
		}

		return true;
	}

	// Filter local models by search query and characteristic filters (client-side)
	const filteredLocalModels = $derived(() => {
		let result = localModels;

		// Apply text search filter
		if (searchQuery.trim()) {
			const query = searchQuery.toLowerCase();
			result = result.filter((m) => m.model_id.toLowerCase().includes(query));
		}

		// Apply characteristic filters
		result = result.filter((m) => matchesFilters(m.characteristics));

		return result;
	});

	// Get active downloads for pinned section (defined before displayResults as it's used there)
	const activeDownloads = $derived(() => {
		return downloadsStore
			.getAllDownloads()
			.filter(
				(d) => d.status === 'pending' || d.status === 'starting' || d.status === 'downloading'
			);
	});

	// Filter search results (online)
	// Also exclude models that are currently being downloaded (they appear in download section)
	const displayResults = $derived(() => {
		const activeIds = new Set(activeDownloads().map((d) => d.model_id));
		// Note: For online results, we don't have characteristics loaded yet
		// Filter matching will be done when characteristics are available
		return searchResults.filter((r) => !activeIds.has(r.model_id));
	});

	// Determine what to show based on mode
	const isLocalSearchMode = $derived(searchMode === 'local');
	const hasOnlineResults = $derived(searchResults.length > 0);

	// Filter state helpers
	const hasActiveFilters = $derived(
		filters.architectures.length > 0 ||
			filters.multimodal !== null ||
			filters.quantization.length > 0
	);

	const activeFilterCount = $derived(
		filters.architectures.length +
			(filters.multimodal !== null ? 1 : 0) +
			filters.quantization.length
	);

	function handleRemoveFilter(
		type: 'architecture' | 'multimodal' | 'quantization',
		value?: string | number
	) {
		if (type === 'architecture' && typeof value === 'string') {
			filters.architectures = filters.architectures.filter((a) => a !== value);
		} else if (type === 'multimodal') {
			filters.multimodal = null;
		} else if (type === 'quantization' && typeof value === 'number') {
			filters.quantization = filters.quantization.filter((q) => q !== value);
		}
	}
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
					<Search
						class="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground"
					/>
					<Input
						bind:value={searchQuery}
						placeholder={searchMode === 'local'
							? 'Filter downloaded models...'
							: 'Search MLX models...'}
						class="pl-10"
					/>
				</div>
				{#if searchMode === 'online'}
					<Button type="submit" disabled={loading}>
						{loading ? 'Searching...' : 'Search'}
					</Button>
				{/if}
			</form>

			<div class="flex items-center gap-4 mt-4">
				<ModelToggle bind:mode={searchMode} />

				{#if searchMode === 'online'}
					<label class="flex items-center gap-2 text-sm">
						<input type="checkbox" bind:checked={filterByMemory} class="rounded" />
						<span>Fits in memory</span>
					</label>
				{/if}

				<Button variant="outline" size="sm" onclick={() => (showFilterModal = true)}>
					<Filter class="w-4 h-4 mr-1" />
					Filters
					{#if hasActiveFilters}
						<Badge variant="secondary" class="ml-1">{activeFilterCount}</Badge>
					{/if}
				</Button>
			</div>

			{#if hasActiveFilters}
				<div class="mt-3">
					<FilterChips {filters} onRemove={handleRemoveFilter} />
				</div>
			{/if}
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

	<!-- Local Search Results (when "My Models" mode is selected) -->
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
			{:else if searchQuery || hasActiveFilters}
				<div class="text-center py-8 text-muted-foreground">
					No downloaded models match your filters.
				</div>
			{:else}
				<div class="text-center py-8 text-muted-foreground">
					No models downloaded yet. Switch to HuggingFace to search and download models.
				</div>
			{/if}
		</section>

		<!-- Online Search Results (when "HuggingFace" mode is selected) -->
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
	{:else if !searchQuery}
		<div class="text-center py-12 text-muted-foreground">
			Search for models on HuggingFace to download.
		</div>
	{/if}
</div>

<FilterModal bind:open={showFilterModal} bind:filters />

<ConfirmDialog
	bind:open={showDeleteConfirm}
	title="Delete Model"
	description={modelToDelete
		? `Are you sure you want to delete ${modelToDelete}? This will remove the model from your local cache and free up disk space.`
		: ''}
	confirmLabel="Delete"
	cancelLabel="Cancel"
	variant="destructive"
	onConfirm={confirmDeleteModel}
	onCancel={() => {
		modelToDelete = null;
	}}
/>
