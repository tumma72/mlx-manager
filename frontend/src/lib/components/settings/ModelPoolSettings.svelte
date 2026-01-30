<script lang="ts">
	import { onMount } from 'svelte';
	import { cn } from '$lib/utils';
	import { models, settings } from '$lib/api/client';
	import { systemStore } from '$stores';
	import type { LocalModel } from '$lib/api/types';
	import { Card, Button, Select } from '$components/ui';
	import { Loader2, Save, X, ChevronDown } from 'lucide-svelte';

	// Types for pool configuration (local until 11-02 provides them)
	type MemoryLimitMode = 'percent' | 'gb';
	type EvictionPolicy = 'lru' | 'lfu' | 'ttl';

	interface ServerPoolConfig {
		memory_limit_mode: MemoryLimitMode;
		memory_limit_value: number;
		eviction_policy: EvictionPolicy;
		preload_models: string[];
	}

	// Component state
	let localModels = $state<LocalModel[]>([]);
	let loading = $state(true);
	let saving = $state(false);
	let error = $state<string | null>(null);
	let successMessage = $state<string | null>(null);
	let showAdvanced = $state(false);

	// Form state
	let memoryMode = $state<MemoryLimitMode>('percent');
	let memoryValue = $state(80);
	let evictionPolicy = $state<EvictionPolicy>('lru');
	let preloadModels = $state<string[]>([]);

	// Preload selector state
	let preloadSearchQuery = $state('');
	let preloadDropdownOpen = $state(false);

	// Derived values
	const maxMemoryGb = $derived(systemStore.memory?.total_gb ?? 64);
	const sliderMax = $derived(memoryMode === 'percent' ? 100 : Math.floor(maxMemoryGb));
	const sliderMin = $derived(memoryMode === 'percent' ? 10 : 1);
	const sliderStep = $derived(memoryMode === 'percent' ? 5 : 1);
	const displayValue = $derived(memoryMode === 'percent' ? `${memoryValue}%` : `${memoryValue} GB`);

	// Filter local models for preload selector
	const filteredModels = $derived(
		localModels.filter(
			(m) =>
				!preloadModels.includes(m.model_id) &&
				m.model_id.toLowerCase().includes(preloadSearchQuery.toLowerCase())
		)
	);

	const evictionOptions: { value: EvictionPolicy; label: string; description: string }[] = [
		{
			value: 'lru',
			label: 'LRU (Least Recently Used)',
			description: 'Evict models not used for longest'
		},
		{
			value: 'lfu',
			label: 'LFU (Least Frequently Used)',
			description: 'Evict models with fewest requests'
		},
		{ value: 'ttl', label: 'TTL (Time To Live)', description: 'Evict models after idle timeout' }
	];

	onMount(async () => {
		try {
			// Ensure system memory is loaded
			await systemStore.refreshMemory();

			// Load pool config and local models in parallel
			const [config, modelsResult] = await Promise.all([settings.getPoolConfig(), models.listLocal()]);

			// Initialize form from config
			memoryMode = config.memory_limit_mode;
			memoryValue = config.memory_limit_value;
			evictionPolicy = config.eviction_policy;
			preloadModels = config.preload_models;
			localModels = modelsResult;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load settings';
		} finally {
			loading = false;
		}
	});

	async function handleSave() {
		saving = true;
		error = null;
		successMessage = null;
		try {
			await settings.updatePoolConfig({
				memory_limit_mode: memoryMode,
				memory_limit_value: memoryValue,
				eviction_policy: evictionPolicy,
				preload_models: preloadModels
			});
			successMessage = 'Settings saved successfully';
			setTimeout(() => {
				successMessage = null;
			}, 3000);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to save settings';
		} finally {
			saving = false;
		}
	}

	function handleModeToggle() {
		if (memoryMode === 'percent') {
			// Switch to GB mode
			memoryMode = 'gb';
			// Convert percentage to GB
			memoryValue = Math.round((memoryValue / 100) * maxMemoryGb);
			// Clamp to valid range
			memoryValue = Math.max(1, Math.min(memoryValue, Math.floor(maxMemoryGb)));
		} else {
			// Switch to percent mode
			memoryMode = 'percent';
			// Convert GB to percentage
			memoryValue = Math.round((memoryValue / maxMemoryGb) * 100);
			// Clamp to valid range and round to nearest 5
			memoryValue = Math.max(10, Math.min(100, Math.round(memoryValue / 5) * 5));
		}
	}

	function addPreloadModel(modelId: string) {
		if (!preloadModels.includes(modelId)) {
			preloadModels = [...preloadModels, modelId];
		}
		preloadSearchQuery = '';
		preloadDropdownOpen = false;
	}

	function removePreloadModel(modelId: string) {
		preloadModels = preloadModels.filter((m) => m !== modelId);
	}

	// Handle slider input
	function handleSliderInput(event: Event) {
		const target = event.target as HTMLInputElement;
		memoryValue = parseInt(target.value, 10);
	}
</script>

{#if loading}
	<Card class="p-6">
		<div class="flex items-center justify-center py-8">
			<Loader2 class="w-6 h-6 animate-spin text-muted-foreground" />
			<span class="ml-2 text-muted-foreground">Loading settings...</span>
		</div>
	</Card>
{:else}
	<Card class="p-6 space-y-6">
		<!-- Error message -->
		{#if error}
			<div class="bg-destructive/10 text-destructive px-4 py-3 rounded-md text-sm">
				{error}
			</div>
		{/if}

		<!-- Success message -->
		{#if successMessage}
			<div class="bg-green-500/10 text-green-600 dark:text-green-400 px-4 py-3 rounded-md text-sm">
				{successMessage}
			</div>
		{/if}

		<!-- Memory Limit Section -->
		<div class="space-y-4">
			<div class="flex items-center justify-between">
				<h3 class="text-sm font-medium">Memory Limit</h3>
				<button
					type="button"
					onclick={handleModeToggle}
					class={cn(
						'relative inline-flex h-6 w-20 items-center rounded-full transition-colors',
						'bg-muted border border-input'
					)}
				>
					<span
						class={cn(
							'absolute w-1/2 h-5 rounded-full bg-primary transition-transform',
							memoryMode === 'gb' ? 'translate-x-9' : 'translate-x-0.5'
						)}
					></span>
					<span
						class={cn(
							'w-1/2 text-xs font-medium text-center z-10 transition-colors',
							memoryMode === 'percent' ? 'text-primary-foreground' : 'text-muted-foreground'
						)}
					>
						%
					</span>
					<span
						class={cn(
							'w-1/2 text-xs font-medium text-center z-10 transition-colors',
							memoryMode === 'gb' ? 'text-primary-foreground' : 'text-muted-foreground'
						)}
					>
						GB
					</span>
				</button>
			</div>

			<div class="space-y-2">
				<div class="flex items-center justify-between">
					<span class="text-3xl font-bold text-foreground">{displayValue}</span>
					<span class="text-sm text-muted-foreground">
						{#if memoryMode === 'percent'}
							of {maxMemoryGb.toFixed(1)} GB total
						{:else}
							of {maxMemoryGb.toFixed(1)} GB available
						{/if}
					</span>
				</div>
				<input
					type="range"
					min={sliderMin}
					max={sliderMax}
					step={sliderStep}
					value={memoryValue}
					oninput={handleSliderInput}
					class="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer accent-primary"
				/>
				<div class="flex justify-between text-xs text-muted-foreground">
					<span>{sliderMin}{memoryMode === 'percent' ? '%' : ' GB'}</span>
					<span>{sliderMax}{memoryMode === 'percent' ? '%' : ' GB'}</span>
				</div>
			</div>
		</div>

		<hr class="border-border" />

		<!-- Preload Models Section -->
		<div class="space-y-4">
			<div>
				<h3 class="text-sm font-medium">Preload Models</h3>
				<p class="text-xs text-muted-foreground mt-1">
					Models to load on server startup. These will never be evicted.
				</p>
			</div>

			<!-- Selected preload models as tags -->
			{#if preloadModels.length > 0}
				<div class="flex flex-wrap gap-2">
					{#each preloadModels as modelId (modelId)}
						<span
							class="inline-flex items-center gap-1 px-2 py-1 bg-primary/10 text-primary rounded-md text-sm"
						>
							{modelId.split('/').pop()}
							<button
								type="button"
								onclick={() => removePreloadModel(modelId)}
								class="hover:bg-primary/20 rounded p-0.5"
								title="Remove"
							>
								<X class="w-3 h-3" />
							</button>
						</span>
					{/each}
				</div>
			{/if}

			<!-- Preload model selector -->
			<div class="relative">
				<div class="relative">
					<input
						type="text"
						placeholder="Search downloaded models..."
						bind:value={preloadSearchQuery}
						onfocus={() => (preloadDropdownOpen = true)}
						class="w-full h-10 px-3 py-2 text-sm rounded-md border border-input bg-background ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
					/>
					<ChevronDown
						class="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground"
					/>
				</div>

				{#if preloadDropdownOpen && filteredModels.length > 0}
					<!-- svelte-ignore a11y_no_static_element_interactions -->
					<div
						class="absolute z-10 w-full mt-1 max-h-60 overflow-auto rounded-md border bg-popover shadow-md"
						onmouseleave={() => (preloadDropdownOpen = false)}
					>
						{#each filteredModels as model (model.model_id)}
							<button
								type="button"
								onclick={() => addPreloadModel(model.model_id)}
								class="w-full px-3 py-2 text-sm text-left hover:bg-accent flex items-center justify-between"
							>
								<span class="truncate">{model.model_id}</span>
								<span class="text-xs text-muted-foreground ml-2">
									{model.size_gb.toFixed(1)} GB
								</span>
							</button>
						{/each}
					</div>
				{/if}

				{#if preloadDropdownOpen && filteredModels.length === 0 && localModels.length > 0}
					<div
						class="absolute z-10 w-full mt-1 p-3 rounded-md border bg-popover shadow-md text-sm text-muted-foreground text-center"
					>
						{#if preloadSearchQuery}
							No models match "{preloadSearchQuery}"
						{:else}
							All downloaded models are already selected
						{/if}
					</div>
				{/if}

				{#if preloadDropdownOpen && localModels.length === 0}
					<div
						class="absolute z-10 w-full mt-1 p-3 rounded-md border bg-popover shadow-md text-sm text-muted-foreground text-center"
					>
						No downloaded models available
					</div>
				{/if}
			</div>
		</div>

		<hr class="border-border" />

		<!-- Advanced Options (Collapsible) -->
		<div class="space-y-4">
			<button
				type="button"
				onclick={() => (showAdvanced = !showAdvanced)}
				class="flex items-center gap-2 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
			>
				<ChevronDown
					class={cn('w-4 h-4 transition-transform', showAdvanced ? 'rotate-180' : '')}
				/>
				Advanced Options
			</button>

			{#if showAdvanced}
				<div class="pl-6 space-y-4">
					<div class="space-y-2">
						<label for="eviction-policy" class="text-sm font-medium">Eviction Policy</label>
						<Select id="eviction-policy" bind:value={evictionPolicy} class="w-full">
							{#each evictionOptions as option (option.value)}
								<option value={option.value}>{option.label}</option>
							{/each}
						</Select>
						<p class="text-xs text-muted-foreground">
							{evictionOptions.find((o) => o.value === evictionPolicy)?.description}
						</p>
					</div>
				</div>
			{/if}
		</div>

		<hr class="border-border" />

		<!-- Save Button -->
		<div class="flex justify-end">
			<Button onclick={handleSave} disabled={saving}>
				{#if saving}
					<Loader2 class="w-4 h-4 mr-2 animate-spin" />
					Saving...
				{:else}
					<Save class="w-4 h-4 mr-2" />
					Save Changes
				{/if}
			</Button>
		</div>
	</Card>
{/if}
