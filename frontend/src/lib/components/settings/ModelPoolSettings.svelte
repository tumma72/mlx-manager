<script lang="ts">
	import { onMount } from 'svelte';
	import { cn } from '$lib/utils';
	import { settings } from '$lib/api/client';
	import { systemStore } from '$stores';
	import type { PreloadedProfileInfo } from '$lib/api/types';
	import { Card, Button, Select } from '$components/ui';
	import { Loader2, Save, ChevronDown, ExternalLink } from 'lucide-svelte';

	// Types for pool configuration
	type MemoryLimitMode = 'percent' | 'gb';
	type EvictionPolicy = 'lru' | 'lfu' | 'ttl';

	// Component state
	let loading = $state(true);
	let saving = $state(false);
	let error = $state<string | null>(null);
	let successMessage = $state<string | null>(null);
	let showAdvanced = $state(false);

	// Form state
	let memoryMode = $state<MemoryLimitMode>('percent');
	let memoryValue = $state(80);
	let evictionPolicy = $state<EvictionPolicy>('lru');
	let preloadedProfiles = $state<PreloadedProfileInfo[]>([]);

	// Derived values
	const maxMemoryGb = $derived(systemStore.memory?.total_gb ?? 64);
	const sliderMax = $derived(memoryMode === 'percent' ? 100 : Math.floor(maxMemoryGb));
	const sliderMin = $derived(memoryMode === 'percent' ? 10 : 1);
	const sliderStep = $derived(memoryMode === 'percent' ? 5 : 1);
	const displayValue = $derived(memoryMode === 'percent' ? `${memoryValue}%` : `${memoryValue} GB`);

	const evictionOptions: { value: EvictionPolicy; label: string; description: string }[] = [
		{
			value: 'lru',
			label: 'LRU (Least Recently Used)',
			description: 'Evict profiles not used for longest'
		},
		{
			value: 'lfu',
			label: 'LFU (Least Frequently Used)',
			description: 'Evict profiles with fewest requests'
		},
		{
			value: 'ttl',
			label: 'TTL (Time To Live)',
			description: 'Evict profiles after idle timeout'
		}
	];

	onMount(async () => {
		try {
			// Ensure system memory is loaded
			await systemStore.refreshMemory();

			// Load pool config
			const config = await settings.getPoolConfig();

			// Initialize form from config
			memoryMode = config.memory_limit_mode;
			memoryValue = config.memory_limit_value;
			evictionPolicy = config.eviction_policy;
			preloadedProfiles = config.preloaded_profiles;
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
			const result = await settings.updatePoolConfig({
				memory_limit_mode: memoryMode,
				memory_limit_value: memoryValue,
				eviction_policy: evictionPolicy
			});
			preloadedProfiles = result.preloaded_profiles;
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

		<!-- Pre-loaded Profiles Section (read-only) -->
		<div class="space-y-4">
			<div>
				<h3 class="text-sm font-medium">Pre-loaded Profiles</h3>
				<p class="text-xs text-muted-foreground mt-1">
					Profiles loaded at startup and protected from eviction. Enable "Auto-load on startup"
					in each profile's settings to add it here.
				</p>
			</div>

			{#if preloadedProfiles.length > 0}
				<div class="space-y-2">
					{#each preloadedProfiles as profile (profile.id)}
						<a
							href="/profiles?edit={profile.id}"
							class="flex items-center justify-between px-3 py-2 rounded-md bg-primary/5 border border-primary/10 hover:bg-primary/10 transition-colors group"
						>
							<div class="flex items-center gap-2 min-w-0">
								<span class="text-sm font-medium truncate">{profile.name}</span>
								<span
									class="text-xs px-1.5 py-0.5 rounded bg-muted text-muted-foreground shrink-0"
								>
									{profile.profile_type}
								</span>
							</div>
							<div class="flex items-center gap-2 shrink-0">
								{#if profile.model_name}
									<span class="text-xs text-muted-foreground">{profile.model_name}</span>
								{/if}
								<ExternalLink
									class="w-3.5 h-3.5 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity"
								/>
							</div>
						</a>
					{/each}
				</div>
			{:else}
				<div
					class="text-sm text-muted-foreground text-center py-4 border border-dashed rounded-md"
				>
					No profiles configured for auto-loading.
					<br />
					<span class="text-xs">
						Enable "Auto-load on startup" in a profile's settings to preload it.
					</span>
				</div>
			{/if}
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
