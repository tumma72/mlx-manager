<script lang="ts">
	import { onMount } from 'svelte';
	import { SortableList, sortItems } from '@rodrigodagostino/svelte-sortable-list';
	import { settings } from '$lib/api/client';
	import type { BackendMapping, CloudCredential } from '$lib/api/types';
	import { Card } from '$lib/components/ui';
	import { Loader2, RefreshCw } from 'lucide-svelte';
	import RuleCard from './RuleCard.svelte';
	import RuleForm from './RuleForm.svelte';
	import RuleTestInput from './RuleTestInput.svelte';

	// State
	let rules = $state<BackendMapping[]>([]);
	let credentials = $state<CloudCredential[]>([]);
	let loading = $state(true);
	let reordering = $state(false);
	let error = $state<string | null>(null);

	// Transform rules to sortable items with string IDs
	const sortableItems = $derived(
		rules.map((rule) => ({
			id: `rule-${rule.id}`,
			rule
		}))
	);

	// Derive configured providers from credentials
	const configuredProviders = $derived(credentials.map((c) => c.backend_type));

	// Check if a rule's backend is unconfigured
	function hasWarning(rule: BackendMapping): boolean {
		return rule.backend_type !== 'local' && !configuredProviders.includes(rule.backend_type);
	}

	// Load data on mount
	async function loadData() {
		loading = true;
		error = null;
		try {
			const [rulesResult, credentialsResult] = await Promise.all([
				settings.listRules(),
				settings.listProviders()
			]);
			rules = rulesResult;
			credentials = credentialsResult;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load routing rules';
		} finally {
			loading = false;
		}
	}

	onMount(loadData);

	// Handle drag end
	async function handleDragEnd(e: SortableList.RootEvents['ondragend']) {
		const { draggedItemIndex, targetItemIndex, isCanceled } = e;

		if (isCanceled || typeof targetItemIndex !== 'number' || draggedItemIndex === targetItemIndex) {
			return;
		}

		// Optimistic update
		const newSortableItems = sortItems(sortableItems, draggedItemIndex, targetItemIndex);
		rules = newSortableItems.map((item) => item.rule);

		// Calculate new priorities (top = highest)
		const priorities = rules.map((rule, index) => ({
			id: rule.id,
			priority: rules.length - index
		}));

		reordering = true;
		try {
			await settings.updateRulePriorities(priorities);
		} catch {
			// Reload on error to restore correct order
			await loadData();
		} finally {
			reordering = false;
		}
	}

	// Handle rule deletion
	async function handleDelete(ruleId: number) {
		if (!confirm('Delete this routing rule?')) return;

		try {
			await settings.deleteRule(ruleId);
			await loadData();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to delete rule';
		}
	}

	// Handle rule created
	async function handleRuleCreated() {
		await loadData();
	}
</script>

<div class="space-y-6">
	<div class="flex items-center justify-between">
		<div>
			<h2 class="text-lg font-semibold">Model Routing Rules</h2>
			<p class="text-sm text-muted-foreground">
				Configure which models route to which backends. Drag to reorder priority.
			</p>
		</div>
		<button
			onclick={loadData}
			disabled={loading || reordering}
			class="p-2 text-muted-foreground hover:text-foreground disabled:opacity-50"
			title="Refresh rules"
		>
			<RefreshCw class="h-4 w-4 {loading || reordering ? 'animate-spin' : ''}" />
		</button>
	</div>

	{#if error}
		<div class="rounded-md bg-destructive/10 p-3 text-sm text-destructive">
			{error}
		</div>
	{/if}

	<!-- Test input -->
	<RuleTestInput {rules} />

	<!-- Add rule form -->
	<RuleForm onSave={handleRuleCreated} {configuredProviders} />

	<!-- Rules list -->
	<Card class="p-4">
		{#if loading}
			<div class="flex items-center justify-center py-8">
				<Loader2 class="h-6 w-6 animate-spin text-muted-foreground" />
				<span class="ml-2 text-muted-foreground">Loading rules...</span>
			</div>
		{:else if rules.length === 0}
			<div class="py-8 text-center text-muted-foreground">
				<p class="text-sm">No routing rules configured.</p>
				<p class="mt-1 text-xs">
					All requests will route to the local MLX server by default.
				</p>
			</div>
		{:else}
			<div class="relative {reordering ? 'pointer-events-none opacity-70' : ''}">
				{#if reordering}
					<div class="absolute inset-0 z-10 flex items-center justify-center">
						<Loader2 class="h-6 w-6 animate-spin" />
					</div>
				{/if}

				<p class="mb-3 text-xs text-muted-foreground">
					Higher rules have priority. Drag to reorder.
				</p>

				<SortableList.Root ondragend={handleDragEnd} class="space-y-2">
					{#each sortableItems as item, index (item.id)}
						<SortableList.Item id={item.id} {index}>
							<RuleCard
								rule={item.rule}
								hasWarning={hasWarning(item.rule)}
								onDelete={() => handleDelete(item.rule.id)}
							/>
						</SortableList.Item>
					{/each}
				</SortableList.Root>
			</div>
		{/if}
	</Card>
</div>

<style>
	/* Override sortable list styles to match our design */
	:global(.ssl-root) {
		list-style: none;
		padding: 0;
		margin: 0;
	}

	:global(.ssl-item) {
		list-style: none;
		padding: 0;
		margin: 0;
	}

	:global(.ssl-item--is-dragged) {
		opacity: 0.9;
		z-index: 50;
	}

	:global(.ssl-item--is-drop-target) {
		opacity: 0.5;
	}
</style>
