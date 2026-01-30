<script lang="ts">
	import { settings } from '$lib/api/client';
	import { Button, Input, Select } from '$lib/components/ui';
	import { Plus, Loader2, AlertTriangle } from 'lucide-svelte';
	import type { BackendType, PatternType } from '$lib/api/types';

	interface Props {
		onSave: () => void;
		configuredProviders: BackendType[];
	}

	let { onSave, configuredProviders }: Props = $props();

	let patternType = $state<PatternType>('prefix');
	let patternValue = $state('');
	let backendType = $state<BackendType>('local');
	let backendModel = $state('');
	let fallbackBackend = $state<BackendType | ''>('');
	let saving = $state(false);
	let error = $state<string | null>(null);

	// Use $derived.by() to maintain reactivity for configuredProviders prop
	// This ensures the warning updates when the parent's configuredProviders changes
	const showWarning = $derived.by(() => {
		return backendType !== 'local' && !configuredProviders.includes(backendType);
	});

	const patternPlaceholders: Record<PatternType, string> = {
		exact: 'gpt-4-turbo',
		prefix: 'gpt-',
		regex: '^claude-3-.*$'
	};

	async function handleSubmit(e: SubmitEvent) {
		e.preventDefault();
		if (!patternValue.trim()) {
			error = 'Pattern is required';
			return;
		}

		saving = true;
		error = null;

		try {
			await settings.createRule({
				pattern_type: patternType,
				model_pattern: patternValue.trim(),
				backend_type: backendType,
				backend_model: backendModel.trim() || undefined,
				fallback_backend: fallbackBackend || undefined
			});
			// Reset form
			patternValue = '';
			backendModel = '';
			fallbackBackend = '';
			onSave();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to create rule';
		} finally {
			saving = false;
		}
	}

	const backendOptions: { value: BackendType; label: string }[] = [
		{ value: 'local', label: 'Local (MLX)' },
		{ value: 'openai', label: 'OpenAI' },
		{ value: 'anthropic', label: 'Anthropic' }
	];

	const patternOptions: { value: PatternType; label: string }[] = [
		{ value: 'exact', label: 'Exact match' },
		{ value: 'prefix', label: 'Prefix match' },
		{ value: 'regex', label: 'Regex' }
	];
</script>

<form onsubmit={handleSubmit} class="space-y-4 rounded-lg border bg-card p-4">
	<h3 class="text-sm font-medium">Add Routing Rule</h3>

	{#if error}
		<div class="rounded-md bg-destructive/10 p-3 text-sm text-destructive">
			{error}
		</div>
	{/if}

	<div class="grid gap-4 sm:grid-cols-2">
		<!-- Pattern Type -->
		<div class="space-y-2">
			<label for="pattern-type" class="text-sm font-medium">Pattern Type</label>
			<Select id="pattern-type" bind:value={patternType}>
				{#each patternOptions as option (option.value)}
					<option value={option.value}>{option.label}</option>
				{/each}
			</Select>
		</div>

		<!-- Pattern Value -->
		<div class="space-y-2">
			<label for="pattern-value" class="text-sm font-medium">Pattern</label>
			<Input
				id="pattern-value"
				bind:value={patternValue}
				placeholder={patternPlaceholders[patternType]}
				required
			/>
		</div>

		<!-- Backend Type -->
		<div class="space-y-2">
			<label for="backend-type" class="text-sm font-medium">Route to Backend</label>
			<Select id="backend-type" bind:value={backendType}>
				{#each backendOptions as option (option.value)}
					<option value={option.value}>{option.label}</option>
				{/each}
			</Select>
			{#if showWarning}
				<p class="flex items-center gap-1 text-xs text-yellow-600 dark:text-yellow-400">
					<AlertTriangle class="h-3 w-3" />
					Provider not configured
				</p>
			{/if}
		</div>

		<!-- Backend Model (optional) -->
		<div class="space-y-2">
			<label for="backend-model" class="text-sm font-medium">
				Backend Model <span class="text-muted-foreground">(optional)</span>
			</label>
			<Input
				id="backend-model"
				bind:value={backendModel}
				placeholder="Override model name for cloud"
			/>
		</div>

		<!-- Fallback Backend (optional) -->
		<div class="space-y-2">
			<label for="fallback-backend" class="text-sm font-medium">
				Fallback Backend <span class="text-muted-foreground">(optional)</span>
			</label>
			<Select id="fallback-backend" bind:value={fallbackBackend}>
				<option value="">No fallback</option>
				{#each backendOptions as option (option.value)}
					{#if option.value !== backendType}
						<option value={option.value}>{option.label}</option>
					{/if}
				{/each}
			</Select>
		</div>
	</div>

	<div class="flex justify-end">
		<Button type="submit" disabled={saving}>
			{#if saving}
				<Loader2 class="mr-2 h-4 w-4 animate-spin" />
			{:else}
				<Plus class="mr-2 h-4 w-4" />
			{/if}
			Add Rule
		</Button>
	</div>
</form>
