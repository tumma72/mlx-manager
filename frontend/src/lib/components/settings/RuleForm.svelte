<script lang="ts">
	import { settings } from '$lib/api/client';
	import { Button, Input, Select } from '$lib/components/ui';
	import { Plus, Pencil, Loader2, AlertTriangle } from 'lucide-svelte';
	import type { BackendMapping, BackendType, PatternType } from '$lib/api/types';
	import { profileStore } from '$lib/stores';

	interface Props {
		onSave: () => void;
		configuredProviders: BackendType[];
		rule?: BackendMapping;
		onCancel?: () => void;
	}

	let { onSave, configuredProviders, rule = undefined, onCancel = undefined }: Props = $props();

	const isEditMode = $derived(rule !== undefined);

	// Initialise form state from rule prop (edit) or defaults (create).
	// Use $derived.by so the form resets whenever the rule prop changes.
	let patternType = $state<PatternType>('prefix');
	let patternValue = $state('');
	let backendType = $state<BackendType>('local');
	let backendModel = $state('');
	let selectedProfileId = $state(''); // string form of profile id, '' means none
	let fallbackBackend = $state<BackendType | ''>('');
	let saving = $state(false);
	let error = $state<string | null>(null);

	// Re-populate form fields when the `rule` prop changes (switching between rules or
	// toggling between create and edit mode).
	$effect(() => {
		if (rule) {
			patternType = rule.pattern_type;
			patternValue = rule.model_pattern;
			backendType = rule.backend_type;
			backendModel = rule.backend_model ?? '';
			selectedProfileId = rule.profile_id != null ? String(rule.profile_id) : '';
			fallbackBackend = rule.fallback_backend ?? '';
		} else {
			patternType = 'prefix';
			patternValue = '';
			backendType = 'local';
			backendModel = '';
			selectedProfileId = '';
			fallbackBackend = '';
		}
		error = null;
	});

	// Use $derived.by() to maintain reactivity for configuredProviders prop
	// This ensures the warning updates when the parent's configuredProviders changes
	const showWarning = $derived.by(() => {
		return backendType !== 'local' && !configuredProviders.includes(backendType);
	});

	const isLocal = $derived(backendType === 'local');

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
			if (isEditMode && rule) {
				await settings.updateRule(rule.id, {
					pattern_type: patternType,
					model_pattern: patternValue.trim(),
					backend_type: backendType,
					...(isLocal
						? { profile_id: selectedProfileId !== '' ? Number(selectedProfileId) : null }
						: { backend_model: backendModel.trim() || null }),
					fallback_backend: fallbackBackend || null
				});
			} else {
				await settings.createRule({
					pattern_type: patternType,
					model_pattern: patternValue.trim(),
					backend_type: backendType,
					// For local rules: send profile_id; for cloud rules: send backend_model
					...(isLocal
						? { profile_id: selectedProfileId !== '' ? Number(selectedProfileId) : null }
						: { backend_model: backendModel.trim() || undefined }),
					fallback_backend: fallbackBackend || undefined
				});
				// Reset form only in create mode
				patternValue = '';
				backendModel = '';
				selectedProfileId = '';
				fallbackBackend = '';
			}
			onSave();
		} catch (e) {
			error = e instanceof Error ? e.message : isEditMode ? 'Failed to update rule' : 'Failed to create rule';
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
	<h3 class="text-sm font-medium">{isEditMode ? 'Edit Routing Rule' : 'Add Routing Rule'}</h3>

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

		<!-- Profile selector (local) or Backend Model override (cloud) -->
		{#if isLocal}
			<div class="space-y-2">
				<label for="profile-select" class="text-sm font-medium">
					Profile <span class="text-muted-foreground">(optional)</span>
				</label>
				<Select id="profile-select" bind:value={selectedProfileId}>
					<option value="">Any available profile</option>
					{#each profileStore.profiles as profile (profile.id)}
						<option value={String(profile.id)}>{profile.name}</option>
					{/each}
				</Select>
				{#if profileStore.profiles.length === 0}
					<p class="text-xs text-muted-foreground">No profiles configured yet.</p>
				{/if}
			</div>
		{:else}
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
		{/if}

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

	<div class="flex justify-end gap-2">
		{#if isEditMode && onCancel}
			<Button type="button" variant="outline" onclick={onCancel} disabled={saving}>
				Cancel
			</Button>
		{/if}
		<Button type="submit" disabled={saving}>
			{#if saving}
				<Loader2 class="mr-2 h-4 w-4 animate-spin" />
			{:else if isEditMode}
				<Pencil class="mr-2 h-4 w-4" />
			{:else}
				<Plus class="mr-2 h-4 w-4" />
			{/if}
			{isEditMode ? 'Save Changes' : 'Add Rule'}
		</Button>
	</div>
</form>
