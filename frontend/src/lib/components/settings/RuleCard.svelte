<script lang="ts">
	import { GripVertical, Trash2, AlertTriangle } from 'lucide-svelte';
	import type { BackendMapping } from '$lib/api/types';

	interface Props {
		rule: BackendMapping;
		hasWarning?: boolean;
		onDelete: () => void;
	}

	let { rule, hasWarning = false, onDelete }: Props = $props();

	const patternTypeColors: Record<string, string> = {
		exact: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
		prefix: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
		regex: 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200'
	};

	const backendTypeLabels: Record<string, string> = {
		local: 'Local',
		openai: 'OpenAI',
		anthropic: 'Anthropic'
	};
</script>

<div
	class="flex items-center gap-3 rounded-lg border bg-card p-4 shadow-sm transition-colors hover:bg-accent/50 {!rule.enabled
		? 'opacity-50'
		: ''}"
	data-rule-id={rule.id}
>
	<!-- Drag handle -->
	<div
		class="sortable-handle cursor-grab text-muted-foreground hover:text-foreground active:cursor-grabbing"
	>
		<GripVertical class="h-5 w-5" />
	</div>

	<!-- Pattern info -->
	<div class="min-w-0 flex-1">
		<div class="flex flex-wrap items-center gap-2">
			<span
				class="inline-flex items-center rounded px-2 py-0.5 text-xs font-medium {patternTypeColors[
					rule.pattern_type
				]}"
			>
				{rule.pattern_type}
			</span>
			<code class="max-w-xs truncate font-mono text-sm">{rule.model_pattern}</code>
			{#if hasWarning}
				<span
					class="inline-flex items-center gap-1 rounded bg-yellow-100 px-2 py-0.5 text-xs font-medium text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200"
				>
					<AlertTriangle class="h-3 w-3" />
					Unconfigured
				</span>
			{/if}
		</div>
		<p class="mt-1 text-sm text-muted-foreground">
			Routes to <span class="font-medium capitalize"
				>{backendTypeLabels[rule.backend_type] || rule.backend_type}</span
			>
			{#if rule.backend_model}
				<span class="text-xs">as {rule.backend_model}</span>
			{/if}
			{#if rule.fallback_backend}
				<span class="text-xs"
					>(fallback: {backendTypeLabels[rule.fallback_backend] || rule.fallback_backend})</span
				>
			{/if}
		</p>
	</div>

	<!-- Actions -->
	<button
		onclick={onDelete}
		class="p-2 text-muted-foreground transition-colors hover:text-destructive"
		title="Delete rule"
	>
		<Trash2 class="h-4 w-4" />
	</button>
</div>
