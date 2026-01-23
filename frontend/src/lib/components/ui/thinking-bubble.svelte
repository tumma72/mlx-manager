<script lang="ts">
	import { Collapsible } from 'bits-ui';
	import { ChevronDown, ChevronRight, Brain, Loader2 } from 'lucide-svelte';

	interface Props {
		content: string;
		duration?: number; // seconds
		streaming?: boolean;
		defaultExpanded?: boolean;
	}

	let { content, duration, streaming = false, defaultExpanded = false }: Props = $props();

	let expanded = $state(getInitialExpanded());
	function getInitialExpanded() {
		return defaultExpanded || streaming; // Auto-expand while streaming
	}

	// Auto-collapse when streaming finishes
	$effect(() => {
		if (!streaming && duration !== undefined) {
			expanded = false;
		}
	});

	const label = $derived.by(() => {
		if (streaming) return 'Thinking...';
		if (duration !== undefined) return `Thought for ${duration.toFixed(1)}s`;
		return 'Thinking';
	});
</script>

<div class="my-2">
	<Collapsible.Root bind:open={expanded}>
		<Collapsible.Trigger
			class="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
		>
			{#if expanded}
				<ChevronDown class="w-4 h-4" />
			{:else}
				<ChevronRight class="w-4 h-4" />
			{/if}
			{#if streaming}
				<Loader2 class="w-4 h-4 animate-spin" />
			{:else}
				<Brain class="w-4 h-4" />
			{/if}
			<span>{label}</span>
		</Collapsible.Trigger>
		<Collapsible.Content>
			<div
				class="mt-2 pl-6 border-l-2 border-muted text-sm text-muted-foreground italic whitespace-pre-wrap"
			>
				{content}
			</div>
		</Collapsible.Content>
	</Collapsible.Root>
</div>
