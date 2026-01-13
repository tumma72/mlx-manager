<script lang="ts">
	import { ChevronDown, ChevronRight, Brain } from 'lucide-svelte';

	interface Props {
		content: string;
		defaultExpanded?: boolean;
	}

	let { content, defaultExpanded = false }: Props = $props();

	let expanded = $state(defaultExpanded);
</script>

<div class="my-2 border border-muted rounded-lg overflow-hidden">
	<button
		type="button"
		class="w-full flex items-center gap-2 px-3 py-2 bg-muted/50 hover:bg-muted/70 transition-colors text-sm text-muted-foreground"
		onclick={() => (expanded = !expanded)}
	>
		{#if expanded}
			<ChevronDown class="w-4 h-4" />
		{:else}
			<ChevronRight class="w-4 h-4" />
		{/if}
		<Brain class="w-4 h-4" />
		<span>Thinking</span>
	</button>
	{#if expanded}
		<div class="px-3 py-2 text-sm text-muted-foreground bg-muted/20 whitespace-pre-wrap">
			{content}
		</div>
	{/if}
</div>
