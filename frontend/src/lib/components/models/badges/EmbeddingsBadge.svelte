<script lang="ts">
	import { Box, CheckCircle2 } from 'lucide-svelte';

	interface Props {
		verified?: boolean;
	}

	let { verified = true }: Props = $props();
</script>

<div
	class="inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium bg-cyan-100 text-cyan-800 border-cyan-300 dark:bg-cyan-900/30 dark:text-cyan-300 dark:border-cyan-700"
	title="Embeddings model verified by probe"
>
	<Box class="w-3 h-3" />
	Embeddings
	{#if verified}
		<CheckCircle2 class="w-3 h-3" />
	{/if}
</div>
