<script lang="ts">
	import { Brain, CheckCircle2 } from 'lucide-svelte';

	interface Props {
		verified?: boolean;
	}

	let { verified = false }: Props = $props();
</script>

<div
	class="inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium bg-teal-100 text-teal-800 border-teal-300 dark:bg-teal-900/30 dark:text-teal-300 dark:border-teal-700"
	title={verified ? 'Verified by model probe' : 'Detected from model configuration'}
>
	<Brain class="w-3 h-3" />
	Thinking
	{#if verified}
		<CheckCircle2 class="w-3 h-3" />
	{/if}
</div>
