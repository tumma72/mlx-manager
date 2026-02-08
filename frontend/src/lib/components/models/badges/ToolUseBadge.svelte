<script lang="ts">
	import { Wrench, CheckCircle2 } from 'lucide-svelte';

	interface Props {
		verified?: boolean;
	}

	let { verified = false }: Props = $props();
</script>

<div
	class="inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium bg-amber-100 text-amber-800 border-amber-300 dark:bg-amber-900/30 dark:text-amber-300 dark:border-amber-700"
	title={verified ? 'Verified by model probe' : 'Detected from model name/tags'}
>
	<Wrench class="w-3 h-3" />
	Tool Use
	{#if verified}
		<CheckCircle2 class="w-3 h-3" />
	{/if}
</div>
