<script lang="ts">
	import { Mic, CheckCircle2 } from 'lucide-svelte';

	interface Props {
		verified?: boolean;
	}

	let { verified = true }: Props = $props();
</script>

<div
	class="inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium bg-indigo-100 text-indigo-800 border-indigo-300 dark:bg-indigo-900/30 dark:text-indigo-300 dark:border-indigo-700"
	title="Speech-to-text verified by model probe"
>
	<Mic class="w-3 h-3" />
	STT
	{#if verified}
		<CheckCircle2 class="w-3 h-3" />
	{/if}
</div>
