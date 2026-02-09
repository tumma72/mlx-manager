<script lang="ts">
	import { Volume2, CheckCircle2 } from 'lucide-svelte';

	interface Props {
		verified?: boolean;
	}

	let { verified = true }: Props = $props();
</script>

<div
	class="inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium bg-rose-100 text-rose-800 border-rose-300 dark:bg-rose-900/30 dark:text-rose-300 dark:border-rose-700"
	title="Text-to-speech verified by model probe"
>
	<Volume2 class="w-3 h-3" />
	TTS
	{#if verified}
		<CheckCircle2 class="w-3 h-3" />
	{/if}
</div>
