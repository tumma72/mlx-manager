<script lang="ts">
	import { Layers } from 'lucide-svelte';

	interface Props {
		bits: number;
	}

	let { bits }: Props = $props();
</script>

<div
	class="inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium bg-purple-100 text-purple-800 border-purple-300 dark:bg-purple-900/30 dark:text-purple-300 dark:border-purple-700"
>
	<Layers class="w-3 h-3" />
	{bits}-bit
</div>
