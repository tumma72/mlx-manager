<script lang="ts">
	import { Cpu } from 'lucide-svelte';

	interface Props {
		family: string;
	}

	let { family }: Props = $props();
</script>

<div
	class="inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium bg-blue-100 text-blue-800 border-blue-300 dark:bg-blue-900/30 dark:text-blue-300 dark:border-blue-700"
>
	<Cpu class="w-3 h-3" />
	{family}
</div>
