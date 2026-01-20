<script lang="ts">
	import { Eye } from 'lucide-svelte';

	interface Props {
		multimodalType?: string;
	}

	let { multimodalType }: Props = $props();

	// Capitalize first letter of type, default to "Vision"
	let label = $derived(
		multimodalType ? multimodalType.charAt(0).toUpperCase() + multimodalType.slice(1) : 'Vision'
	);
</script>

<div
	class="inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium bg-green-100 text-green-800 border-green-300 dark:bg-green-900/30 dark:text-green-300 dark:border-green-700"
>
	<Eye class="w-3 h-3" />
	{label}
</div>
