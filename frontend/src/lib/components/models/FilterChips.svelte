<script lang="ts">
	import type { FilterState } from './filter-types';
	import { Badge } from '$components/ui';
	import { X } from 'lucide-svelte';

	interface Props {
		filters: FilterState;
		onRemove: (type: 'architecture' | 'multimodal' | 'quantization', value?: string | number) => void;
	}

	let { filters, onRemove }: Props = $props();
</script>

<div class="flex flex-wrap gap-2">
	{#each filters.architectures as arch (arch)}
		<Badge variant="secondary" class="cursor-pointer gap-1 pr-1">
			{arch}
			<button
				type="button"
				onclick={() => onRemove('architecture', arch)}
				class="ml-1 rounded-full hover:bg-muted-foreground/20 p-0.5"
			>
				<X class="w-3 h-3" />
			</button>
		</Badge>
	{/each}

	{#if filters.multimodal === true}
		<Badge variant="secondary" class="cursor-pointer gap-1 pr-1">
			Multimodal
			<button
				type="button"
				onclick={() => onRemove('multimodal')}
				class="ml-1 rounded-full hover:bg-muted-foreground/20 p-0.5"
			>
				<X class="w-3 h-3" />
			</button>
		</Badge>
	{:else if filters.multimodal === false}
		<Badge variant="secondary" class="cursor-pointer gap-1 pr-1">
			Text-only
			<button
				type="button"
				onclick={() => onRemove('multimodal')}
				class="ml-1 rounded-full hover:bg-muted-foreground/20 p-0.5"
			>
				<X class="w-3 h-3" />
			</button>
		</Badge>
	{/if}

	{#each filters.quantization as bits (bits)}
		<Badge variant="secondary" class="cursor-pointer gap-1 pr-1">
			{bits === 16 ? 'fp16' : `${bits}-bit`}
			<button
				type="button"
				onclick={() => onRemove('quantization', bits)}
				class="ml-1 rounded-full hover:bg-muted-foreground/20 p-0.5"
			>
				<X class="w-3 h-3" />
			</button>
		</Badge>
	{/each}
</div>
