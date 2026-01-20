<script lang="ts">
	import type { ModelCharacteristics } from '$api';
	import { ChevronDown, ChevronUp } from 'lucide-svelte';
	import { slide } from 'svelte/transition';

	interface Props {
		characteristics: ModelCharacteristics;
	}

	let { characteristics }: Props = $props();
	let expanded = $state(false);

	function toggle() {
		expanded = !expanded;
	}

	// Format numbers for display
	function formatNumber(value: number | undefined): string {
		return value?.toLocaleString() ?? '-';
	}

	// Check if we have any specs to show
	let hasSpecs = $derived(
		characteristics.max_position_embeddings ||
			characteristics.num_hidden_layers ||
			characteristics.hidden_size ||
			characteristics.vocab_size ||
			characteristics.num_attention_heads
	);
</script>

{#if hasSpecs}
	<div class="mt-2">
		<button
			type="button"
			onclick={toggle}
			class="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
		>
			{#if expanded}
				<ChevronUp class="w-3 h-3" />
				Hide specs
			{:else}
				<ChevronDown class="w-3 h-3" />
				Show specs
			{/if}
		</button>

		{#if expanded}
			<div
				transition:slide={{ duration: 200 }}
				class="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-muted-foreground"
			>
				{#if characteristics.max_position_embeddings}
					<span class="font-medium">Context</span>
					<span>{formatNumber(characteristics.max_position_embeddings)}</span>
				{/if}
				{#if characteristics.num_hidden_layers}
					<span class="font-medium">Layers</span>
					<span>{formatNumber(characteristics.num_hidden_layers)}</span>
				{/if}
				{#if characteristics.hidden_size}
					<span class="font-medium">Hidden size</span>
					<span>{formatNumber(characteristics.hidden_size)}</span>
				{/if}
				{#if characteristics.vocab_size}
					<span class="font-medium">Vocab size</span>
					<span>{formatNumber(characteristics.vocab_size)}</span>
				{/if}
				{#if characteristics.num_attention_heads}
					<span class="font-medium">Attention</span>
					<span>
						{formatNumber(characteristics.num_attention_heads)}
						{#if characteristics.num_key_value_heads && characteristics.num_key_value_heads !== characteristics.num_attention_heads}
							<span class="text-muted-foreground/70">
								(KV: {formatNumber(characteristics.num_key_value_heads)})
							</span>
						{/if}
					</span>
				{/if}
				{#if characteristics.use_cache !== undefined}
					<span class="font-medium">KV Cache</span>
					<span>{characteristics.use_cache ? 'Yes' : 'No'}</span>
				{/if}
			</div>
		{/if}
	</div>
{/if}
