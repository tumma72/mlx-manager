<script lang="ts">
	import type { ModelCharacteristics, ModelCapabilities } from '$api';
	import ArchitectureBadge from './badges/ArchitectureBadge.svelte';
	import MultimodalBadge from './badges/MultimodalBadge.svelte';
	import QuantizationBadge from './badges/QuantizationBadge.svelte';
	import ToolUseBadge from './badges/ToolUseBadge.svelte';
	import ThinkingBadge from './badges/ThinkingBadge.svelte';
	import TTSBadge from './badges/TTSBadge.svelte';
	import STTBadge from './badges/STTBadge.svelte';
	import EmbeddingsBadge from './badges/EmbeddingsBadge.svelte';

	interface Props {
		characteristics: ModelCharacteristics | null | undefined;
		capabilities?: ModelCapabilities | null;
		loading?: boolean;
	}

	let { characteristics, capabilities = null, loading = false }: Props = $props();

	// Check if we should show architecture badge
	let showArchitecture = $derived(
		characteristics?.architecture_family && characteristics.architecture_family !== 'Unknown'
	);

	// Tool Use badge logic: probed result takes precedence over heuristic
	let showToolUse = $derived(
		capabilities
			? capabilities.supports_native_tools === true
			: characteristics?.is_tool_use === true
	);
	let toolUseVerified = $derived(capabilities?.supports_native_tools === true);
</script>

<div class="flex flex-wrap items-center gap-1.5">
	{#if loading}
		<!-- Skeleton badges while loading -->
		<div class="h-5 w-14 animate-pulse rounded-full bg-muted"></div>
		<div class="h-5 w-12 animate-pulse rounded-full bg-muted"></div>
	{:else if characteristics}
		{#if showArchitecture}
			<ArchitectureBadge family={characteristics.architecture_family!} />
		{/if}
		{#if characteristics.is_multimodal}
			<MultimodalBadge multimodalType={characteristics.multimodal_type} />
		{/if}
		{#if characteristics.quantization_bits}
			<QuantizationBadge bits={characteristics.quantization_bits} />
		{/if}
		{#if capabilities?.supports_thinking === true}
			<ThinkingBadge />
		{/if}
		{#if showToolUse}
			<ToolUseBadge verified={toolUseVerified} />
		{/if}
		{#if capabilities?.supports_tts === true}
			<TTSBadge />
		{/if}
		{#if capabilities?.supports_stt === true}
			<STTBadge />
		{/if}
		{#if capabilities?.model_type === 'embeddings'}
			<EmbeddingsBadge />
		{/if}
	{/if}
</div>
