<script lang="ts">
	import { Dialog } from 'bits-ui';
	import {
		X,
		Loader2,
		CheckCircle2,
		XCircle,
		MinusCircle,
		FlaskConical,
		Volume2,
		Pause
	} from 'lucide-svelte';
	import type { ProbeState } from '$stores/probe.svelte';
	import { Button } from '$components/ui';
	import ToolUseBadge from './badges/ToolUseBadge.svelte';
	import ThinkingBadge from './badges/ThinkingBadge.svelte';
	import TTSBadge from './badges/TTSBadge.svelte';
	import STTBadge from './badges/STTBadge.svelte';
	import EmbeddingsBadge from './badges/EmbeddingsBadge.svelte';

	interface Props {
		open: boolean;
		modelId: string;
		probe: ProbeState;
	}

	let { open = $bindable(), modelId, probe }: Props = $props();

	const stepLabels: Record<string, string> = {
		detect_type: 'Detecting model type',
		load_model: 'Loading model into memory',
		check_context: 'Testing context window',
		test_thinking: 'Testing thinking/reasoning',
		test_tools: 'Testing native tool calling',
		detect_model_type: 'Detecting model type',
		detect_vision: 'Detecting vision capabilities',
		test_multi_image: 'Testing multi-image support',
		test_video: 'Testing video support',
		detect_audio_type: 'Detecting audio capabilities',
		test_tts: 'Testing text-to-speech',
		test_stt: 'Testing speech-to-text',
		test_embeddings: 'Testing embedding generation',
		test_normalization: 'Testing embedding normalization',
		save_results: 'Saving probe results',
		cleanup: 'Cleaning up'
	};

	// Audio playback from base64 TTS result
	let audioUrl = $state<string | null>(null);
	let isPlaying = $state(false);
	let audioElement: HTMLAudioElement | null = null;

	$effect(() => {
		const ttsStep = probe.steps.find(
			(s) => s.step === 'test_tts' && s.status === 'completed' && typeof s.value === 'string'
		);

		if (ttsStep && typeof ttsStep.value === 'string') {
			const binary = atob(ttsStep.value);
			const bytes = new Uint8Array(binary.length);
			for (let i = 0; i < binary.length; i++) {
				bytes[i] = binary.charCodeAt(i);
			}
			const blob = new Blob([bytes], { type: 'audio/wav' });
			const url = URL.createObjectURL(blob);
			audioUrl = url;

			// Create audio element for playback
			const audio = new Audio(url);
			audio.addEventListener('ended', () => {
				isPlaying = false;
			});
			audioElement = audio;

			return () => {
				audio.pause();
				audioElement = null;
				isPlaying = false;
				URL.revokeObjectURL(url);
			};
		} else {
			audioUrl = null;
			audioElement = null;
			isPlaying = false;
		}
	});

	function togglePlayback() {
		if (!audioElement) return;
		if (isPlaying) {
			audioElement.pause();
			isPlaying = false;
		} else {
			audioElement.play();
			isPlaying = true;
		}
	}

	// Format capability values for display
	function formatCapValue(value: unknown): string {
		if (typeof value === 'boolean') return value ? 'Yes' : 'No';
		if (typeof value === 'number') return value.toLocaleString();
		if (typeof value === 'string') return value;
		return String(value);
	}

	const capLabels: Record<string, string> = {
		supports_native_tools: 'Native Tools',
		supports_thinking: 'Thinking',
		tool_format: 'Tool Format',
		practical_max_tokens: 'Max Tokens',
		supports_multi_image: 'Multi-Image',
		supports_video: 'Video',
		embedding_dimensions: 'Dimensions',
		max_sequence_length: 'Max Sequence',
		is_normalized: 'Normalized',
		supports_tts: 'TTS',
		supports_stt: 'STT'
	};

	let capEntries = $derived(
		Object.entries(probe.capabilities).filter(
			([key, val]) => key !== 'model_id' && val !== undefined && val !== null && key in capLabels
		)
	);

	// Derive which badges to show from capabilities
	let showToolUse = $derived(probe.capabilities.supports_native_tools === true);
	let showThinking = $derived(probe.capabilities.supports_thinking === true);
	let showTTS = $derived(probe.capabilities.supports_tts === true);
	let showSTT = $derived(probe.capabilities.supports_stt === true);
	let showEmbeddings = $derived(
		probe.capabilities.embedding_dimensions !== undefined &&
			probe.capabilities.embedding_dimensions !== null
	);
	let hasBadges = $derived(showToolUse || showThinking || showTTS || showSTT || showEmbeddings);
</script>

<Dialog.Root bind:open>
	<Dialog.Portal>
		<Dialog.Overlay
			class="fixed inset-0 z-50 bg-black/50 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0"
		/>
		<Dialog.Content
			class="fixed left-[50%] top-[50%] z-50 w-full max-w-[600px] translate-x-[-50%] translate-y-[-50%] border bg-background p-6 shadow-lg duration-200 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 sm:rounded-lg max-h-[80vh] overflow-y-auto"
		>
			<!-- Header -->
			<div class="flex items-center justify-between mb-4">
				<div class="flex items-center gap-2">
					<FlaskConical class="w-5 h-5 text-primary" />
					<Dialog.Title class="text-lg font-semibold">Model Probe</Dialog.Title>
				</div>
				<Dialog.Close
					class="rounded-sm opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
				>
					<X class="h-4 w-4" />
					<span class="sr-only">Close</span>
				</Dialog.Close>
			</div>

			<p class="text-sm text-muted-foreground mb-4 font-mono truncate">{modelId}</p>

			<!-- Steps -->
			<div class="space-y-2 mb-4">
				{#each probe.steps as step (step.step)}
					<div class="flex items-start gap-2">
						<div class="mt-0.5">
							{#if step.status === 'running'}
								<Loader2 class="w-4 h-4 animate-spin text-blue-500" />
							{:else if step.status === 'completed'}
								<CheckCircle2 class="w-4 h-4 text-green-500" />
							{:else if step.status === 'failed'}
								<XCircle class="w-4 h-4 text-red-500" />
							{:else}
								<MinusCircle class="w-4 h-4 text-muted-foreground" />
							{/if}
						</div>
						<div class="flex-1 min-w-0">
							<span
								class="text-sm {step.status === 'failed'
									? 'text-red-500'
									: step.status === 'skipped'
										? 'text-muted-foreground'
										: ''}"
							>
								{stepLabels[step.step] ?? step.step}
							</span>
							{#if step.status === 'failed' && step.error}
								<p class="text-xs text-red-400 mt-0.5 truncate">{step.error}</p>
							{/if}
						</div>
					</div>

					<!-- Audio player after TTS step -->
					{#if step.step === 'test_tts' && audioUrl}
						<div class="ml-6 mt-1 mb-2">
							<button
								onclick={togglePlayback}
								class="inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium transition-colors bg-rose-100 text-rose-800 border border-rose-300 hover:bg-rose-200 dark:bg-rose-900/30 dark:text-rose-300 dark:border-rose-700 dark:hover:bg-rose-900/50"
							>
								{#if isPlaying}
									<Pause class="w-3 h-3" />
									Pause
								{:else}
									<Volume2 class="w-3 h-3" />
									Play Test
								{/if}
							</button>
						</div>
					{/if}
				{/each}
			</div>

			<!-- Overall error -->
			{#if probe.error}
				<div class="rounded-md bg-red-50 dark:bg-red-900/20 p-3 mb-4">
					<p class="text-sm text-red-600 dark:text-red-400">{probe.error}</p>
				</div>
			{/if}

			<!-- Capabilities summary -->
			{#if probe.status === 'completed' && capEntries.length > 0}
				<div class="border-l-2 border-green-500 pl-4 mt-4">
					<h4 class="text-sm font-medium mb-2">Discovered Capabilities</h4>

					<!-- Badges -->
					{#if hasBadges}
						<div class="flex flex-wrap gap-1.5 mb-3">
							{#if showToolUse}
								<ToolUseBadge verified={true} />
							{/if}
							{#if showThinking}
								<ThinkingBadge />
							{/if}
							{#if showTTS}
								<TTSBadge verified={true} />
							{/if}
							{#if showSTT}
								<STTBadge verified={true} />
							{/if}
							{#if showEmbeddings}
								<EmbeddingsBadge verified={true} />
							{/if}
						</div>
					{/if}

					<div class="grid grid-cols-2 gap-x-4 gap-y-1">
						{#each capEntries as [key, val] (key)}
							<div class="flex justify-between text-sm">
								<span class="text-muted-foreground">{capLabels[key] ?? key}</span>
								<span class="font-medium">{formatCapValue(val)}</span>
							</div>
						{/each}
					</div>
				</div>
			{/if}

			<!-- Close button -->
			<div class="flex justify-end mt-6">
				<Button variant="outline" onclick={() => (open = false)}>Close</Button>
			</div>
		</Dialog.Content>
	</Dialog.Portal>
</Dialog.Root>
