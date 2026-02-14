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
		Pause,
		Copy,
		Check
	} from 'lucide-svelte';
	import type { ProbeState } from '$stores/probe.svelte';
	import type { ProbeDiagnostic } from '$lib/api/types';
	import { systemStore } from '$stores';
	import { Button } from '$components/ui';
	import ToolUseBadge from './badges/ToolUseBadge.svelte';
	import ThinkingBadge from './badges/ThinkingBadge.svelte';
	import TTSBadge from './badges/TTSBadge.svelte';
	import STTBadge from './badges/STTBadge.svelte';
	import EmbeddingsBadge from './badges/EmbeddingsBadge.svelte';
	import VisionImageBadge from './badges/VisionImageBadge.svelte';
	import VisionVideoBadge from './badges/VisionVideoBadge.svelte';

	interface Props {
		open: boolean;
		modelId: string;
		probe: ProbeState;
	}

	let { open = $bindable(), modelId, probe }: Props = $props();

	const stepLabels: Record<string, string> = {
		detect_type: 'Detecting model type',
		detect_family: 'Detecting model family',
		find_strategy: 'Finding probe strategy',
		load_model: 'Loading model into memory',
		check_context: 'Testing context window',
		check_processor: 'Checking image processor',
		check_multi_image: 'Checking multi-image support',
		check_video: 'Checking video support',
		test_thinking: 'Testing thinking/reasoning',
		test_tools: 'Testing native tool calling',
		detect_audio_type: 'Detecting audio capabilities',
		test_tts: 'Testing text-to-speech',
		test_stt: 'Testing speech-to-text',
		test_encode: 'Testing embedding generation',
		check_normalization: 'Checking embedding normalization',
		check_max_length: 'Checking max sequence length',
		test_similarity: 'Testing similarity ordering',
		save_results: 'Saving probe results',
		cleanup: 'Cleaning up',
		strategy_error: 'Strategy error',
		probe_complete: 'Probe complete'
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

	// STT results from probe (value is {transcript, audio_b64} dict)
	let sttResult = $derived.by(() => {
		const sttStep = probe.steps.find((s) => s.step === 'test_stt' && s.status === 'completed');
		if (!sttStep?.value) return null;
		const val = sttStep.value as Record<string, unknown>;
		if (typeof val === 'object' && val !== null && 'transcript' in val) {
			return {
				transcript: val.transcript as string,
				audioB64: (val.audio_b64 as string) ?? null
			};
		}
		// Backwards compat: plain string transcript
		if (typeof sttStep.value === 'string') {
			return { transcript: sttStep.value, audioB64: null };
		}
		return null;
	});

	// STT audio playback (separate from TTS audio)
	let sttAudioUrl = $state<string | null>(null);
	let sttIsPlaying = $state(false);
	let sttAudioElement: HTMLAudioElement | null = null;

	$effect(() => {
		const b64 = sttResult?.audioB64;
		if (b64) {
			const binary = atob(b64);
			const bytes = new Uint8Array(binary.length);
			for (let i = 0; i < binary.length; i++) {
				bytes[i] = binary.charCodeAt(i);
			}
			const blob = new Blob([bytes], { type: 'audio/wav' });
			const url = URL.createObjectURL(blob);
			sttAudioUrl = url;

			const audio = new Audio(url);
			audio.addEventListener('ended', () => {
				sttIsPlaying = false;
			});
			sttAudioElement = audio;

			return () => {
				audio.pause();
				sttAudioElement = null;
				sttIsPlaying = false;
				URL.revokeObjectURL(url);
			};
		} else {
			sttAudioUrl = null;
			sttAudioElement = null;
			sttIsPlaying = false;
		}
	});

	function toggleSttPlayback() {
		if (!sttAudioElement) return;
		if (sttIsPlaying) {
			sttAudioElement.pause();
			sttIsPlaying = false;
		} else {
			sttAudioElement.play();
			sttIsPlaying = true;
		}
	}

	// Derive which badges to show from capabilities
	// tool_format is the SSE-emitted capability (supports_native_tools is only in backend ProbeResult)
	let showToolUse = $derived(
		probe.capabilities.supports_native_tools === true ||
			(typeof probe.capabilities.tool_format === 'string' &&
				probe.capabilities.tool_format.length > 0)
	);
	let showThinking = $derived(probe.capabilities.supports_thinking === true);
	let showTTS = $derived(probe.capabilities.supports_tts === true);
	let showSTT = $derived(probe.capabilities.supports_stt === true);
	let showEmbeddings = $derived(
		probe.capabilities.embedding_dimensions !== undefined &&
			probe.capabilities.embedding_dimensions !== null
	);
	let showMultiImage = $derived(probe.capabilities.supports_multi_image === true);
	let showVideo = $derived(probe.capabilities.supports_video === true);
	let hasBadges = $derived(showToolUse || showThinking || showTTS || showSTT || showEmbeddings || showMultiImage || showVideo);

	// Filter out probe_complete from displayed steps
	let displayedSteps = $derived(probe.steps.filter((s) => s.step !== 'probe_complete'));

	// Sorted diagnostics: action_needed first, then warning, then info
	const levelOrder: Record<string, number> = { action_needed: 0, warning: 1, info: 2 };
	let sortedDiagnostics = $derived(
		[...probe.diagnostics].sort(
			(a, b) => (levelOrder[a.level] ?? 3) - (levelOrder[b.level] ?? 3)
		)
	);

	// Diagnostic level badge styling
	function diagBadgeClass(level: ProbeDiagnostic['level']): string {
		switch (level) {
			case 'action_needed':
				return 'bg-red-500/20 text-red-600 dark:text-red-400';
			case 'warning':
				return 'bg-yellow-500/20 text-yellow-600 dark:text-yellow-400';
			case 'info':
				return 'bg-blue-500/20 text-blue-600 dark:text-blue-400';
			default:
				return 'bg-muted text-muted-foreground';
		}
	}

	function diagLevelLabel(level: ProbeDiagnostic['level']): string {
		switch (level) {
			case 'action_needed':
				return 'Action Needed';
			case 'warning':
				return 'Warning';
			case 'info':
				return 'Info';
			default:
				return level;
		}
	}

	const categoryLabels: Record<string, string> = {
		family: 'Model Family',
		tool_dialect: 'Tool Dialect',
		thinking_dialect: 'Thinking Dialect',
		type: 'Type Detection',
		unsupported: 'Unsupported'
	};

	// Copy report functionality
	let copied = $state(false);

	function generateReport(): string {
		const caps = probe.capabilities;
		const sys = systemStore.info;
		let report = `# Probe Diagnostic Report: \`${modelId}\`\n\n`;
		report += `## Model Information\n`;
		report += `- **Model ID:** \`${modelId}\`\n`;
		if (caps.model_type) report += `- **Detected Type:** ${caps.model_type}\n`;
		if (caps.model_family) report += `- **Model Family:** ${caps.model_family}\n`;
		report += `\n## Diagnostics\n`;
		for (const diag of probe.diagnostics) {
			report += `- **[${diag.category}]** ${diag.message}\n`;
		}
		report += `\n## Environment\n`;
		if (sys?.os_version) report += `- **OS:** ${sys.os_version}\n`;
		if (sys?.python_version) report += `- **Python:** ${sys.python_version}\n`;
		if (sys?.mlx_version) report += `- **MLX:** ${sys.mlx_version}\n`;
		return report;
	}

	async function copyReport() {
		try {
			await navigator.clipboard.writeText(generateReport());
			copied = true;
			setTimeout(() => {
				copied = false;
			}, 2000);
		} catch {
			// Fallback for older browsers
			const textarea = document.createElement('textarea');
			textarea.value = generateReport();
			document.body.appendChild(textarea);
			textarea.select();
			document.execCommand('copy');
			document.body.removeChild(textarea);
			copied = true;
			setTimeout(() => {
				copied = false;
			}, 2000);
		}
	}
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
				{#each displayedSteps as step (step.step)}
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
							{#if step.diagnostics?.length}
								<span class="ml-1 text-[10px] px-1.5 py-0.5 rounded-full bg-yellow-500/20 text-yellow-600 dark:text-yellow-400">
									{step.diagnostics.length} {step.diagnostics.some(d => d.level === 'action_needed') ? 'action needed' : step.diagnostics.length === 1 ? 'warning' : 'warnings'}
								</span>
							{/if}
							{#if step.status === 'failed' && step.error}
								<p class="text-xs text-red-400 mt-0.5 truncate">{step.error}</p>
							{/if}
						</div>
					</div>

					<!-- Play button after TTS step (only when TTS completed) -->
					{#if step.step === 'test_tts' && step.status === 'completed' && audioUrl}
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

					<!-- STT results after STT step (play button + transcript) -->
					{#if step.step === 'test_stt' && step.status === 'completed' && sttResult}
						<div class="ml-6 mt-1 mb-2 space-y-1.5">
							{#if sttAudioUrl}
								<button
									onclick={toggleSttPlayback}
									class="inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium transition-colors bg-indigo-100 text-indigo-800 border border-indigo-300 hover:bg-indigo-200 dark:bg-indigo-900/30 dark:text-indigo-300 dark:border-indigo-700 dark:hover:bg-indigo-900/50"
								>
									{#if sttIsPlaying}
										<Pause class="w-3 h-3" />
										Pause
									{:else}
										<Volume2 class="w-3 h-3" />
										Play
									{/if}
								</button>
							{/if}
							{#if sttResult.transcript}
								<p
									class="text-xs text-muted-foreground italic rounded bg-muted/50 px-2 py-1.5 whitespace-pre-wrap break-words"
								>
									"{sttResult.transcript}"
								</p>
							{/if}
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

			<!-- Diagnostics -->
			{#if sortedDiagnostics.length > 0}
				<div class="border-l-2 border-yellow-500 pl-4 mt-4 mb-4">
					<h4 class="text-sm font-medium mb-2">Diagnostics</h4>
					<div class="space-y-2">
						{#each sortedDiagnostics as diag, i (i)}
							<div class="flex items-start gap-2">
								<span class="inline-flex items-center text-[10px] font-medium px-1.5 py-0.5 rounded-full shrink-0 mt-0.5 {diagBadgeClass(diag.level)}">
									{diagLevelLabel(diag.level)}
								</span>
								<div class="min-w-0">
									<p class="text-sm">{diag.message}</p>
									<span class="text-[10px] text-muted-foreground">{categoryLabels[diag.category] ?? diag.category}</span>
								</div>
							</div>
						{/each}
					</div>
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
							{#if showMultiImage}
								<VisionImageBadge />
							{/if}
							{#if showVideo}
								<VisionVideoBadge />
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

			<!-- Footer buttons -->
			<div class="flex justify-end gap-2 mt-6">
				{#if probe.status === 'completed' && probe.diagnostics.length > 0}
					<Button variant="outline" onclick={copyReport}>
						{#if copied}
							<Check class="w-4 h-4 mr-1.5" />
							Copied!
						{:else}
							<Copy class="w-4 h-4 mr-1.5" />
							Copy Report
						{/if}
					</Button>
				{/if}
				<Button variant="outline" onclick={() => (open = false)}>Close</Button>
			</div>
		</Dialog.Content>
	</Dialog.Portal>
</Dialog.Root>
