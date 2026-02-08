<script lang="ts">
	import { Loader2, CheckCircle2, XCircle, MinusCircle } from 'lucide-svelte';
	import type { ProbeState } from '$stores/probe.svelte';

	interface Props {
		probe: ProbeState;
	}

	let { probe }: Props = $props();

	const stepLabels: Record<string, string> = {
		load_model: 'Loading model',
		check_context: 'Checking context',
		test_thinking: 'Testing thinking',
		test_tools: 'Testing tools',
		save_results: 'Saving results',
		cleanup: 'Cleaning up'
	};
</script>

<div class="space-y-1 text-xs">
	{#each probe.steps as step (step.step)}
		<div class="flex items-center gap-1.5">
			{#if step.status === 'running'}
				<Loader2 class="w-3 h-3 animate-spin text-blue-500" />
			{:else if step.status === 'completed'}
				<CheckCircle2 class="w-3 h-3 text-green-500" />
			{:else if step.status === 'failed'}
				<XCircle class="w-3 h-3 text-red-500" />
			{:else}
				<MinusCircle class="w-3 h-3 text-muted-foreground" />
			{/if}
			<span class={step.status === 'failed' ? 'text-red-500' : 'text-muted-foreground'}>
				{stepLabels[step.step] ?? step.step}
			</span>
			{#if step.status === 'failed' && step.error}
				<span class="text-red-400 truncate">— {step.error}</span>
			{/if}
		</div>
	{/each}
	{#if probe.error}
		<p class="text-red-500 mt-1">{probe.error}</p>
	{/if}
</div>
