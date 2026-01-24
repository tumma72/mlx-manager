<script lang="ts">
	import { Collapsible } from 'bits-ui';
	import { ChevronDown, ChevronRight, Wrench } from 'lucide-svelte';

	interface ToolCallData {
		id: string;
		name: string;
		arguments: string;
		result?: string;
		error?: string;
	}

	interface Props {
		calls: ToolCallData[];
	}

	let { calls }: Props = $props();
	let expanded = $state(false);

	const label = $derived(
		calls.length === 1
			? `Tool: ${calls[0].name}`
			: `Tool Calls: ${calls.length}`
	);

	function formatArgs(args: string): string {
		try {
			return JSON.stringify(JSON.parse(args), null, 2);
		} catch {
			return args;
		}
	}
</script>

<div class="my-2">
	<Collapsible.Root bind:open={expanded}>
		<Collapsible.Trigger
			class="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
		>
			{#if expanded}
				<ChevronDown class="w-4 h-4" />
			{:else}
				<ChevronRight class="w-4 h-4" />
			{/if}
			<Wrench class="w-4 h-4" />
			<span>{label}</span>
		</Collapsible.Trigger>
		<Collapsible.Content>
			<div class="mt-2 pl-6 border-l-2 border-amber-300 dark:border-amber-700 space-y-3">
				{#each calls as call (call.id)}
					<div class="text-sm">
						<div class="font-medium text-foreground">{call.name}</div>
						<pre class="mt-1 p-2 rounded bg-muted text-xs font-mono overflow-x-auto whitespace-pre-wrap">{formatArgs(call.arguments)}</pre>
						{#if call.result}
							<div class="mt-1 text-xs text-muted-foreground">Result:</div>
							<pre class="mt-0.5 p-2 rounded bg-green-50 dark:bg-green-950/30 text-xs font-mono overflow-x-auto whitespace-pre-wrap text-green-700 dark:text-green-300">{call.result}</pre>
						{/if}
						{#if call.error}
							<div class="mt-1 text-xs text-red-600 dark:text-red-400">{call.error}</div>
						{/if}
					</div>
				{/each}
			</div>
		</Collapsible.Content>
	</Collapsible.Root>
</div>
