<script lang="ts">
	import { settings } from '$lib/api/client';
	import { Search, Loader2, Check, X } from 'lucide-svelte';
	import type { RuleTestResult, BackendMapping } from '$lib/api/types';

	interface Props {
		rules: BackendMapping[];
	}

	let { rules }: Props = $props();

	let modelName = $state('');
	let testing = $state(false);
	let result = $state<RuleTestResult | null>(null);
	let error = $state<string | null>(null);

	// Use $derived.by() for complex derivations to maintain type narrowing
	const matchedRule = $derived.by(() => {
		const res = result;
		return res?.matched_rule_id ? rules.find((r) => r.id === res.matched_rule_id) ?? null : null;
	});

	const backendTypeLabels: Record<string, string> = {
		local: 'Local (MLX)',
		openai: 'OpenAI',
		anthropic: 'Anthropic'
	};

	async function handleTest() {
		if (!modelName.trim()) return;

		testing = true;
		result = null;
		error = null;

		try {
			result = await settings.testRule(modelName.trim());
		} catch (e) {
			error = e instanceof Error ? e.message : 'Test failed';
		} finally {
			testing = false;
		}
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter') {
			handleTest();
		}
	}
</script>

<div class="space-y-3 rounded-lg border bg-card p-4">
	<h3 class="text-sm font-medium">Test Model Routing</h3>

	<div class="flex gap-2">
		<div class="flex-1">
			<input
				type="text"
				bind:value={modelName}
				onkeydown={handleKeydown}
				placeholder="Enter model name to test (e.g., gpt-4, claude-3)"
				class="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
			/>
		</div>
		<button
			onclick={handleTest}
			disabled={testing || !modelName.trim()}
			class="inline-flex h-10 items-center justify-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:pointer-events-none disabled:opacity-50"
		>
			{#if testing}
				<Loader2 class="h-4 w-4 animate-spin" />
			{:else}
				<Search class="h-4 w-4" />
			{/if}
			Test
		</button>
	</div>

	{#if error}
		<div class="flex items-center gap-2 rounded-md bg-destructive/10 p-3 text-sm text-destructive">
			<X class="h-4 w-4" />
			{error}
		</div>
	{/if}

	{#if result}
		<div
			class="rounded-md p-3 {matchedRule
				? 'bg-green-50 dark:bg-green-900/20'
				: 'bg-muted'}"
		>
			{#if matchedRule}
				<div class="flex items-center gap-2 text-sm">
					<Check class="h-4 w-4 text-green-600 dark:text-green-400" />
					<span class="font-medium">Matched:</span>
					<code class="rounded bg-muted px-1 dark:bg-background">{matchedRule.model_pattern}</code>
					<span class="text-muted-foreground">-&gt;</span>
					<span class="font-medium"
						>{backendTypeLabels[result.backend_type] || result.backend_type}</span
					>
				</div>
			{:else}
				<p class="text-sm text-muted-foreground">
					No rule matched. Routes to <span class="font-medium">Local (MLX)</span> (default).
				</p>
			{/if}
		</div>
	{/if}
</div>
