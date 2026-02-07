<script lang="ts">
	import type { ServerProfile, ServerProfileCreate, ServerProfileUpdate } from '$api';
	import { Card, Button, Input, Select } from '$components/ui';

	interface Props {
		profile?: ServerProfile;
		initialModelPath?: string;
		onSubmit: (data: ServerProfileCreate | ServerProfileUpdate) => Promise<void>;
		onCancel: () => void;
	}

	let { profile, initialModelPath = '', onSubmit, onCancel }: Props = $props();

	let loading = $state(false);
	let error = $state<string | null>(null);

	// Form state - profile configuration fields
	let name = $state('');
	let description = $state('');
	let systemPrompt = $state('');
	let modelPath = $state('');
	let modelType = $state('lm');
	let autoStart = $state(false);

	// Generation parameters
	let temperature = $state(0.7);
	let maxTokens = $state(4096);
	let topP = $state(1.0);

	// Tool calling
	let enablePromptInjection = $state(false);

	let showAdvanced = $state(false);

	// Reset form when profile or initialModelPath changes
	$effect(() => {
		name = profile?.name ?? '';
		description = profile?.description ?? '';
		systemPrompt = profile?.system_prompt ?? '';
		modelPath = profile?.model_path ?? initialModelPath;
		// Map unsupported model types to 'lm'
		const profileModelType = profile?.model_type ?? 'lm';
		modelType = ['lm', 'multimodal', 'embeddings', 'audio'].includes(profileModelType)
			? profileModelType
			: 'lm';
		autoStart = profile?.auto_start ?? false;
		// Generation parameters
		temperature = profile?.temperature ?? 0.7;
		maxTokens = profile?.max_tokens ?? 4096;
		topP = profile?.top_p ?? 1.0;
		// Tool calling
		enablePromptInjection = profile?.enable_prompt_injection ?? false;
	});

	async function handleSubmit(e: Event) {
		e.preventDefault();
		loading = true;
		error = null;

		try {
			const data: ServerProfileCreate | ServerProfileUpdate = {
				name,
				description: description || undefined,
				model_path: modelPath,
				model_type: modelType,
				auto_start: autoStart,
				system_prompt: systemPrompt || undefined,
				// Generation parameters
				temperature,
				max_tokens: maxTokens,
				top_p: topP,
				// Tool calling
				enable_prompt_injection: enablePromptInjection
			};

			await onSubmit(data);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to save profile';
		} finally {
			loading = false;
		}
	}
</script>

<form onsubmit={handleSubmit}>
	<Card class="p-6">
		<h2 class="text-xl font-semibold mb-6">
			{profile ? 'Edit Profile' : 'Create Profile'}
		</h2>

		<div class="space-y-4">
			<!-- Basic Info -->
			<div>
				<label for="name" class="block text-sm font-medium mb-1">Name *</label>
				<Input id="name" bind:value={name} placeholder="e.g., Coding Assistant" required />
			</div>

			<div>
				<label for="description" class="block text-sm font-medium mb-1">Description</label>
				<textarea
					id="description"
					bind:value={description}
					placeholder="Optional description"
					rows="2"
					class="flex w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
				></textarea>
			</div>

			<div>
				<label for="systemPrompt" class="block text-sm font-medium mb-1">
					System Prompt
					{#if systemPrompt.length > 0}
						<span class="text-xs text-muted-foreground ml-2">
							{systemPrompt.length} chars
							{#if systemPrompt.length > 2000}
								<span class="text-amber-500">(long prompt may affect performance)</span>
							{/if}
						</span>
					{/if}
				</label>
				<textarea
					id="systemPrompt"
					bind:value={systemPrompt}
					placeholder="e.g., You are a helpful coding assistant specializing in Python..."
					rows="4"
					class="flex w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
				></textarea>
				<p class="text-xs text-muted-foreground mt-1">
					Sets the model's behavior context. Appears as first message when chatting.
				</p>
			</div>

			<div>
				<label for="modelPath" class="block text-sm font-medium mb-1">Model Path *</label>
				<Input
					id="modelPath"
					bind:value={modelPath}
					placeholder="e.g., mlx-community/Qwen2.5-Coder-32B-Instruct-4bit"
					required
				/>
			</div>

			<div>
				<label for="modelType" class="block text-sm font-medium mb-1">Model Type</label>
				<Select id="modelType" bind:value={modelType}>
					<option value="lm">Language Model (lm)</option>
					<option value="multimodal">Multimodal (Vision)</option>
					<option value="embeddings">Embeddings</option>
					<option value="audio">Audio (TTS/STT)</option>
				</Select>
			</div>

			<!-- Generation Settings -->
			<div class="pt-4 border-t">
				<h3 class="text-sm font-medium mb-3">Generation Settings</h3>
				<p class="text-xs text-muted-foreground mb-4">
					Control how the model generates responses. These can be overridden per-request.
				</p>

				<div class="space-y-4">
					<div>
						<label for="temperature" class="block text-sm font-medium mb-1">
							Temperature: {temperature.toFixed(1)}
						</label>
						<input
							id="temperature"
							type="range"
							min="0"
							max="2"
							step="0.1"
							bind:value={temperature}
							class="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer"
						/>
						<div class="flex justify-between text-xs text-muted-foreground mt-1">
							<span>Deterministic (0)</span>
							<span>Creative (2)</span>
						</div>
					</div>

					<div>
						<label for="maxTokens" class="block text-sm font-medium mb-1">Max Tokens</label>
						<input
							id="maxTokens"
							type="number"
							bind:value={maxTokens}
							min="1"
							max="128000"
							placeholder="4096"
							class="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
						/>
						<p class="text-xs text-muted-foreground mt-1">
							Maximum number of tokens to generate (1-128,000).
						</p>
					</div>

					<div>
						<label for="topP" class="block text-sm font-medium mb-1">
							Top P (Nucleus Sampling): {topP.toFixed(2)}
						</label>
						<input
							id="topP"
							type="range"
							min="0"
							max="1"
							step="0.05"
							bind:value={topP}
							class="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer"
						/>
						<div class="flex justify-between text-xs text-muted-foreground mt-1">
							<span>Focused (0)</span>
							<span>Diverse (1)</span>
						</div>
					</div>
				</div>
			</div>

			<!-- Advanced Options -->
			<div class="pt-4">
				<button
					type="button"
					class="text-sm text-primary hover:underline"
					onclick={() => (showAdvanced = !showAdvanced)}
				>
					{showAdvanced ? 'Hide' : 'Show'} Advanced Options
				</button>
			</div>

			{#if showAdvanced}
				<div class="space-y-4 pt-4 border-t">
					<div class="space-y-2">
						<label class="flex items-center gap-2">
							<input type="checkbox" bind:checked={autoStart} class="rounded" />
							<span class="text-sm">Auto-load on startup</span>
						</label>
						<p class="text-xs text-muted-foreground ml-6">
							Preload this model when the server starts for faster first response.
						</p>
					</div>
					<div class="space-y-2">
						<label class="flex items-center gap-2">
							<input type="checkbox" bind:checked={enablePromptInjection} class="rounded" />
							<span class="text-sm">Enable experimental tool support (via prompt injection)</span>
						</label>
						<p class="text-xs text-muted-foreground ml-6">
							Experimental: Enables tool calling for models without native support by
							injecting tool descriptions into the system prompt. Results may vary. Use
							'Test Capabilities' on the Models page to check if your model has native
							support.
						</p>
					</div>
				</div>
			{/if}
		</div>

		{#if error}
			<div class="mt-4 text-sm text-red-500 dark:text-red-400">{error}</div>
		{/if}

		<div class="mt-6 flex justify-end gap-3">
			<Button variant="outline" onclick={onCancel} disabled={loading}>Cancel</Button>
			<Button type="submit" disabled={loading}>
				{loading ? 'Saving...' : profile ? 'Update Profile' : 'Create Profile'}
			</Button>
		</div>
	</Card>
</form>
