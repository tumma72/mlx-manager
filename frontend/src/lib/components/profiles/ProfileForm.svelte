<script lang="ts">
	import type { ExecutionProfile, ExecutionProfileCreate, ExecutionProfileUpdate, DownloadedModel, TemplateParamInfo } from '$api';
	import { models as modelsApi } from '$api';
	import { Card, Button, Input } from '$components/ui';
	import { resolve } from '$app/paths';
	import { onMount } from 'svelte';

	interface Props {
		profile?: ExecutionProfile;
		initialModelId?: number;
		onSubmit: (data: ExecutionProfileCreate | ExecutionProfileUpdate) => Promise<void>;
		onCancel: () => void;
	}

	let { profile, initialModelId, onSubmit, onCancel }: Props = $props();

	let loading = $state(false);
	let error = $state<string | null>(null);
	let downloadedModels = $state<DownloadedModel[]>([]);
	let modelsLoading = $state(true);

	// Form state
	let name = $state('');
	let description = $state('');
	let systemPrompt = $state('');
	let modelId = $state<number | string | null>(null);
	let autoStart = $state(false);

	// Generation parameters
	let temperature = $state<number | null>(null);
	let maxTokens = $state<number | null>(null);
	let topP = $state<number | null>(null);

	// Tool calling
	let enablePromptInjection = $state(false);

	// Audio parameters
	let ttsDefaultVoice = $state<string | null>(null);
	let ttsDefaultSpeed = $state<number | null>(null);
	let ttsSampleRate = $state<number | null>(null);
	let sttDefaultLanguage = $state<string | null>(null);

	// Model-specific template options
	let modelOptions = $state<Record<string, unknown>>({});

	let showAdvanced = $state(false);

	// Derive model type from selected model
	const selectedModel = $derived(downloadedModels.find((m) => m.id === (typeof modelId === 'number' ? modelId : null)));
	const modelType = $derived(selectedModel?.model_type ?? null);
	const isTextOrVision = $derived(modelType === 'text-gen' || modelType === 'vision');
	const isAudio = $derived(modelType === 'audio');

	// Derive template params from selected model's capabilities
	const templateParams = $derived(selectedModel?.capabilities?.template_params ?? null);

	// Load downloaded models on mount
	onMount(async () => {
		try {
			downloadedModels = await modelsApi.listDownloaded();
		} catch (e) {
			console.error('Failed to load models:', e);
		} finally {
			modelsLoading = false;
		}
	});

	// Reset form when profile changes
	$effect(() => {
		name = profile?.name ?? '';
		description = profile?.description ?? '';
		systemPrompt = profile?.context?.system_prompt ?? '';
		modelId = profile?.model_id ?? initialModelId ?? null;
		autoStart = profile?.auto_start ?? false;
		// Generation parameters
		temperature = profile?.inference?.temperature ?? null;
		maxTokens = profile?.inference?.max_tokens ?? null;
		topP = profile?.inference?.top_p ?? null;
		// Tool calling
		enablePromptInjection = profile?.context?.enable_tool_injection ?? false;
		// Audio parameters
		ttsDefaultVoice = profile?.audio?.tts_voice ?? null;
		ttsDefaultSpeed = profile?.audio?.tts_speed ?? null;
		ttsSampleRate = profile?.audio?.tts_sample_rate ?? null;
		sttDefaultLanguage = profile?.audio?.stt_language ?? null;
		// Model options
		modelOptions = profile?.model_options ?? {};
	});

	async function handleSubmit(e: Event) {
		e.preventDefault();
		const numericModelId = typeof modelId === 'number' ? modelId : null;
		if (!numericModelId) {
			error = 'Please select a model';
			return;
		}
		loading = true;
		error = null;

		try {
			const data: ExecutionProfileCreate | ExecutionProfileUpdate = {
				name,
				description: description || undefined,
				model_id: numericModelId,
				auto_start: autoStart,
				// Nest inference parameters (only for text/vision)
				inference: isTextOrVision
					? {
							temperature: temperature ?? null,
							max_tokens: maxTokens ?? null,
							top_p: topP ?? null,
						}
					: undefined,
				// Nest context (only for text/vision)
				context: isTextOrVision
					? {
							context_length: null,
							system_prompt: systemPrompt || null,
							enable_tool_injection: enablePromptInjection,
						}
					: undefined,
				// Nest audio parameters (only for audio)
				audio: isAudio
					? {
							tts_voice: ttsDefaultVoice ?? null,
							tts_speed: ttsDefaultSpeed ?? null,
							tts_sample_rate: ttsSampleRate ?? null,
							stt_language: sttDefaultLanguage ?? null,
						}
					: undefined,
				// Model-specific template options
				model_options: Object.keys(modelOptions).length > 0 ? modelOptions : undefined,
			};

			await onSubmit(data);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to save profile';
		} finally {
			loading = false;
		}
	}

	function getModelLabel(model: DownloadedModel): string {
		const type = model.model_type ?? 'unknown';
		const size = model.size_gb ? ` (${model.size_gb} GB)` : '';
		return `${model.repo_id}${size} [${type}]`;
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

			<!-- Model Selection -->
			<div>
				<label for="modelSelect" class="block text-sm font-medium mb-1">Model *</label>
				{#if modelsLoading}
					<p class="text-sm text-muted-foreground">Loading models...</p>
				{:else if downloadedModels.length === 0}
					<p class="text-sm text-muted-foreground">
						No models downloaded. <a href={resolve('/models')} class="text-primary hover:underline">Download a model</a> first.
					</p>
				{:else}
					<select
						id="modelSelect"
						bind:value={modelId}
						class="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
					>
						<option value="">Select a model...</option>
						{#each downloadedModels as model (model.id)}
							<option value={model.id}>{getModelLabel(model)}</option>
						{/each}
					</select>
					{#if selectedModel}
						<p class="text-xs text-muted-foreground mt-1">
							Type: {selectedModel.model_type ?? 'unknown'}
							{#if selectedModel.size_gb}
								| Size: {selectedModel.size_gb} GB
							{/if}
						</p>
					{/if}
				{/if}
			</div>

			<!-- System Prompt (text/vision only) -->
			{#if isTextOrVision}
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
			{/if}

			<!-- Generation Settings (text/vision only) -->
			{#if isTextOrVision}
				<div class="pt-4 border-t">
					<h3 class="text-sm font-medium mb-3">Generation Settings</h3>
					<p class="text-xs text-muted-foreground mb-4">
						Control how the model generates responses. These can be overridden per-request.
					</p>

					<div class="space-y-4">
						<div>
							<label for="temperature" class="block text-sm font-medium mb-1">
								Temperature: {(temperature ?? 0.7).toFixed(1)}
							</label>
							<input
								id="temperature"
								type="range"
								min="0"
								max="2"
								step="0.1"
								value={temperature ?? 0.7}
								oninput={(e) => (temperature = parseFloat(e.currentTarget.value))}
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
								value={maxTokens ?? 4096}
								oninput={(e) => (maxTokens = parseInt(e.currentTarget.value) || null)}
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
								Top P (Nucleus Sampling): {(topP ?? 1.0).toFixed(2)}
							</label>
							<input
								id="topP"
								type="range"
								min="0"
								max="1"
								step="0.05"
								value={topP ?? 1.0}
								oninput={(e) => (topP = parseFloat(e.currentTarget.value))}
								class="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer"
							/>
							<div class="flex justify-between text-xs text-muted-foreground mt-1">
								<span>Focused (0)</span>
								<span>Diverse (1)</span>
							</div>
						</div>
					</div>
				</div>
			{/if}

			<!-- Audio Settings (audio only) -->
			{#if isAudio}
				<div class="pt-4 border-t">
					<h3 class="text-sm font-medium mb-3">Audio Settings</h3>

					<div class="space-y-4">
						<div>
							<label for="ttsVoice" class="block text-sm font-medium mb-1">Default Voice</label>
							<Input
								id="ttsVoice"
								value={ttsDefaultVoice ?? ''}
								oninput={(e: Event) => {
									const target = e.currentTarget as HTMLInputElement;
									ttsDefaultVoice = target.value || null;
								}}
								placeholder="e.g., af_heart"
							/>
						</div>

						<div>
							<label for="ttsSpeed" class="block text-sm font-medium mb-1">
								Speech Speed: {(ttsDefaultSpeed ?? 1.0).toFixed(1)}x
							</label>
							<input
								id="ttsSpeed"
								type="range"
								min="0.25"
								max="4"
								step="0.25"
								value={ttsDefaultSpeed ?? 1.0}
								oninput={(e) => (ttsDefaultSpeed = parseFloat(e.currentTarget.value))}
								class="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer"
							/>
							<div class="flex justify-between text-xs text-muted-foreground mt-1">
								<span>Slow (0.25x)</span>
								<span>Fast (4x)</span>
							</div>
						</div>

						<div>
							<label for="ttsSampleRate" class="block text-sm font-medium mb-1">Sample Rate</label>
							<input
								id="ttsSampleRate"
								type="number"
								value={ttsSampleRate ?? ''}
								oninput={(e) => (ttsSampleRate = parseInt(e.currentTarget.value) || null)}
								placeholder="24000"
								class="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
							/>
						</div>

						<div>
							<label for="sttLanguage" class="block text-sm font-medium mb-1">STT Language</label>
							<Input
								id="sttLanguage"
								value={sttDefaultLanguage ?? ''}
								oninput={(e: Event) => {
									const target = e.currentTarget as HTMLInputElement;
									sttDefaultLanguage = target.value || null;
								}}
								placeholder="e.g., en"
							/>
						</div>
					</div>
				</div>
			{/if}

			<!-- Model Options (from discovered template params) -->
			{#if templateParams && Object.keys(templateParams).length > 0}
				<div class="pt-4 border-t">
					<h3 class="text-sm font-medium mb-3">Model Options</h3>
					<p class="text-xs text-muted-foreground mb-4">
						Options discovered from this model's chat template. Changes affect how the model processes conversations.
					</p>

					<div class="space-y-3">
						{#each Object.entries(templateParams) as [key, param]}
							{#if param.param_type === 'bool'}
								<div class="space-y-1">
									<label class="flex items-center gap-2">
										<input
											type="checkbox"
											checked={modelOptions[key] !== undefined
												? Boolean(modelOptions[key])
												: Boolean(param.default)}
											onchange={(e) => {
												modelOptions = { ...modelOptions, [key]: e.currentTarget.checked };
											}}
											class="rounded"
										/>
										<span class="text-sm">{param.label}</span>
									</label>
									<p class="text-xs text-muted-foreground ml-6">{param.description}</p>
								</div>
							{:else}
								<div>
									<label for={`opt-${key}`} class="block text-sm font-medium mb-1">{param.label}</label>
									<Input
										id={`opt-${key}`}
										value={String(modelOptions[key] ?? param.default ?? '')}
										oninput={(e: Event) => {
											const target = e.currentTarget as HTMLInputElement;
											modelOptions = { ...modelOptions, [key]: target.value || null };
										}}
										placeholder={param.default ? String(param.default) : ''}
									/>
									<p class="text-xs text-muted-foreground mt-1">{param.description}</p>
								</div>
							{/if}
						{/each}
					</div>
				</div>
			{/if}

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
					{#if isTextOrVision}
						<div class="space-y-2">
							<label class="flex items-center gap-2">
								<input type="checkbox" bind:checked={enablePromptInjection} class="rounded" />
								<span class="text-sm">Enable experimental tool support (via prompt injection)</span>
							</label>
							<p class="text-xs text-muted-foreground ml-6">
								Experimental: Enables tool calling for models without native support by
								injecting tool descriptions into the system prompt. Results may vary.
							</p>
						</div>
					{/if}
				</div>
			{/if}
		</div>

		{#if error}
			<div class="mt-4 text-sm text-red-500 dark:text-red-400">{error}</div>
		{/if}

		<div class="mt-6 flex justify-end gap-3">
			<Button variant="outline" onclick={onCancel} disabled={loading}>Cancel</Button>
			<Button type="submit" disabled={loading || typeof modelId !== 'number'}>
				{loading ? 'Saving...' : profile ? 'Update Profile' : 'Create Profile'}
			</Button>
		</div>
	</Card>
</form>
