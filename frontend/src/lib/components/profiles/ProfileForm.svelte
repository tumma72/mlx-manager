<script lang="ts">
	import type { ServerProfile, ServerProfileCreate, ServerProfileUpdate } from '$api';
	import { models as modelsApi } from '$api';
	import { Card, Button, Input, Select } from '$components/ui';

	interface Props {
		profile?: ServerProfile;
		nextPort?: number;
		initialModelPath?: string;
		onSubmit: (data: ServerProfileCreate | ServerProfileUpdate) => Promise<void>;
		onCancel: () => void;
	}

	let { profile, nextPort = 10240, initialModelPath = '', onSubmit, onCancel }: Props = $props();

	let loading = $state(false);
	let error = $state<string | null>(null);
	let detectingOptions = $state(false);
	let detectedFamily = $state<string | null>(null);

	// Form state - only fields supported by mlx-openai-server CLI
	// Supported: model-path, model-type (lm|multimodal), port, host,
	// max-concurrency, queue-timeout, queue-size
	// Also: tool-call-parser, reasoning-parser, message-converter for MiniMax/Qwen/GLM
	let name = $state('');
	let description = $state('');
	let modelPath = $state('');
	let modelType = $state('lm');
	let port = $state(10240);
	let host = $state('127.0.0.1');
	let maxConcurrency = $state(1);
	let queueTimeout = $state(300);
	let queueSize = $state(100);
	let autoStart = $state(false);
	let toolCallParser = $state('');
	let reasoningParser = $state('');
	let messageConverter = $state('');

	let showAdvanced = $state(false);

	// Available parser options
	const parserOptions = [
		{ value: '', label: 'Default' },
		{ value: 'minimax_m2', label: 'MiniMax M2' },
		{ value: 'qwen3', label: 'Qwen3' },
		{ value: 'glm4', label: 'GLM4' },
		{ value: 'hermes', label: 'Hermes' },
		{ value: 'llama', label: 'Llama' },
		{ value: 'mistral', label: 'Mistral' }
	];

	// Reset form when profile, nextPort, or initialModelPath changes
	$effect(() => {
		name = profile?.name ?? '';
		description = profile?.description ?? '';
		modelPath = profile?.model_path ?? initialModelPath;
		// Map unsupported model types to 'lm'
		const profileModelType = profile?.model_type ?? 'lm';
		modelType = ['lm', 'multimodal'].includes(profileModelType) ? profileModelType : 'lm';
		port = profile?.port ?? nextPort;
		host = profile?.host ?? '127.0.0.1';
		maxConcurrency = profile?.max_concurrency ?? 1;
		queueTimeout = profile?.queue_timeout ?? 300;
		queueSize = profile?.queue_size ?? 100;
		autoStart = profile?.auto_start ?? false;
		toolCallParser = profile?.tool_call_parser ?? '';
		reasoningParser = profile?.reasoning_parser ?? '';
		messageConverter = profile?.message_converter ?? '';
	});

	// Auto-detect model family when model path changes
	async function detectModelOptions(path: string) {
		if (!path) {
			detectedFamily = null;
			return;
		}

		detectingOptions = true;
		try {
			const info = await modelsApi.detectOptions(path);
			detectedFamily = info.model_family;

			// Only auto-fill if no values are already set
			if (info.recommended_options) {
				if (!toolCallParser && info.recommended_options.tool_call_parser) {
					toolCallParser = info.recommended_options.tool_call_parser;
				}
				if (!reasoningParser && info.recommended_options.reasoning_parser) {
					reasoningParser = info.recommended_options.reasoning_parser;
				}
				if (!messageConverter && info.recommended_options.message_converter) {
					messageConverter = info.recommended_options.message_converter;
				}
			}
		} catch {
			// Detection failed - not critical, user can still set manually
			detectedFamily = null;
		} finally {
			detectingOptions = false;
		}
	}

	// Detect options when model path changes (with debounce)
	let detectTimeout: ReturnType<typeof setTimeout> | null = null;
	$effect(() => {
		if (modelPath && !profile) {
			// Only auto-detect for new profiles
			if (detectTimeout) clearTimeout(detectTimeout);
			detectTimeout = setTimeout(() => detectModelOptions(modelPath), 500);
		}
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
				port,
				host,
				max_concurrency: maxConcurrency,
				queue_timeout: queueTimeout,
				queue_size: queueSize,
				auto_start: autoStart,
				tool_call_parser: toolCallParser || undefined,
				reasoning_parser: reasoningParser || undefined,
				message_converter: messageConverter || undefined
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
			<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
				<div>
					<label for="name" class="block text-sm font-medium mb-1">Name *</label>
					<Input id="name" bind:value={name} placeholder="e.g., Coding Assistant" required />
				</div>

				<div>
					<label for="port" class="block text-sm font-medium mb-1">Port *</label>
					<Input id="port" type="number" bind:value={port} required />
				</div>
			</div>

			<div>
				<label for="description" class="block text-sm font-medium mb-1">Description</label>
				<Input id="description" bind:value={description} placeholder="Optional description" />
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
				</Select>
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
					<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
						<div>
							<label for="host" class="block text-sm font-medium mb-1">Host</label>
							<Input id="host" bind:value={host} />
						</div>

						<div>
							<label for="maxConcurrency" class="block text-sm font-medium mb-1"
								>Max Concurrency</label
							>
							<Input id="maxConcurrency" type="number" bind:value={maxConcurrency} />
						</div>
					</div>

					<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
						<div>
							<label for="queueTimeout" class="block text-sm font-medium mb-1"
								>Queue Timeout (seconds)</label
							>
							<Input id="queueTimeout" type="number" bind:value={queueTimeout} />
						</div>

						<div>
							<label for="queueSize" class="block text-sm font-medium mb-1"
								>Queue Size</label
							>
							<Input id="queueSize" type="number" bind:value={queueSize} />
						</div>
					</div>

					<!-- Parser Options for MiniMax, Qwen, GLM models -->
					<div class="pt-4 border-t">
						<div class="flex items-center gap-2 mb-3">
							<h3 class="text-sm font-medium">Model-Specific Parsers</h3>
							{#if detectingOptions}
								<span class="text-xs text-muted-foreground">Detecting...</span>
							{:else if detectedFamily}
								<span class="text-xs text-green-600">Detected: {detectedFamily}</span>
							{/if}
						</div>
						<p class="text-xs text-muted-foreground mb-3">
							Required for MiniMax, Qwen3, and GLM models to enable tool calling and reasoning.
						</p>

						<div class="grid grid-cols-1 md:grid-cols-3 gap-4">
							<div>
								<label for="toolCallParser" class="block text-sm font-medium mb-1"
									>Tool Call Parser</label
								>
								<Select id="toolCallParser" bind:value={toolCallParser}>
									{#each parserOptions as opt (opt.value)}
										<option value={opt.value}>{opt.label}</option>
									{/each}
								</Select>
							</div>

							<div>
								<label for="reasoningParser" class="block text-sm font-medium mb-1"
									>Reasoning Parser</label
								>
								<Select id="reasoningParser" bind:value={reasoningParser}>
									{#each parserOptions as opt (opt.value)}
										<option value={opt.value}>{opt.label}</option>
									{/each}
								</Select>
							</div>

							<div>
								<label for="messageConverter" class="block text-sm font-medium mb-1"
									>Message Converter</label
								>
								<Select id="messageConverter" bind:value={messageConverter}>
									{#each parserOptions as opt (opt.value)}
										<option value={opt.value}>{opt.label}</option>
									{/each}
								</Select>
							</div>
						</div>
					</div>

					<div class="space-y-2">
						<label class="flex items-center gap-2">
							<input type="checkbox" bind:checked={autoStart} class="rounded" />
							<span class="text-sm">Start on Login (launchd)</span>
						</label>
					</div>
				</div>
			{/if}
		</div>

		{#if error}
			<div class="mt-4 text-sm text-red-500">{error}</div>
		{/if}

		<div class="mt-6 flex justify-end gap-3">
			<Button variant="outline" onclick={onCancel} disabled={loading}>Cancel</Button>
			<Button type="submit" disabled={loading}>
				{loading ? 'Saving...' : profile ? 'Update Profile' : 'Create Profile'}
			</Button>
		</div>
	</Card>
</form>
