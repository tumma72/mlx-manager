<script lang="ts">
	import type { ServerProfile, ServerProfileCreate, ServerProfileUpdate } from '$api';
	import { Card, Button, Input, Select } from '$components/ui';

	interface Props {
		profile?: ServerProfile;
		nextPort?: number;
		onSubmit: (data: ServerProfileCreate | ServerProfileUpdate) => Promise<void>;
		onCancel: () => void;
	}

	let { profile, nextPort = 10240, onSubmit, onCancel }: Props = $props();

	let loading = $state(false);
	let error = $state<string | null>(null);

	// Form state
	let name = $state(profile?.name ?? '');
	let description = $state(profile?.description ?? '');
	let modelPath = $state(profile?.model_path ?? '');
	let modelType = $state(profile?.model_type ?? 'lm');
	let port = $state(profile?.port ?? nextPort);
	let host = $state(profile?.host ?? '127.0.0.1');
	let contextLength = $state<number | undefined>(profile?.context_length ?? undefined);
	let maxConcurrency = $state(profile?.max_concurrency ?? 1);
	let queueTimeout = $state(profile?.queue_timeout ?? 300);
	let queueSize = $state(profile?.queue_size ?? 100);
	let toolCallParser = $state(profile?.tool_call_parser ?? '');
	let reasoningParser = $state(profile?.reasoning_parser ?? '');
	let enableAutoToolChoice = $state(profile?.enable_auto_tool_choice ?? false);
	let trustRemoteCode = $state(profile?.trust_remote_code ?? false);
	let logLevel = $state(profile?.log_level ?? 'INFO');
	let autoStart = $state(profile?.auto_start ?? false);

	let showAdvanced = $state(false);

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
				context_length: contextLength || undefined,
				max_concurrency: maxConcurrency,
				queue_timeout: queueTimeout,
				queue_size: queueSize,
				tool_call_parser: toolCallParser || undefined,
				reasoning_parser: reasoningParser || undefined,
				enable_auto_tool_choice: enableAutoToolChoice,
				trust_remote_code: trustRemoteCode,
				log_level: logLevel,
				auto_start: autoStart
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
					<option value="multimodal">Multimodal</option>
					<option value="whisper">Whisper</option>
					<option value="embeddings">Embeddings</option>
					<option value="image-generation">Image Generation</option>
					<option value="image-edit">Image Edit</option>
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
							<label for="contextLength" class="block text-sm font-medium mb-1"
								>Context Length</label
							>
							<Input
								id="contextLength"
								type="number"
								bind:value={contextLength}
								placeholder="Model default"
							/>
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
							<label for="toolCallParser" class="block text-sm font-medium mb-1"
								>Tool Call Parser</label
							>
							<Select id="toolCallParser" bind:value={toolCallParser}>
								<option value="">None</option>
								<option value="qwen3">qwen3</option>
								<option value="glm4_moe">glm4_moe</option>
								<option value="harmony">harmony</option>
								<option value="minimax">minimax</option>
							</Select>
						</div>

						<div>
							<label for="reasoningParser" class="block text-sm font-medium mb-1"
								>Reasoning Parser</label
							>
							<Select id="reasoningParser" bind:value={reasoningParser}>
								<option value="">None</option>
								<option value="qwen3">qwen3</option>
								<option value="deepseek_r1">deepseek_r1</option>
							</Select>
						</div>
					</div>

					<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
						<div>
							<label for="logLevel" class="block text-sm font-medium mb-1">Log Level</label>
							<Select id="logLevel" bind:value={logLevel}>
								<option value="DEBUG">DEBUG</option>
								<option value="INFO">INFO</option>
								<option value="WARNING">WARNING</option>
								<option value="ERROR">ERROR</option>
							</Select>
						</div>

						<div>
							<label for="host" class="block text-sm font-medium mb-1">Host</label>
							<Input id="host" bind:value={host} />
						</div>
					</div>

					<div class="space-y-2">
						<label class="flex items-center gap-2">
							<input type="checkbox" bind:checked={enableAutoToolChoice} class="rounded" />
							<span class="text-sm">Enable Auto Tool Choice</span>
						</label>

						<label class="flex items-center gap-2">
							<input type="checkbox" bind:checked={trustRemoteCode} class="rounded" />
							<span class="text-sm">Trust Remote Code</span>
						</label>

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
