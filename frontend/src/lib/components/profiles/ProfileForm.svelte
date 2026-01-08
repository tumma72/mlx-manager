<script lang="ts">
	import type { ServerProfile, ServerProfileCreate, ServerProfileUpdate } from '$api';
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

	// Form state - only fields supported by mlx-openai-server CLI
	// Supported: model-path, model-type (lm|multimodal), port, host,
	// max-concurrency, queue-timeout, queue-size
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

	let showAdvanced = $state(false);

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
