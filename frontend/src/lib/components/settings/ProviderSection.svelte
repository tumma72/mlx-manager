<script lang="ts">
	import { onMount } from 'svelte';
	import { Accordion } from 'bits-ui';
	import { ChevronDown } from 'lucide-svelte';
	import { settings } from '$lib/api/client';
	import type { CloudCredential, BackendType } from '$lib/api/types';
	import ProviderForm from './ProviderForm.svelte';
	import { Card } from '$components/ui';

	// Cloud provider types that can be configured (excludes 'local')
	type CloudBackendType = Exclude<BackendType, 'local'>;

	const PROVIDERS: { type: CloudBackendType; label: string; description: string }[] = [
		{ type: 'openai', label: 'OpenAI', description: 'GPT-4o, GPT-4, GPT-3.5 and other models' },
		{
			type: 'anthropic',
			label: 'Anthropic',
			description: 'Claude 4, Claude 3.5 Sonnet and other models'
		},
		{
			type: 'together',
			label: 'Together AI',
			description: 'Llama, Mistral, and other open-source models'
		},
		{ type: 'groq', label: 'Groq', description: 'Ultra-fast inference for Llama and Mixtral' },
		{
			type: 'fireworks',
			label: 'Fireworks AI',
			description: 'Fast inference for various open models'
		},
		{
			type: 'mistral',
			label: 'Mistral AI',
			description: 'Mistral and Mixtral models from Mistral AI'
		},
		{ type: 'deepseek', label: 'DeepSeek', description: 'DeepSeek Coder and Chat models' },
		{
			type: 'openai_compatible',
			label: 'Custom (OpenAI-compatible)',
			description: 'Any API using the OpenAI-compatible format'
		},
		{
			type: 'anthropic_compatible',
			label: 'Custom (Anthropic-compatible)',
			description: 'Any API using the Anthropic-compatible format'
		}
	];

	let credentials = $state<CloudCredential[]>([]);
	let connectionStatus = $state<Record<CloudBackendType, 'connected' | 'error' | 'unconfigured'>>({
		openai: 'unconfigured',
		anthropic: 'unconfigured',
		together: 'unconfigured',
		groq: 'unconfigured',
		fireworks: 'unconfigured',
		mistral: 'unconfigured',
		deepseek: 'unconfigured',
		openai_compatible: 'unconfigured',
		anthropic_compatible: 'unconfigured'
	});
	let loading = $state(true);

	async function loadProviders() {
		loading = true;
		try {
			credentials = await settings.listProviders();

			// Reset status before testing
			connectionStatus = {
				openai: 'unconfigured',
				anthropic: 'unconfigured',
				together: 'unconfigured',
				groq: 'unconfigured',
				fireworks: 'unconfigured',
				mistral: 'unconfigured',
				deepseek: 'unconfigured',
				openai_compatible: 'unconfigured',
				anthropic_compatible: 'unconfigured'
			};

			// Test connection for each configured provider
			for (const cred of credentials) {
				if (cred.backend_type !== 'local') {
					try {
						await settings.testProvider(cred.backend_type);
						connectionStatus[cred.backend_type as CloudBackendType] = 'connected';
					} catch {
						connectionStatus[cred.backend_type as CloudBackendType] = 'error';
					}
				}
			}
		} catch (e) {
			console.error('Failed to load providers:', e);
		} finally {
			loading = false;
		}
	}

	onMount(loadProviders);

	function getCredential(type: CloudBackendType): CloudCredential | null {
		return credentials.find((c) => c.backend_type === type) ?? null;
	}

	function getStatusColor(type: CloudBackendType): string {
		switch (connectionStatus[type]) {
			case 'connected':
				return 'bg-green-500';
			case 'error':
				return 'bg-red-500';
			case 'unconfigured':
			default:
				return 'bg-gray-400';
		}
	}

	function getStatusText(type: CloudBackendType): string {
		switch (connectionStatus[type]) {
			case 'connected':
				return 'Connected';
			case 'error':
				return 'Error';
			case 'unconfigured':
			default:
				return 'Not configured';
		}
	}
</script>

<Card class="p-0 overflow-hidden">
	{#if loading}
		<div class="p-6 text-sm text-muted-foreground">Loading providers...</div>
	{:else}
		<Accordion.Root type="single">
			{#each PROVIDERS as provider (provider.type)}
				{@const credential = getCredential(provider.type)}
				<Accordion.Item value={provider.type} class="border-b last:border-b-0">
					<Accordion.Trigger
						class="flex w-full items-center justify-between p-4 text-left hover:bg-muted/50 transition-colors group"
					>
						<div class="flex items-center gap-3">
							<!-- Status dot -->
							<span
								class="h-2.5 w-2.5 rounded-full {getStatusColor(provider.type)}"
								title={getStatusText(provider.type)}
							></span>
							<div>
								<span class="font-medium">{provider.label}</span>
								<p class="text-xs text-muted-foreground">{provider.description}</p>
							</div>
						</div>
						<div class="flex items-center gap-2">
							<span class="text-xs text-muted-foreground">
								{getStatusText(provider.type)}
							</span>
							<ChevronDown
								class="h-4 w-4 text-muted-foreground transition-transform duration-200 group-data-[state=open]:rotate-180"
							/>
						</div>
					</Accordion.Trigger>
					<Accordion.Content
						class="overflow-hidden data-[state=closed]:animate-accordion-up data-[state=open]:animate-accordion-down"
					>
						<div class="p-4 pt-0 border-t bg-muted/20">
							<ProviderForm
								backendType={provider.type}
								existingCredential={credential}
								onSave={loadProviders}
								onDelete={loadProviders}
							/>
						</div>
					</Accordion.Content>
				</Accordion.Item>
			{/each}
		</Accordion.Root>
	{/if}
</Card>
