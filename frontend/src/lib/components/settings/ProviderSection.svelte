<script lang="ts">
	import { onMount } from 'svelte';
	import { Accordion } from 'bits-ui';
	import { ChevronDown } from 'lucide-svelte';
	import { settings } from '$lib/api/client';
	import type { CloudCredential, BackendType } from '$lib/api/types';
	import ProviderForm from './ProviderForm.svelte';
	import { Card } from '$components/ui';

	const PROVIDERS: { type: BackendType; label: string; description: string }[] = [
		{ type: 'openai', label: 'OpenAI', description: 'GPT-4o, GPT-4, GPT-3.5 and other models' },
		{
			type: 'anthropic',
			label: 'Anthropic',
			description: 'Claude 4, Claude 3.5 Sonnet and other models'
		}
	];

	let credentials = $state<CloudCredential[]>([]);
	let connectionStatus = $state<Record<BackendType, 'connected' | 'error' | 'unconfigured'>>({
		local: 'connected', // Always "connected" for local
		openai: 'unconfigured',
		anthropic: 'unconfigured'
	});
	let loading = $state(true);

	async function loadProviders() {
		loading = true;
		try {
			credentials = await settings.listProviders();

			// Reset status before testing
			connectionStatus = {
				local: 'connected',
				openai: 'unconfigured',
				anthropic: 'unconfigured'
			};

			// Test connection for each configured provider
			for (const cred of credentials) {
				try {
					await settings.testProvider(cred.backend_type);
					connectionStatus[cred.backend_type] = 'connected';
				} catch {
					connectionStatus[cred.backend_type] = 'error';
				}
			}
		} catch (e) {
			console.error('Failed to load providers:', e);
		} finally {
			loading = false;
		}
	}

	onMount(loadProviders);

	function getCredential(type: BackendType): CloudCredential | null {
		return credentials.find((c) => c.backend_type === type) ?? null;
	}

	function getStatusColor(type: BackendType): string {
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

	function getStatusText(type: BackendType): string {
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
