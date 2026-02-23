<script lang="ts">
	import { Tabs } from 'bits-ui';
	import {
		ProviderSection,
		ModelPoolSettings,
		RoutingRulesSection,
		TimeoutSettings,
		AuditLogPanel
	} from '$lib/components/settings';

	let activeTab = $state('providers');
</script>

<svelte:head>
	<title>Settings | MLX Manager</title>
</svelte:head>

<div class="space-y-6">
	<h1 class="text-2xl font-bold text-gray-900 dark:text-gray-100">Settings</h1>

	<Tabs.Root bind:value={activeTab}>
		<Tabs.List
			class="inline-flex h-10 items-center justify-center rounded-md bg-muted p-1 text-muted-foreground"
		>
			<Tabs.Trigger
				value="providers"
				class="inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 data-[state=active]:bg-background data-[state=active]:text-foreground data-[state=active]:shadow-sm"
			>
				Providers
			</Tabs.Trigger>
			<Tabs.Trigger
				value="pool"
				class="inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 data-[state=active]:bg-background data-[state=active]:text-foreground data-[state=active]:shadow-sm"
			>
				Profiles Pool
			</Tabs.Trigger>
			<Tabs.Trigger
				value="routing"
				class="inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 data-[state=active]:bg-background data-[state=active]:text-foreground data-[state=active]:shadow-sm"
			>
				Routing
			</Tabs.Trigger>
			<Tabs.Trigger
				value="timeouts"
				class="inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 data-[state=active]:bg-background data-[state=active]:text-foreground data-[state=active]:shadow-sm"
			>
				Timeouts
			</Tabs.Trigger>
			<Tabs.Trigger
				value="logs"
				class="inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 data-[state=active]:bg-background data-[state=active]:text-foreground data-[state=active]:shadow-sm"
			>
				Logs
			</Tabs.Trigger>
		</Tabs.List>

		<!-- Cloud Providers Tab -->
		<Tabs.Content value="providers" class="mt-6 space-y-4">
			<div>
				<h2 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">Cloud Providers</h2>
				<p class="text-sm text-muted-foreground mb-4">
					Configure API keys for cloud inference providers. Keys are encrypted at rest.
				</p>
			</div>
			<ProviderSection />
		</Tabs.Content>

		<!-- Profiles Pool Tab -->
		<Tabs.Content value="pool" class="mt-6 space-y-4">
			<div>
				<h2 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">Profiles Pool</h2>
				<p class="text-sm text-muted-foreground mb-4">
					Configure memory limits and caching behavior for the execution profiles pool.
				</p>
			</div>
			<ModelPoolSettings />
		</Tabs.Content>

		<!-- Routing Rules Tab -->
		<Tabs.Content value="routing" class="mt-6 space-y-4">
			<div>
				<h2 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
					Model Routing Rules
				</h2>
				<p class="text-sm text-muted-foreground mb-4">
					Configure model-to-backend routing rules with pattern matching. Rules are evaluated by
					priority (lower number = higher priority).
				</p>
			</div>
			<RoutingRulesSection />
		</Tabs.Content>

		<!-- Timeouts Tab -->
		<Tabs.Content value="timeouts" class="mt-6 space-y-4">
			<div>
				<h2 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">Request Timeouts</h2>
				<p class="text-sm text-muted-foreground mb-4">
					Configure per-endpoint timeouts for inference requests. Longer timeouts allow for larger
					models and longer outputs.
				</p>
			</div>
			<TimeoutSettings />
		</Tabs.Content>

		<!-- Request Logs Tab -->
		<Tabs.Content value="logs" class="mt-6 space-y-4">
			<div>
				<h2 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">Request Logs</h2>
				<p class="text-sm text-muted-foreground mb-4">
					View and export inference request logs. Live updates when connected.
				</p>
			</div>
			<AuditLogPanel />
		</Tabs.Content>
	</Tabs.Root>
</div>
