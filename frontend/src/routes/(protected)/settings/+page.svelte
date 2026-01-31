<script lang="ts">
	import {
		ProviderSection,
		ModelPoolSettings,
		RoutingRulesSection,
		TimeoutSettings,
		AuditLogPanel
	} from '$lib/components/settings';
</script>

<svelte:head>
	<title>Settings | MLX Manager</title>
</svelte:head>

<div class="space-y-8">
	<h1 class="text-2xl font-bold text-gray-900 dark:text-gray-100">Settings</h1>

	<!-- Cloud Providers -->
	<section>
		<h2 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">Cloud Providers</h2>
		<p class="text-sm text-muted-foreground mb-4">
			Configure API keys for cloud inference providers. Keys are encrypted at rest.
		</p>
		<ProviderSection />
	</section>

	<hr class="border-gray-200 dark:border-gray-700" />

	<!-- Model Pool -->
	<section>
		<h2 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">Model Pool</h2>
		<p class="text-sm text-muted-foreground mb-4">
			Configure memory limits and caching behavior for the local model pool.
		</p>
		<ModelPoolSettings />
	</section>

	<hr class="border-gray-200 dark:border-gray-700" />

	<!-- Routing Rules -->
	<section>
		<h2 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">Model Routing Rules</h2>
		<p class="text-sm text-muted-foreground mb-4">
			Configure model-to-backend routing rules with pattern matching. Rules are evaluated by priority (lower number = higher priority).
		</p>
		<RoutingRulesSection />
	</section>

	<hr class="border-gray-200 dark:border-gray-700" />

	<!-- Timeouts -->
	<section>
		<h2 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">Request Timeouts</h2>
		<p class="text-sm text-muted-foreground mb-4">
			Configure per-endpoint timeouts for inference requests. Longer timeouts allow for larger models and longer outputs.
		</p>
		<TimeoutSettings />
	</section>

	<hr class="border-gray-200 dark:border-gray-700" />

	<!-- Audit Logs -->
	<section>
		<h2 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">Request Logs</h2>
		<p class="text-sm text-muted-foreground mb-4">
			View and export inference request logs. Live updates when connected.
		</p>
		<AuditLogPanel />
	</section>
</div>
