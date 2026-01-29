<script lang="ts">
	import { ProviderSection, ModelPoolSettings } from '$lib/components/settings';
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

	<!-- Model Pool -->
	<section>
		<h2 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">Model Pool</h2>
		<p class="text-sm text-muted-foreground mb-4">
			Configure memory limits and caching behavior for the local model pool.
		</p>
		<ModelPoolSettings />
	</section>

	<!-- Routing Rules (placeholder for Plan 04) -->
	<section>
		<h2 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">Model Routing Rules</h2>
		<p class="text-sm text-muted-foreground">
			Configure model-to-backend routing rules with pattern matching. Coming in the next update.
		</p>
	</section>
</div>
