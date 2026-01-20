<script lang="ts">
	import '../../app.css';
	import { onMount } from 'svelte';
	import { Navbar } from '$components/layout';
	import { serverStore, profileStore, systemStore, downloadsStore, pollingCoordinator } from '$stores';

	let { children } = $props();

	onMount(() => {
		// Initialize global polling for all stores
		serverStore.startPolling();
		profileStore.startPolling();
		systemStore.startMemoryPolling();

		// Load any active downloads and reconnect SSE streams
		downloadsStore.loadActiveDownloads();

		// Cleanup on unmount (e.g., hot reload)
		return () => {
			pollingCoordinator.destroy();
			downloadsStore.cleanup();
		};
	});
</script>

<div class="min-h-screen bg-gray-50 dark:bg-gray-900">
	<Navbar />
	<main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
		{@render children()}
	</main>
</div>
