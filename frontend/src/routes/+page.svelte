<script lang="ts">
	import { goto } from '$app/navigation';
	import { onMount } from 'svelte';

	// Redirect to servers page
	onMount(() => {
		goto('/servers');
	});
</script>

<div class="flex items-center justify-center min-h-[60vh]">
	<p class="text-muted-foreground">Redirecting to servers...</p>
</div>
