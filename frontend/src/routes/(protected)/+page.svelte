<script lang="ts">
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { onMount } from 'svelte';

	// Redirect to servers page
	onMount(async () => {
		await goto(resolve('/servers'));
	});
</script>

<div class="flex items-center justify-center min-h-[60vh]">
	<p class="text-muted-foreground">Redirecting to servers...</p>
</div>
