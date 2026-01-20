<script lang="ts">
	import '../../app.css';
	let { children } = $props();
</script>

<div class="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
	{@render children()}
</div>
