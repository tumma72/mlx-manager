<script lang="ts">
	interface Props {
		mode: 'local' | 'online';
		onModeChange?: (mode: 'local' | 'online') => void;
	}

	let { mode = $bindable(), onModeChange }: Props = $props();

	function setMode(newMode: 'local' | 'online') {
		mode = newMode;
		onModeChange?.(newMode);
	}
</script>

<div class="flex rounded-full bg-muted p-1">
	<button
		type="button"
		class="px-4 py-1.5 text-sm font-medium rounded-full transition-all
			{mode === 'local'
			? 'bg-background shadow-sm text-foreground'
			: 'text-muted-foreground hover:text-foreground'}"
		onclick={() => setMode('local')}
	>
		My Models
	</button>
	<button
		type="button"
		class="px-4 py-1.5 text-sm font-medium rounded-full transition-all
			{mode === 'online'
			? 'bg-background shadow-sm text-foreground'
			: 'text-muted-foreground hover:text-foreground'}"
		onclick={() => setMode('online')}
	>
		HuggingFace
	</button>
</div>
