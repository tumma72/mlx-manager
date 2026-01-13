<script lang="ts" module>
	export type BadgeVariant = 'default' | 'secondary' | 'destructive' | 'outline' | 'success' | 'warning';
</script>

<script lang="ts">
	import { cn } from '$lib/utils';

	interface Props {
		variant?: BadgeVariant;
		class?: string;
		children?: import('svelte').Snippet;
	}

	const variants = {
		default: 'bg-primary text-primary-foreground hover:bg-primary/80',
		secondary: 'bg-secondary text-secondary-foreground hover:bg-secondary/80',
		destructive: 'bg-destructive text-destructive-foreground hover:bg-destructive/80',
		outline: 'text-foreground border',
		success: 'bg-green-500 text-white hover:bg-green-600',
		warning: 'bg-yellow-500 text-white hover:bg-yellow-600'
	};

	let { variant = 'default', class: className = '', children }: Props = $props();
</script>

<div
	class={cn(
		'inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
		variants[variant],
		className
	)}
>
	{@render children?.()}
</div>
