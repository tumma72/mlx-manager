<script lang="ts">
	import { cn } from '$lib/utils';

	interface Props {
		value?: string;
		class?: string;
		id?: string;
		name?: string;
		disabled?: boolean;
		onchange?: (e: Event) => void;
		children?: import('svelte').Snippet;
	}

	let {
		value = $bindable(''),
		class: className = '',
		id,
		name,
		disabled = false,
		onchange,
		children
	}: Props = $props();
</script>

<select
	bind:value
	{id}
	{name}
	{disabled}
	{onchange}
	class={cn(
		'flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50',
		className
	)}
>
	{@render children?.()}
</select>
