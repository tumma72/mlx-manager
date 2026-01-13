<script lang="ts">
	import { marked } from 'marked';

	interface Props {
		content: string;
		class?: string;
	}

	let { content, class: className = '' }: Props = $props();

	// Configure marked for safe rendering
	marked.setOptions({
		breaks: true,
		gfm: true
	});

	const html = $derived(marked.parse(content) as string);
</script>

<div class="prose prose-sm dark:prose-invert max-w-none {className}">
	<!-- eslint-disable-next-line svelte/no-at-html-tags -- @html is intentional for rendering markdown -->
	{@html html}
</div>
