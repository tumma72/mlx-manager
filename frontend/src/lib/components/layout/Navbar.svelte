<script lang="ts">
	import { page } from '$app/stores';
	import { resolve } from '$app/paths';
	import { systemStore } from '$stores';
	import { Server, Package, Settings, MessageSquare, Cpu } from 'lucide-svelte';

	const navigation = [
		{ href: '/servers' as const, label: 'Servers', icon: Server },
		{ href: '/chat' as const, label: 'Chat', icon: MessageSquare },
		{ href: '/models' as const, label: 'Models', icon: Package },
		{ href: '/profiles' as const, label: 'Profiles', icon: Settings }
	];

	// Memory polling is handled globally by +layout.svelte
	// Just trigger initial load if not already loaded
	$effect(() => {
		if (!systemStore.memory) {
			systemStore.refreshMemory();
		}
	});
</script>

<nav class="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
	<div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
		<div class="flex justify-between h-16">
			<div class="flex items-center">
				<a href={resolve('/')} class="flex items-center gap-2">
					<Cpu class="w-8 h-8 text-mlx-500" />
					<span class="font-bold text-xl">MLX Manager</span>
				</a>

				<div class="hidden md:flex ml-10 space-x-4">
					{#each navigation as item (item.href)}
						{@const isActive = $page.url.pathname.startsWith(item.href)}
						<a
							href={resolve(item.href)}
							class="flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors
								{isActive
								? 'bg-mlx-100 text-mlx-700 dark:bg-mlx-900 dark:text-mlx-100'
								: 'text-gray-600 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-700'}"
						>
							<item.icon class="w-4 h-4" />
							{item.label}
						</a>
					{/each}
				</div>
			</div>

			<div class="flex items-center gap-4">
				{#if systemStore.memory}
					<div class="hidden sm:flex items-center gap-2 text-sm text-muted-foreground">
						<span>Memory:</span>
						<span class="font-mono">
							{systemStore.memory.used_gb.toFixed(1)} / {systemStore.memory.total_gb.toFixed(0)} GB
						</span>
						<div class="w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
							<div
								class="h-full bg-mlx-500 transition-all"
								style="width: {systemStore.memory.percent_used}%"
							></div>
						</div>
					</div>
				{/if}
			</div>
		</div>
	</div>
</nav>
