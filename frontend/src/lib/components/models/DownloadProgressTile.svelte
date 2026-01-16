<script lang="ts">
	import type { DownloadState } from '$lib/stores/downloads.svelte';
	import { formatBytes } from '$lib/utils/format';
	import { Card } from '$components/ui';
	import { Download, CheckCircle, XCircle, Loader2 } from 'lucide-svelte';

	interface Props {
		download: DownloadState;
	}

	let { download }: Props = $props();

	// Extract short name (after /) for display
	const shortName = $derived(() => {
		const parts = download.model_id.split('/');
		return parts[parts.length - 1];
	});

	// Get status display
	const statusDisplay = $derived(() => {
		switch (download.status) {
			case 'pending':
			case 'starting':
				return 'Preparing...';
			case 'downloading':
				return download.progress > 0 ? `${download.progress}%` : 'Starting...';
			case 'completed':
				return 'Complete';
			case 'failed':
				return 'Failed';
			default:
				return download.status;
		}
	});

	// Get status icon component
	const isSpinning = $derived(download.status === 'pending' || download.status === 'starting');
	const isComplete = $derived(download.status === 'completed');
	const isFailed = $derived(download.status === 'failed');
</script>

<Card class="p-3">
	<div class="flex items-center gap-3">
		<!-- Status icon -->
		<div class="flex-shrink-0">
			{#if isComplete}
				<CheckCircle class="w-5 h-5 text-green-500 dark:text-green-400" />
			{:else if isFailed}
				<XCircle class="w-5 h-5 text-red-500 dark:text-red-400" />
			{:else if isSpinning}
				<Loader2 class="w-5 h-5 text-primary animate-spin" />
			{:else}
				<Download class="w-5 h-5 text-primary" />
			{/if}
		</div>

		<!-- Content -->
		<div class="flex-1 min-w-0">
			<!-- Model name with full path on hover -->
			<div class="flex items-center justify-between gap-2">
				<span
					class="font-medium text-sm truncate"
					title={download.model_id}
				>
					{shortName()}
				</span>
				<span class="text-xs text-muted-foreground flex-shrink-0">
					{statusDisplay()}
				</span>
			</div>

			<!-- Progress bar (only show when downloading) -->
			{#if download.status === 'downloading' || download.status === 'starting' || download.status === 'pending'}
				<div class="mt-1.5 space-y-1">
					<div class="w-full bg-muted rounded-full h-1.5 overflow-hidden">
						<div
							class="bg-primary h-1.5 rounded-full transition-all duration-300"
							style="width: {download.progress}%"
						></div>
					</div>
					{#if download.total_bytes > 0}
						<div class="text-xs text-muted-foreground">
							{formatBytes(download.downloaded_bytes)} / {formatBytes(download.total_bytes)}
						</div>
					{/if}
				</div>
			{/if}

			<!-- Error message -->
			{#if download.error}
				<div class="mt-1 text-xs text-red-500 dark:text-red-400 truncate" title={download.error}>
					{download.error}
				</div>
			{/if}
		</div>
	</div>
</Card>
