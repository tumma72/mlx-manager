<script lang="ts">
	import type { DownloadState } from '$lib/stores/downloads.svelte';
	import { downloadsStore } from '$lib/stores/downloads.svelte';
	import { formatBytes } from '$lib/utils/format';
	import { Card } from '$components/ui';
	import {
		Download,
		CheckCircle,
		XCircle,
		Loader2,
		Pause,
		Play,
		X
	} from 'lucide-svelte';

	interface Props {
		download: DownloadState;
	}

	let { download }: Props = $props();

	let isLoading = $state(false);
	let showCancelConfirm = $state(false);

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
			case 'paused':
				return 'Paused';
			case 'completed':
				return 'Complete';
			case 'failed':
				return 'Failed';
			case 'cancelled':
				return 'Cancelled';
			default:
				return download.status;
		}
	});

	// State checks
	const isSpinning = $derived(download.status === 'pending' || download.status === 'starting');
	const isComplete = $derived(download.status === 'completed');
	const isFailed = $derived(download.status === 'failed');
	const isPaused = $derived(download.status === 'paused');
	const isDownloading = $derived(download.status === 'downloading');
	const isActive = $derived(isDownloading || download.status === 'starting' || download.status === 'pending');
	const canPause = $derived(isDownloading && download.download_id != null);
	const canResume = $derived(isPaused && download.download_id != null);
	const canCancel = $derived((isActive || isPaused) && download.download_id != null);

	async function handlePause() {
		isLoading = true;
		try {
			await downloadsStore.pauseDownload(download.model_id);
		} finally {
			isLoading = false;
		}
	}

	async function handleResume() {
		isLoading = true;
		try {
			await downloadsStore.resumeDownload(download.model_id);
		} finally {
			isLoading = false;
		}
	}

	async function handleCancel() {
		isLoading = true;
		showCancelConfirm = false;
		try {
			await downloadsStore.cancelDownload(download.model_id);
		} finally {
			isLoading = false;
		}
	}
</script>

<Card class="p-3">
	<div class="flex items-center gap-3">
		<!-- Status icon -->
		<div class="flex-shrink-0">
			{#if isComplete}
				<CheckCircle class="w-5 h-5 text-green-500 dark:text-green-400" />
			{:else if isFailed}
				<XCircle class="w-5 h-5 text-red-500 dark:text-red-400" />
			{:else if isPaused}
				<Pause class="w-5 h-5 text-amber-500 dark:text-amber-400" />
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
				<span class="font-medium text-sm truncate" title={download.model_id}>
					{shortName()}
				</span>
				<span
					class="text-xs flex-shrink-0"
					class:text-muted-foreground={!isPaused}
					class:text-amber-500={isPaused}
					class:dark:text-amber-400={isPaused}
				>
					{statusDisplay()}
				</span>
			</div>

			<!-- Progress bar -->
			{#if isActive || isPaused}
				<div class="mt-1.5 space-y-1">
					<div class="w-full bg-muted rounded-full h-1.5 overflow-hidden">
						<div
							class="h-1.5 rounded-full transition-all duration-300"
							class:bg-primary={!isPaused}
							class:bg-amber-500={isPaused}
							class:dark:bg-amber-400={isPaused}
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
				<div
					class="mt-1 text-xs text-red-500 dark:text-red-400 truncate"
					title={download.error}
				>
					{download.error}
				</div>
			{/if}

			<!-- Action buttons -->
			{#if canPause || canResume || canCancel}
				<div class="mt-2 flex items-center gap-1.5">
					{#if showCancelConfirm}
						<!-- Cancel confirmation -->
						<span class="text-xs text-muted-foreground mr-1">Cancel download?</span>
						<button
							class="inline-flex items-center gap-1 px-2 py-0.5 text-xs rounded
								bg-destructive text-destructive-foreground hover:bg-destructive/90
								disabled:opacity-50 disabled:pointer-events-none"
							disabled={isLoading}
							onclick={handleCancel}
						>
							{#if isLoading}
								<Loader2 class="w-3 h-3 animate-spin" />
							{/if}
							Confirm
						</button>
						<button
							class="inline-flex items-center gap-1 px-2 py-0.5 text-xs rounded
								bg-muted text-muted-foreground hover:bg-muted/80"
							onclick={() => (showCancelConfirm = false)}
						>
							Keep
						</button>
					{:else}
						{#if canPause}
							<button
								class="inline-flex items-center gap-1 px-2 py-0.5 text-xs rounded
									bg-muted text-muted-foreground hover:bg-muted/80
									disabled:opacity-50 disabled:pointer-events-none"
								disabled={isLoading}
								onclick={handlePause}
								title="Pause download"
							>
								{#if isLoading}
									<Loader2 class="w-3 h-3 animate-spin" />
								{:else}
									<Pause class="w-3 h-3" />
								{/if}
								Pause
							</button>
						{/if}

						{#if canResume}
							<button
								class="inline-flex items-center gap-1 px-2 py-0.5 text-xs rounded
									bg-primary text-primary-foreground hover:bg-primary/90
									disabled:opacity-50 disabled:pointer-events-none"
								disabled={isLoading}
								onclick={handleResume}
								title="Resume download"
							>
								{#if isLoading}
									<Loader2 class="w-3 h-3 animate-spin" />
								{:else}
									<Play class="w-3 h-3" />
								{/if}
								Resume
							</button>
						{/if}

						{#if canCancel}
							<button
								class="inline-flex items-center gap-1 px-2 py-0.5 text-xs rounded
									text-destructive hover:bg-destructive/10
									disabled:opacity-50 disabled:pointer-events-none"
								disabled={isLoading}
								onclick={() => (showCancelConfirm = true)}
								title="Cancel download"
							>
								<X class="w-3 h-3" />
								Cancel
							</button>
						{/if}
					{/if}
				</div>
			{/if}
		</div>
	</div>
</Card>
