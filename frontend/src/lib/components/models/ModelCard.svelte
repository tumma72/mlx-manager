<script lang="ts">
	import type { ModelSearchResult } from '$api';
	import { models } from '$api';
	import { formatNumber, formatBytes } from '$lib/utils/format';
	import { Card, Button, Badge, ConfirmDialog } from '$components/ui';
	import { Download, Trash2, Check, HardDrive, Heart, ArrowDownToLine } from 'lucide-svelte';

	interface Props {
		model: ModelSearchResult;
		onUse?: (modelId: string) => void;
		onDeleted?: () => void;
	}

	let { model, onUse, onDeleted }: Props = $props();

	let downloading = $state(false);
	let deleting = $state(false);
	let error = $state<string | null>(null);
	let showDeleteConfirm = $state(false);
	// Track local override for download status (null means use prop value)
	let downloadStatusOverride = $state<boolean | null>(null);
	let isDownloaded = $derived(downloadStatusOverride ?? model.is_downloaded);

	// Download progress tracking
	let downloadProgress = $state(0);
	let downloadedBytes = $state(0);
	let totalBytes = $state(0);
	let downloadSpeed = $state(0);

	// Reset override when model prop changes
	$effect(() => {
		void model.model_id; // Track model changes
		downloadStatusOverride = null;
	});

	async function handleDownload() {
		downloading = true;
		error = null;
		downloadProgress = 0;
		downloadedBytes = 0;
		totalBytes = 0;
		downloadSpeed = 0;

		try {
			const { task_id } = await models.startDownload(model.model_id);

			// Connect to SSE for progress
			const eventSource = new EventSource(`/api/models/download/${task_id}/progress`);

			eventSource.onmessage = (event) => {
				const data = JSON.parse(event.data);

				if (data.status === 'starting') {
					totalBytes = data.total_bytes || 0;
				} else if (data.status === 'downloading') {
					downloadProgress = data.progress || 0;
					downloadedBytes = data.downloaded_bytes || 0;
					totalBytes = data.total_bytes || totalBytes;
					downloadSpeed = data.speed_mbps || 0;
				} else if (data.status === 'completed') {
					downloadProgress = 100;
					downloadStatusOverride = true;
					downloading = false;
					eventSource.close();
				} else if (data.status === 'failed') {
					error = data.error || 'Download failed';
					downloading = false;
					eventSource.close();
				}
			};

			eventSource.onerror = () => {
				error = 'Connection lost';
				downloading = false;
				eventSource.close();
			};
		} catch (e) {
			error = e instanceof Error ? e.message : 'Download failed';
			downloading = false;
		}
	}

	function requestDelete() {
		showDeleteConfirm = true;
	}

	async function confirmDelete() {
		deleting = true;
		error = null;
		try {
			await models.delete(model.model_id);
			downloadStatusOverride = false;
			onDeleted?.();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Delete failed';
		} finally {
			deleting = false;
		}
	}

	function handleUse() {
		onUse?.(model.model_id);
	}
</script>

<Card class="p-4">
	<div class="flex items-start justify-between">
		<div class="flex-1 min-w-0">
			<h3 class="font-semibold text-base truncate">{model.model_id}</h3>
			<div class="flex items-center gap-4 mt-1 text-sm text-muted-foreground">
				<span class="flex items-center gap-1">
					<ArrowDownToLine class="w-4 h-4" />
					{formatNumber(model.downloads)}
				</span>
				<span class="flex items-center gap-1">
					<Heart class="w-4 h-4" />
					{formatNumber(model.likes)}
				</span>
				<span class="flex items-center gap-1">
					<HardDrive class="w-4 h-4" />
					~{model.estimated_size_gb} GB
				</span>
			</div>
		</div>

		<div class="flex items-center gap-2">
			{#if isDownloaded}
				<Badge variant="success">
					<Check class="w-3 h-3 mr-1" />
					Downloaded
				</Badge>
			{/if}
		</div>
	</div>

	{#if model.tags.length > 0}
		<div class="mt-3 flex flex-wrap gap-1">
			{#each model.tags.slice(0, 5) as tag (tag)}
				<Badge variant="secondary" class="text-xs">{tag}</Badge>
			{/each}
			{#if model.tags.length > 5}
				<Badge variant="outline" class="text-xs">+{model.tags.length - 5}</Badge>
			{/if}
		</div>
	{/if}

	{#if downloading}
		<!-- Download progress bar -->
		<div class="mt-4 space-y-2">
			<div class="flex justify-between text-xs text-muted-foreground">
				<span>
					{formatBytes(downloadedBytes)} / {formatBytes(totalBytes)}
				</span>
				<span>
					{downloadSpeed > 0 ? `${downloadSpeed.toFixed(1)} MB/s` : 'Starting...'}
				</span>
			</div>
			<div class="w-full bg-muted rounded-full h-2 overflow-hidden">
				<div
					class="bg-primary h-2 rounded-full transition-all duration-300"
					style="width: {downloadProgress}%"
				></div>
			</div>
			<div class="text-center text-xs text-muted-foreground">
				{downloadProgress}% complete
			</div>
		</div>
	{:else}
		<div class="mt-4 flex justify-end gap-2">
			{#if isDownloaded}
				<Button variant="outline" size="sm" onclick={requestDelete} disabled={deleting}>
					<Trash2 class="w-4 h-4 mr-1" />
					{deleting ? 'Deleting...' : 'Delete'}
				</Button>
				<Button variant="default" size="sm" onclick={handleUse}>
					Use
				</Button>
			{:else}
				<Button variant="default" size="sm" onclick={handleDownload} disabled={downloading}>
					<Download class="w-4 h-4 mr-1" />
					Download
				</Button>
			{/if}
		</div>
	{/if}

	{#if error}
		<div class="mt-2 text-sm text-red-500">{error}</div>
	{/if}
</Card>

<ConfirmDialog
	bind:open={showDeleteConfirm}
	title="Delete Model"
	description="Are you sure you want to delete {model.model_id}? This will remove the model from your local cache and free up disk space."
	confirmLabel="Delete"
	cancelLabel="Cancel"
	variant="destructive"
	onConfirm={confirmDelete}
	onCancel={() => {}}
/>
