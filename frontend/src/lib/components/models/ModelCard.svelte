<script lang="ts">
	import type { ModelSearchResult } from '$api';
	import { models } from '$api';
	import { formatNumber, formatBytes } from '$lib/utils/format';
	import { Card, Button, Badge, ConfirmDialog } from '$components/ui';
	import { Download, Trash2, Check, HardDrive, Heart, ArrowDownToLine } from 'lucide-svelte';
	import { downloadsStore } from '$lib/stores';

	interface Props {
		model: ModelSearchResult;
		onUse?: (modelId: string) => void;
		onDeleted?: () => void;
	}

	let { model, onUse, onDeleted }: Props = $props();

	let deleting = $state(false);
	let error = $state<string | null>(null);
	let showDeleteConfirm = $state(false);
	// Track local override for download status (null means use prop value)
	let downloadStatusOverride = $state<boolean | null>(null);

	// Get download state from global store
	let downloadState = $derived(downloadsStore.getProgress(model.model_id));
	let downloading = $derived(downloadsStore.isDownloading(model.model_id));

	// Determine if downloaded: check store for completed, then override, then prop
	let isDownloaded = $derived(() => {
		if (downloadState?.status === 'completed') return true;
		if (downloadStatusOverride !== null) return downloadStatusOverride;
		return model.is_downloaded;
	});

	// Reset override when model prop changes
	$effect(() => {
		void model.model_id; // Track model changes
		downloadStatusOverride = null;
	});

	async function handleDownload() {
		error = null;
		try {
			await downloadsStore.startDownload(model.model_id);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Download failed';
		}
	}

	// Watch for download completion to update override
	$effect(() => {
		if (downloadState?.status === 'completed') {
			downloadStatusOverride = true;
		} else if (downloadState?.status === 'failed' && downloadState.error) {
			error = downloadState.error;
		}
	});

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
			{#if isDownloaded()}
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

	{#if downloading && downloadState}
		<!-- Download progress bar -->
		<div class="mt-4 space-y-2">
			<div class="flex justify-between text-xs text-muted-foreground">
				<span>
					{formatBytes(downloadState.downloaded_bytes)} / {formatBytes(downloadState.total_bytes)}
				</span>
				<span>
					{downloadState.progress > 0 ? `${downloadState.progress}%` : 'Starting...'}
				</span>
			</div>
			<div class="w-full bg-muted rounded-full h-2 overflow-hidden">
				<div
					class="bg-primary h-2 rounded-full transition-all duration-300"
					style="width: {downloadState.progress}%"
				></div>
			</div>
			<div class="text-center text-xs text-muted-foreground">
				{downloadState.status === 'starting' || downloadState.status === 'pending'
					? 'Preparing download...'
					: `${downloadState.progress}% complete`}
			</div>
		</div>
	{:else}
		<div class="mt-4 flex justify-end gap-2">
			{#if isDownloaded()}
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
		<div class="mt-2 text-sm text-red-500 dark:text-red-400">{error}</div>
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
