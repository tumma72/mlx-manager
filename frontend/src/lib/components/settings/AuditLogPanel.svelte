<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { Download, RefreshCw, Filter, Wifi, WifiOff } from 'lucide-svelte';
	import { Button } from '$components/ui';
	import { auditLogs } from '$lib/api/client';
	import type { AuditLog, AuditLogFilter, AuditStats } from '$lib/api/types';

	// State
	let logs = $state<AuditLog[]>([]);
	let stats = $state<AuditStats | null>(null);
	let loading = $state(false);
	let wsConnected = $state(false);
	let ws: WebSocket | null = null;

	// Filters
	let filterModel = $state('');
	let filterBackend = $state('');
	let filterStatus = $state('');
	let offset = $state(0);
	const limit = 50;

	// Load logs with current filters
	async function loadLogs(append = false) {
		loading = true;
		try {
			const filter: AuditLogFilter = {
				limit,
				offset: append ? offset : 0
			};
			if (filterModel) filter.model = filterModel;
			if (filterBackend) filter.backend_type = filterBackend;
			if (filterStatus) filter.status = filterStatus;

			const newLogs = await auditLogs.list(filter);

			if (append) {
				logs = [...logs, ...newLogs];
			} else {
				logs = newLogs;
				offset = 0;
			}
		} catch (e) {
			console.error('Failed to load logs:', e);
		} finally {
			loading = false;
		}
	}

	// Load stats
	async function loadStats() {
		try {
			stats = await auditLogs.stats();
		} catch (e) {
			console.error('Failed to load stats:', e);
		}
	}

	// Connect WebSocket for live updates
	function connectWebSocket() {
		try {
			ws = auditLogs.createWebSocket();

			ws.onopen = () => {
				wsConnected = true;
			};

			ws.onmessage = (event) => {
				const msg = JSON.parse(event.data);
				if (msg.type === 'log') {
					// Prepend new log to list
					logs = [msg.data, ...logs.slice(0, 99)];
					// Update stats when new log arrives
					loadStats();
				}
			};

			ws.onclose = () => {
				wsConnected = false;
				// Reconnect after delay
				setTimeout(connectWebSocket, 5000);
			};

			ws.onerror = () => {
				wsConnected = false;
			};
		} catch (e) {
			console.error('WebSocket error:', e);
		}
	}

	// Load more (infinite scroll)
	function loadMore() {
		offset += limit;
		loadLogs(true);
	}

	// Export logs
	function handleExport(format: 'jsonl' | 'csv') {
		const filter: AuditLogFilter = {};
		if (filterModel) filter.model = filterModel;
		if (filterBackend) filter.backend_type = filterBackend;
		if (filterStatus) filter.status = filterStatus;

		const url = auditLogs.exportUrl(filter, format);
		window.open(url, '_blank');
	}

	// Format duration
	function formatDuration(ms: number): string {
		if (ms < 1000) return `${ms}ms`;
		if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
		return `${(ms / 60000).toFixed(1)}m`;
	}

	// Format timestamp
	function formatTime(ts: string): string {
		return new Date(ts).toLocaleString();
	}

	// Status badge color
	function statusColor(status: string): string {
		switch (status) {
			case 'success':
				return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
			case 'error':
				return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
			case 'timeout':
				return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
			default:
				return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200';
		}
	}

	onMount(() => {
		loadLogs();
		loadStats();
		connectWebSocket();
	});

	onDestroy(() => {
		if (ws) {
			ws.close();
		}
	});
</script>

<div class="space-y-4">
	<!-- Stats row -->
	{#if stats}
		<div class="grid grid-cols-4 gap-4">
			<div class="rounded-lg bg-gray-50 p-3 dark:bg-gray-800">
				<div class="text-2xl font-bold">{stats.total_requests}</div>
				<div class="text-sm text-muted-foreground">Total Requests</div>
			</div>
			<div class="rounded-lg bg-gray-50 p-3 dark:bg-gray-800">
				<div class="text-2xl font-bold text-green-600">{stats.by_status.success || 0}</div>
				<div class="text-sm text-muted-foreground">Successful</div>
			</div>
			<div class="rounded-lg bg-gray-50 p-3 dark:bg-gray-800">
				<div class="text-2xl font-bold text-red-600">{stats.by_status.error || 0}</div>
				<div class="text-sm text-muted-foreground">Errors</div>
			</div>
			<div class="rounded-lg bg-gray-50 p-3 dark:bg-gray-800">
				<div class="text-2xl font-bold">{stats.unique_models}</div>
				<div class="text-sm text-muted-foreground">Models Used</div>
			</div>
		</div>
	{/if}

	<!-- Filters and controls -->
	<div class="flex flex-wrap items-center gap-4">
		<div class="flex items-center gap-2">
			<Filter class="h-4 w-4 text-muted-foreground" />
			<input
				type="text"
				placeholder="Model"
				class="w-40 rounded border px-2 py-1 text-sm"
				bind:value={filterModel}
				onchange={() => loadLogs()}
			/>
			<select
				class="rounded border px-2 py-1 text-sm"
				bind:value={filterBackend}
				onchange={() => loadLogs()}
			>
				<option value="">All Backends</option>
				<option value="local">Local</option>
				<option value="openai">OpenAI</option>
				<option value="anthropic">Anthropic</option>
			</select>
			<select
				class="rounded border px-2 py-1 text-sm"
				bind:value={filterStatus}
				onchange={() => loadLogs()}
			>
				<option value="">All Status</option>
				<option value="success">Success</option>
				<option value="error">Error</option>
				<option value="timeout">Timeout</option>
			</select>
		</div>

		<div class="ml-auto flex items-center gap-2">
			<!-- WebSocket status -->
			<span class="flex items-center gap-1 text-sm text-muted-foreground">
				{#if wsConnected}
					<Wifi class="h-4 w-4 text-green-500" />
					<span>Live</span>
				{:else}
					<WifiOff class="h-4 w-4 text-gray-400" />
					<span>Offline</span>
				{/if}
			</span>

			<Button variant="outline" size="sm" onclick={() => loadLogs()}>
				<RefreshCw class="mr-1 h-4 w-4" />
				Refresh
			</Button>
			<Button variant="outline" size="sm" onclick={() => handleExport('jsonl')}>
				<Download class="mr-1 h-4 w-4" />
				JSONL
			</Button>
			<Button variant="outline" size="sm" onclick={() => handleExport('csv')}>
				<Download class="mr-1 h-4 w-4" />
				CSV
			</Button>
		</div>
	</div>

	<!-- Log table -->
	<div class="overflow-hidden rounded-lg border">
		<table class="w-full text-sm">
			<thead class="bg-gray-50 dark:bg-gray-800">
				<tr>
					<th class="px-4 py-2 text-left">Time</th>
					<th class="px-4 py-2 text-left">Model</th>
					<th class="px-4 py-2 text-left">Backend</th>
					<th class="px-4 py-2 text-left">Duration</th>
					<th class="px-4 py-2 text-left">Status</th>
					<th class="px-4 py-2 text-left">Tokens</th>
				</tr>
			</thead>
			<tbody class="divide-y">
				{#each logs as log (log.id)}
					<tr class="hover:bg-gray-50 dark:hover:bg-gray-800">
						<td class="whitespace-nowrap px-4 py-2 text-muted-foreground">
							{formatTime(log.timestamp)}
						</td>
						<td class="max-w-[200px] truncate px-4 py-2 font-mono text-xs" title={log.model}>
							{log.model}
						</td>
						<td class="px-4 py-2 capitalize">{log.backend_type}</td>
						<td class="px-4 py-2">{formatDuration(log.duration_ms)}</td>
						<td class="px-4 py-2">
							<span class={`rounded px-2 py-0.5 text-xs ${statusColor(log.status)}`}>
								{log.status}
							</span>
						</td>
						<td class="px-4 py-2">
							{#if log.total_tokens}
								{log.total_tokens}
							{:else}
								<span class="text-muted-foreground">-</span>
							{/if}
						</td>
					</tr>
				{:else}
					<tr>
						<td colspan="6" class="px-4 py-8 text-center text-muted-foreground">
							{#if loading}
								Loading...
							{:else}
								No audit logs found
							{/if}
						</td>
					</tr>
				{/each}
			</tbody>
		</table>
	</div>

	<!-- Load more -->
	{#if logs.length >= limit}
		<div class="text-center">
			<Button variant="ghost" size="sm" onclick={loadMore} disabled={loading}>
				{#if loading}
					Loading...
				{:else}
					Load More
				{/if}
			</Button>
		</div>
	{/if}
</div>
