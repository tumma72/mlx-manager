<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/stores';
	import { serverStore, profileStore } from '$stores';
	import { Card, Button, Input, Select, Markdown, ThinkingBubble } from '$components/ui';
	import { Send, Loader2, Bot, User } from 'lucide-svelte';

	interface Message {
		role: 'user' | 'assistant';
		content: string;
	}

	let messages = $state<Message[]>([]);
	let input = $state('');
	let loading = $state(false);
	let error = $state<string | null>(null);
	let selectedProfileId = $state<number | null>(null);

	// Get running profiles
	const runningProfiles = $derived(
		profileStore.profiles.filter((p) => serverStore.isRunning(p.id))
	);

	const selectedProfile = $derived(
		runningProfiles.find((p) => p.id === selectedProfileId)
	);

	onMount(() => {
		// Initial data load - polling is handled globally by +layout.svelte
		profileStore.refresh();
		serverStore.refresh();

		// Check for profile query param
		const profileParam = $page.url.searchParams.get('profile');
		if (profileParam) {
			selectedProfileId = parseInt(profileParam, 10);
		}
	});

	async function handleSubmit(e: Event) {
		e.preventDefault();
		if (!input.trim() || !selectedProfile || loading) return;

		const userMessage = input.trim();
		input = '';
		error = null;

		// Add user message
		messages = [...messages, { role: 'user', content: userMessage }];

		loading = true;
		try {
			// Call the OpenAI-compatible API
			const response = await fetch(`http://${selectedProfile.host}:${selectedProfile.port}/v1/chat/completions`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
				},
				body: JSON.stringify({
					model: selectedProfile.model_path,
					messages: messages.map(m => ({ role: m.role, content: m.content })),
					stream: false,
				}),
			});

			if (!response.ok) {
				const errorData = await response.text();
				throw new Error(`HTTP ${response.status}: ${errorData}`);
			}

			const data = await response.json();
			const assistantMessage = data.choices?.[0]?.message?.content || 'No response';

			messages = [...messages, { role: 'assistant', content: assistantMessage }];
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to send message';
			// Remove the user message on error
			messages = messages.slice(0, -1);
		} finally {
			loading = false;
		}
	}

	function handleClear() {
		messages = [];
		error = null;
	}

	function handleProfileChange(e: Event) {
		const target = e.target as HTMLSelectElement;
		selectedProfileId = target.value ? parseInt(target.value, 10) : null;
		messages = [];
		error = null;
	}

	// Extract model name without owner prefix (e.g., "mlx-community/Qwen3-0.6B-4bit" -> "Qwen3-0.6B-4bit")
	function getModelShortName(modelPath: string): string {
		const parts = modelPath.split('/');
		return parts.length > 1 ? parts[parts.length - 1] : modelPath;
	}

	// Parse thinking content from assistant messages (supports <think>...</think> tags)
	function parseThinking(content: string): { thinking: string | null; response: string } {
		const thinkMatch = content.match(/<think>([\s\S]*?)<\/think>/);
		if (thinkMatch) {
			const thinking = thinkMatch[1].trim();
			const response = content.replace(/<think>[\s\S]*?<\/think>/, '').trim();
			return { thinking, response };
		}
		return { thinking: null, response: content };
	}
</script>

<div class="space-y-6 h-[calc(100vh-8rem)] flex flex-col">
	<div class="flex items-center justify-between">
		<h1 class="text-2xl font-bold">Chat</h1>
		<div class="flex items-center gap-4">
			<Select onchange={handleProfileChange} value={selectedProfileId?.toString() ?? ''}>
				<option value="">Select a running server...</option>
				{#each runningProfiles as profile (profile.id)}
					<option value={profile.id.toString()}>
						{profile.name} ({getModelShortName(profile.model_path)})
					</option>
				{/each}
			</Select>
			{#if messages.length > 0}
				<Button variant="outline" onclick={handleClear}>
					Clear Chat
				</Button>
			{/if}
		</div>
	</div>

	{#if runningProfiles.length === 0}
		<Card class="flex-1 flex items-center justify-center">
			<div class="text-center">
				<Bot class="w-16 h-16 mx-auto text-muted-foreground mb-4" />
				<p class="text-muted-foreground mb-4">No servers running.</p>
				<Button href="/servers">Go to Servers</Button>
			</div>
		</Card>
	{:else if !selectedProfile}
		<Card class="flex-1 flex items-center justify-center">
			<div class="text-center">
				<Bot class="w-16 h-16 mx-auto text-muted-foreground mb-4" />
				<p class="text-muted-foreground">Select a running server to start chatting.</p>
			</div>
		</Card>
	{:else}
		<!-- Chat Messages -->
		<Card class="flex-1 overflow-hidden flex flex-col">
			<div class="flex-1 overflow-y-auto p-4 space-y-4">
				{#if messages.length === 0}
					<div class="h-full flex items-center justify-center">
						<div class="text-center text-muted-foreground">
							<Bot class="w-12 h-12 mx-auto mb-4" />
							<p>Start a conversation with <strong>{selectedProfile.name}</strong></p>
							<p class="text-sm mt-1">Model: {selectedProfile.model_path}</p>
						</div>
					</div>
				{:else}
					{#each messages as message (message)}
						<div class="flex gap-3 {message.role === 'user' ? 'justify-end' : ''}">
							{#if message.role === 'assistant'}
								<div class="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
									<Bot class="w-5 h-5 text-primary" />
								</div>
							{/if}
							<div
								class="max-w-[80%] rounded-lg px-4 py-2 {message.role === 'user'
									? 'bg-primary text-primary-foreground'
									: 'bg-muted'}"
							>
								{#if message.role === 'assistant'}
									{@const parsed = parseThinking(message.content)}
									{#if parsed.thinking}
										<ThinkingBubble content={parsed.thinking} />
									{/if}
									<Markdown content={parsed.response} />
								{:else}
									<p class="whitespace-pre-wrap">{message.content}</p>
								{/if}
							</div>
							{#if message.role === 'user'}
								<div class="w-8 h-8 rounded-full bg-primary flex items-center justify-center flex-shrink-0">
									<User class="w-5 h-5 text-primary-foreground" />
								</div>
							{/if}
						</div>
					{/each}

					{#if loading}
						<div class="flex gap-3">
							<div class="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
								<Bot class="w-5 h-5 text-primary" />
							</div>
							<div class="bg-muted rounded-lg px-4 py-2">
								<Loader2 class="w-5 h-5 animate-spin" />
							</div>
						</div>
					{/if}
				{/if}
			</div>

			{#if error}
				<div class="px-4 py-2 bg-red-100 text-red-700 text-sm">
					{error}
				</div>
			{/if}

			<!-- Input -->
			<form onsubmit={handleSubmit} class="p-4 border-t">
				<div class="flex gap-2">
					<Input
						bind:value={input}
						placeholder="Type a message..."
						disabled={loading}
						class="flex-1"
					/>
					<Button type="submit" disabled={loading || !input.trim()}>
						{#if loading}
							<Loader2 class="w-4 h-4 animate-spin" />
						{:else}
							<Send class="w-4 h-4" />
						{/if}
					</Button>
				</div>
			</form>
		</Card>
	{/if}
</div>
