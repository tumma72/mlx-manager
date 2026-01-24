<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/stores';
	import { resolve } from '$app/paths';
	import { serverStore, profileStore, authStore } from '$stores';
	import { Card, Button, Input, Select, Markdown, ThinkingBubble, ErrorMessage } from '$components/ui';
	import { Send, Loader2, Bot, User, Paperclip, X, AlertCircle } from 'lucide-svelte';
	import type { Attachment, ContentPart } from '$lib/api/types';

	interface Message {
		role: 'user' | 'assistant';
		content: string | ContentPart[];
	}

	let messages = $state<Message[]>([]);
	let input = $state('');
	let loading = $state(false);
	let chatError = $state<{ summary: string; details?: string } | null>(null);
	let errorMessageRef = $state<{ collapse: () => void } | undefined>(undefined);
	let selectedProfileId = $state<number | null>(null);
	let attachments = $state<Attachment[]>([]);
	let dragOver = $state(false);
	let fileInputRef = $state<HTMLInputElement | undefined>(undefined);
	let streamingThinking = $state('');
	let streamingResponse = $state('');
	let thinkingDuration = $state<number | undefined>(undefined);
	let retryAttempt = $state(0);
	let retryMax = 3;
	let isRetrying = $state(false);
	let lastFailedMessage = $state<{ content: string | ContentPart[]; attachments: Attachment[] } | null>(null);

	// Get running profiles - use servers directly to determine running state
	// This avoids issues with serverStore.isRunning() which can return false
	// for profiles that are in the servers list but still in startingProfiles
	const runningProfiles = $derived.by(() => {
		const runningIds = new Set(serverStore.servers.map((s) => s.profile_id));
		return profileStore.profiles.filter((p) => runningIds.has(p.id));
	});

	const selectedProfile = $derived(runningProfiles.find((p) => p.id === selectedProfileId));

	// Determine if current profile supports multimodal
	const isMultimodal = $derived(
		selectedProfile?.model_type === 'multimodal'
	);

	// Accept string based on model capabilities - CHANGE 2
	const acceptedFormats = $derived(
		isMultimodal
			? 'image/*,video/*,.txt,.md,.csv,.json,.xml,.yaml,.yml,.log,.py,.js,.ts,.html,.css,.sh,.sql,.conf,.ini,.toml'
			: '.txt,.md,.csv,.json,.xml,.yaml,.yml,.log,.py,.js,.ts,.html,.css,.sh,.sql,.conf,.ini,.toml'
	);

	// Handle URL query parameter reactively (not just on mount)
	// This ensures we find the profile even if stores aren't loaded yet
	let urlProfileId = $state<number | null>(null);

	onMount(() => {
		// Initial data load - polling is handled globally by +layout.svelte
		profileStore.refresh();
		serverStore.refresh();

		// Get profile from URL query param
		const profileParam = $page.url.searchParams.get('profile');
		if (profileParam) {
			urlProfileId = parseInt(profileParam, 10);
		}
	});

	// Reactively set selectedProfileId when stores load and URL param is present
	$effect(() => {
		if (urlProfileId !== null && selectedProfileId === null) {
			// Check if this profile is actually running
			const isRunning = runningProfiles.some((p) => p.id === urlProfileId);
			if (isRunning) {
				selectedProfileId = urlProfileId;
			}
		}
	});

	// Validate video duration (max 2 minutes)
	async function validateVideoDuration(file: File): Promise<boolean> {
		return new Promise((resolve) => {
			const video = document.createElement('video');
			video.preload = 'metadata';
			video.onloadedmetadata = () => {
				URL.revokeObjectURL(video.src);
				resolve(video.duration <= 120);
			};
			video.onerror = () => {
				URL.revokeObjectURL(video.src);
				resolve(false);
			};
			video.src = URL.createObjectURL(file);
		});
	}

	// Add attachment with validation - CHANGE 3 & 4
	async function addAttachment(file: File): Promise<void> {
		if (attachments.length >= 3) {
			chatError = { summary: 'Maximum 3 attachments per message' };
			return;
		}

		const isVideo = file.type.startsWith('video/');
		const isImage = file.type.startsWith('image/');
		const isText = file.type.startsWith('text/') ||
			file.type === 'application/json' ||
			file.type === 'application/xml' ||
			file.type === 'application/x-yaml' ||
			file.type === 'application/x-sh' ||
			file.type === 'application/sql';

		// Validate based on model type
		if (!isText && !isVideo && !isImage) {
			chatError = { summary: 'Unsupported file type' };
			return;
		}

		// Text-only models: reject media files
		if (!isMultimodal && (isImage || isVideo)) {
			chatError = { summary: 'This model only supports text file attachments' };
			return;
		}

		// Multimodal models: accept everything
		// Text models: accept only text files

		if (isVideo) {
			const valid = await validateVideoDuration(file);
			if (!valid) {
				chatError = { summary: 'Video must be 2 minutes or less' };
				return;
			}
		}

		const preview = isText ? file.name : URL.createObjectURL(file);
		attachments.push({
			file,
			preview,
			type: isText ? 'text' : isVideo ? 'video' : 'image'
		});
	}

	// Remove attachment and cleanup object URL - CHANGE 5
	function removeAttachment(index: number): void {
		const attachment = attachments[index];
		// Only revoke object URLs (not filenames for text files)
		if (attachment.type !== 'text') {
			URL.revokeObjectURL(attachment.preview);
		}
		attachments.splice(index, 1);
	}

	// Handle file input change
	async function handleFileSelect(e: Event): Promise<void> {
		const input = e.target as HTMLInputElement;
		const files = Array.from(input.files || []);
		for (const file of files) {
			await addAttachment(file);
		}
		input.value = ''; // Reset for same file re-selection
	}

	// Handle drag-drop
	function handleDrop(e: DragEvent): void {
		e.preventDefault();
		dragOver = false;
		const files = Array.from(e.dataTransfer?.files || []);
		for (const file of files) {
			addAttachment(file);
		}
	}

	function handleDragOver(e: DragEvent): void {
		e.preventDefault();
		dragOver = true;
	}

	function handleDragLeave(e: DragEvent): void {
		e.preventDefault();
		dragOver = false;
	}

	async function encodeFileAsBase64(file: File): Promise<string> {
		return new Promise((resolve, reject) => {
			const reader = new FileReader();
			reader.onload = () => resolve(reader.result as string);
			reader.onerror = reject;
			reader.readAsDataURL(file);
		});
	}

	async function readFileAsText(file: File): Promise<string> {
		return new Promise((resolve, reject) => {
			const reader = new FileReader();
			reader.onload = () => resolve(reader.result as string);
			reader.onerror = reject;
			reader.readAsText(file);
		});
	}

	async function buildMessageContent(text: string, currentAttachments: Attachment[]): Promise<string | ContentPart[]> {
		if (currentAttachments.length === 0) {
			return text;
		}

		const parts: ContentPart[] = [{ type: 'text', text }];

		for (const attachment of currentAttachments) {
			if (attachment.type === 'text') {
				// Read text files as plain text and include inline
				const fileContent = await readFileAsText(attachment.file);
				parts.push({
					type: 'text',
					text: `[File: ${attachment.file.name}]\n${fileContent}`
				});
			} else {
				// Images and videos: encode as base64 data URL
				const base64 = await encodeFileAsBase64(attachment.file);
				parts.push({
					type: 'image_url',
					image_url: { url: base64 }
				});
			}
		}

		return parts;
	}

	async function sendWithRetry(userContent: string | ContentPart[], userAttachments: Attachment[], attempt: number = 1): Promise<boolean> {
		retryAttempt = attempt;
		isRetrying = attempt > 1;

		if (!selectedProfile) return false;

		// Build messages array for API (use full content with images)
		const apiMessages: Array<{ role: 'system' | 'user' | 'assistant'; content: string | ContentPart[] }> = [];

		// Add system prompt as first message if present
		if (selectedProfile.system_prompt) {
			apiMessages.push({ role: 'system', content: selectedProfile.system_prompt });
		}

		// Add conversation history (exclude last user message since we're about to add it)
		messages.slice(0, -1).forEach(m => {
			apiMessages.push({ role: m.role, content: m.content });
		});

		// Add current user message
		apiMessages.push({ role: 'user', content: userContent });

		try {
			const response = await fetch('/api/chat/completions', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					'Authorization': `Bearer ${authStore.token}`,
				},
				body: JSON.stringify({
					profile_id: selectedProfile.id,
					messages: apiMessages,
				}),
			});

			if (!response.ok) {
				// Only retry on server errors (5xx) or 502, 503, 504
				const isServerError = response.status >= 500 && response.status < 600;
				const isGatewayError = [502, 503, 504].includes(response.status);

				if ((isServerError || isGatewayError) && attempt < retryMax) {
					// Linear backoff: 2s, 4s, 6s
					const waitMs = attempt * 2000;
					await new Promise(resolve => setTimeout(resolve, waitMs));
					return sendWithRetry(userContent, userAttachments, attempt + 1);
				}

				throw new Error(`Server error: ${response.status}`);
			}

			const reader = response.body?.getReader();
			if (!reader) throw new Error('No response body');

			const decoder = new TextDecoder();
			let buffer = '';

			while (true) {
				const { done, value } = await reader.read();
				if (done) break;

				buffer += decoder.decode(value, { stream: true });
				const lines = buffer.split('\n');
				buffer = lines.pop() || ''; // Keep incomplete line in buffer

				for (const line of lines) {
					if (!line.startsWith('data: ')) continue;

					try {
						const data = JSON.parse(line.slice(6));

						switch (data.type) {
							case 'thinking':
								streamingThinking += data.content;
								break;
							case 'thinking_done':
								thinkingDuration = data.duration;
								break;
							case 'response':
								streamingResponse += data.content;
								break;
							case 'error':
								// Parse connection errors vs server errors
								if (data.content.includes('Failed to connect')) {
									chatError = {
										summary: 'Connection failed',
										details: data.content
									};
								} else if (data.content.includes('timed out')) {
									chatError = {
										summary: 'Request timed out',
										details: data.content
									};
								} else {
									chatError = {
										summary: 'Server error',
										details: data.content
									};
								}
								break;
							case 'done':
								// Finalize message
								if (streamingResponse || streamingThinking) {
									const finalContent = streamingThinking
										? `<think>${streamingThinking}</think>${streamingResponse}`
										: streamingResponse;
									messages.push({ role: 'assistant', content: finalContent });
								}
								break;
						}
					} catch {
						// Ignore parse errors
					}
				}
			}

			// On success, reset retry state
			retryAttempt = 0;
			isRetrying = false;
			lastFailedMessage = null;
			return true;

		} catch (error) {
			// Check for connection/network errors (TypeError for network issues)
			const isNetworkError = error instanceof TypeError;
			const errorMsg = error instanceof Error ? error.message : 'Failed to send message';
			const isConnectionError = errorMsg.toLowerCase().includes('failed to fetch') ||
									  errorMsg.toLowerCase().includes('network') ||
									  errorMsg.toLowerCase().includes('connection');

			if ((isNetworkError || isConnectionError) && attempt < retryMax) {
				// Linear backoff: 2s, 4s, 6s
				const waitMs = attempt * 2000;
				await new Promise(resolve => setTimeout(resolve, waitMs));
				return sendWithRetry(userContent, userAttachments, attempt + 1);
			}

			// All retries exhausted
			retryAttempt = 0;
			isRetrying = false;
			lastFailedMessage = { content: userContent, attachments: userAttachments };

			// Show error in chat area
			chatError = {
				summary: 'Failed to connect to model after 3 attempts',
				details: `The model may still be loading. Check the Servers page for status.\n\nError: ${errorMsg}`
			};

			return false;
		}
	}

	async function handleSubmit(e: Event) {
		e.preventDefault();
		if (!input.trim() || !selectedProfile || loading) return;

		// Collapse previous error
		errorMessageRef?.collapse();
		chatError = null;

		const userMessage = input.trim();
		input = '';

		// Build message content (with attachments if any)
		const content = await buildMessageContent(userMessage, attachments);

		// Store attachments for potential retry
		const currentAttachments = [...attachments];

		// Clear attachments - CHANGE 8
		for (const attachment of attachments) {
			if (attachment.type !== 'text') {
				URL.revokeObjectURL(attachment.preview);
			}
		}
		attachments = [];

		// Add user message to UI (display text only for user messages)
		messages.push({ role: 'user', content: userMessage });

		// Reset streaming state
		streamingThinking = '';
		streamingResponse = '';
		thinkingDuration = undefined;
		loading = true;

		try {
			const success = await sendWithRetry(content, currentAttachments, 1);
			if (!success) {
				// Remove the user message on failure
				messages.pop();
			}
		} finally {
			loading = false;
		}
	}

	function handleClear() {
		messages = [];
		chatError = null;
	}

	function handleProfileChange(e: Event) {
		const target = e.target as HTMLSelectElement;
		selectedProfileId = target.value ? parseInt(target.value, 10) : null;
		messages = [];
		chatError = null;
		// Clear attachments when switching profiles - CHANGE 7
		for (const attachment of attachments) {
			if (attachment.type !== 'text') {
				URL.revokeObjectURL(attachment.preview);
			}
		}
		attachments = [];
	}

	// Extract model name without owner prefix (e.g., "mlx-community/Qwen3-0.6B-4bit" -> "Qwen3-0.6B-4bit")
	function getModelShortName(modelPath: string): string {
		const parts = modelPath.split('/');
		return parts.length > 1 ? parts[parts.length - 1] : modelPath;
	}

	// Parse thinking content from assistant messages
	// Supports multiple tag formats used by different model families:
	// - <think>...</think> (Qwen3 style)
	// - <thinking>...</thinking> (alternative format)
	// - <reasoning>...</reasoning> (reasoning models)
	function parseThinking(content: string | ContentPart[]): { thinking: string | null; response: string } {
		// Content in stored messages is always string (ContentPart[] only used in API calls)
		if (typeof content !== 'string') {
			return { thinking: null, response: '' };
		}
		// Try each pattern in order of likelihood
		const patterns = [
			/<think>([\s\S]*?)<\/think>/,         // Qwen3 style
			/<thinking>([\s\S]*?)<\/thinking>/,   // Alternative format
			/<reasoning>([\s\S]*?)<\/reasoning>/, // Reasoning format
		];

		for (const pattern of patterns) {
			const match = content.match(pattern);
			if (match) {
				const thinking = match[1].trim();
				const response = content.replace(pattern, '').trim();
				return { thinking, response };
			}
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
		<div
			class="flex-1 overflow-hidden"
			role="region"
			aria-label="Chat message area with drag-drop file upload"
			ondragover={handleDragOver}
			ondragleave={handleDragLeave}
			ondrop={handleDrop}
		>
		<Card class="h-full overflow-hidden flex flex-col {dragOver ? 'ring-2 ring-primary' : ''}">
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
					<!-- System Prompt (pinned first message) -->
					{#if selectedProfile?.system_prompt}
						<div class="flex gap-3 opacity-60">
							<div class="flex-shrink-0 w-8 h-8 rounded-full bg-muted flex items-center justify-center">
								<Bot class="w-4 h-4 text-muted-foreground" />
							</div>
							<div class="flex-1 min-w-0">
								<p class="text-xs text-muted-foreground mb-1 italic">System Prompt</p>
								<div class="text-sm text-muted-foreground italic whitespace-pre-wrap">
									{selectedProfile.system_prompt}
								</div>
							</div>
						</div>
					{:else if selectedProfile && !selectedProfile.system_prompt}
						<div class="flex items-center gap-2 text-xs text-muted-foreground bg-muted/50 rounded-lg px-3 py-2" id="system-prompt-hint">
							<AlertCircle class="w-3 h-3" />
							<span>No system prompt set.</span>
							<a href={resolve(`/profiles/${selectedProfile.id}`)} class="text-primary hover:underline">Set one in profile settings</a>
							<button
								class="ml-auto text-muted-foreground hover:text-foreground"
								onclick={(e) => { (e.currentTarget.closest('#system-prompt-hint') as HTMLElement)?.remove() }}
								type="button"
							>
								<X class="w-3 h-3" />
							</button>
						</div>
					{/if}
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
									<p class="whitespace-pre-wrap">{typeof message.content === 'string' ? message.content : ''}</p>
								{/if}
							</div>
							{#if message.role === 'user'}
								<div class="w-8 h-8 rounded-full bg-primary flex items-center justify-center flex-shrink-0">
									<User class="w-5 h-5 text-primary-foreground" />
								</div>
							{/if}
						</div>
					{/each}

					{#if isRetrying}
						<div class="flex items-center gap-2 text-sm text-muted-foreground animate-pulse px-4 py-2">
							<Loader2 class="w-4 h-4 animate-spin" />
							<span>Connecting to model... (attempt {retryAttempt}/{retryMax})</span>
						</div>
					{/if}

					{#if loading}
						<div class="flex gap-3">
							<div class="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
								<Bot class="w-5 h-5 text-primary" />
							</div>
							<div class="max-w-[80%] rounded-lg px-4 py-2 bg-muted">
								{#if streamingThinking}
									<ThinkingBubble
										content={streamingThinking}
										duration={thinkingDuration}
										streaming={thinkingDuration === undefined}
									/>
								{/if}
								{#if streamingResponse}
									<Markdown content={streamingResponse} />
								{:else if !streamingThinking}
									<Loader2 class="w-5 h-5 animate-spin" />
								{/if}
							</div>
						</div>
					{/if}

					{#if chatError}
						<div class="flex gap-3">
							<div class="w-8 h-8 rounded-full bg-destructive/10 flex items-center justify-center flex-shrink-0">
								<AlertCircle class="w-5 h-5 text-destructive" />
							</div>
							<div class="max-w-[80%]">
								<ErrorMessage
									bind:this={errorMessageRef}
									summary={chatError.summary}
									details={chatError.details}
								/>
							</div>
						</div>
					{/if}

					{#if lastFailedMessage}
						<div class="flex justify-center py-2">
							<Button
								variant="outline"
								size="sm"
								onclick={async () => {
									if (lastFailedMessage && selectedProfile) {
										errorMessageRef?.collapse();
										chatError = null;
										// Re-add user message to UI
										const displayContent = typeof lastFailedMessage.content === 'string'
											? lastFailedMessage.content
											: lastFailedMessage.content.find(p => p.type === 'text')?.text || '';
										messages.push({ role: 'user', content: displayContent });
										// Reset streaming state
										streamingThinking = '';
										streamingResponse = '';
										thinkingDuration = undefined;
										loading = true;
										try {
											const success = await sendWithRetry(lastFailedMessage.content, lastFailedMessage.attachments, 1);
											if (!success) {
												messages.pop();
											}
										} finally {
											loading = false;
										}
									}
								}}
								disabled={loading}
							>
								Retry
							</Button>
						</div>
					{/if}
				{/if}
			</div>

			<!-- Input -->
			<form onsubmit={handleSubmit} class="p-4 border-t">
				{#if attachments.length > 0}
					<div class="px-4 pt-4 flex gap-2 flex-wrap">
						{#each attachments as attachment, i (attachment.preview)}
							<div class="relative group">
								{#if attachment.type === 'image'}
									<img
										src={attachment.preview}
										alt="Attachment"
										class="w-16 h-16 object-cover rounded-lg border"
									/>
								{:else if attachment.type === 'video'}
									<video
										src={attachment.preview}
										class="w-16 h-16 object-cover rounded-lg border"
										muted
									>
										<track kind="captions" />
									</video>
								{:else if attachment.type === 'text'}
									<div class="w-16 h-16 flex flex-col items-center justify-center rounded-lg border bg-muted text-muted-foreground text-xs p-2">
										<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="mb-1">
											<path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"></path>
											<polyline points="14 2 14 8 20 8"></polyline>
										</svg>
										<span class="truncate w-full text-center">{attachment.preview.split('.').pop()}</span>
									</div>
								{/if}
								<button
									type="button"
									onclick={() => removeAttachment(i)}
									class="absolute -top-2 -right-2 w-5 h-5 bg-destructive text-destructive-foreground rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
								>
									<X class="w-3 h-3" />
								</button>
							</div>
						{/each}
					</div>
				{/if}
				<div class="flex gap-2">
					<Button
						type="button"
						variant="ghost"
						size="icon"
						onclick={() => fileInputRef?.click()}
						disabled={loading || attachments.length >= 3}
					>
						<Paperclip class="w-4 h-4" />
					</Button>
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
				<input
					type="file"
					bind:this={fileInputRef}
					accept={acceptedFormats}
					multiple
					onchange={handleFileSelect}
					class="hidden"
				/>
			</form>
		</Card>
		</div>
	{/if}
</div>
