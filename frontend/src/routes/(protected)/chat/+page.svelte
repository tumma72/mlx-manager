<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/stores';
	import { resolve } from '$app/paths';
	import { serverStore, profileStore, authStore } from '$stores';
	import { Card, Button, Select, Markdown, ThinkingBubble, ErrorMessage, ToolCallBubble } from '$components/ui';
	import { Send, Loader2, Bot, User, Paperclip, X, AlertCircle, Wrench, Copy, Square } from 'lucide-svelte';
	import { mcp } from '$lib/api/client';
	import type { Attachment, ContentPart, ToolDefinition } from '$lib/api/types';
	import { isChatCapable } from '$lib/utils';

	const TEXT_EXTENSIONS = new Set([
		'txt', 'md', 'csv', 'json', 'xml', 'yaml', 'yml',
		'log', 'py', 'js', 'ts', 'html', 'css', 'sh',
		'sql', 'conf', 'ini', 'toml'
	]);

	// Known text filenames without extensions
	const KNOWN_TEXT_FILENAMES = new Set([
		// Standard extensionless files
		'readme', 'makefile', 'dockerfile', 'license', 'procfile',
		'gemfile', 'brewfile', 'vagrantfile', 'rakefile', 'guardfile',
		'podfile', 'fastfile', 'dangerfile', 'berksfile', 'thorfile',
		'capfile', 'puppetfile', 'appraisals', 'codeowners',
		'changelog', 'contributing', 'authors', 'todo', 'copying',
		'notice', 'patents', 'version', 'manifest',
		// Common dotfiles (also extensionless)
		'.gitignore', '.dockerignore', '.editorconfig', '.env.example',
		'.npmrc', '.nvmrc', '.prettierrc', '.eslintrc', '.babelrc',
		'.gitattributes', '.mailmap', '.env', '.env.local', '.env.development',
		'.env.production', '.env.test'
	]);

	interface ToolCallData {
		id: string;
		name: string;
		arguments: string;
		result?: string;
		error?: string;
	}

	interface Message {
		role: 'user' | 'assistant';
		content: string | ContentPart[];
		toolCalls?: ToolCallData[];
		thinkingDuration?: number;
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
	let streamingToolCalls = $state<ToolCallData[]>([]);
	let retryAttempt = $state(0);
	let retryMax = 3;
	let isRetrying = $state(false);
	let lastFailedMessage = $state<{ content: string | ContentPart[]; attachments: Attachment[] } | null>(null);
	let textareaRef = $state<HTMLTextAreaElement | null>(null);
	let toolsEnabled = $state(false);
	let availableTools = $state<ToolDefinition[]>([]);
	let toolsLoaded = $state(false);
	let copyFeedback = $state(false);
	let abortController = $state<AbortController | null>(null);
	let protocol = $state<'openai' | 'anthropic'>('openai');

	// With embedded server, all profiles can be used for chat
	// The model will be loaded on-demand by the embedded MLX Server
	const availableProfiles = $derived(
		profileStore.profiles.filter(p => isChatCapable(p.model_type) && serverStore.isRunning(p.id))
	);

	const selectedProfile = $derived(availableProfiles.find((p) => p.id === selectedProfileId));

	// Determine if current profile supports multimodal
	const isMultimodal = $derived(
		selectedProfile?.model_type === 'vision'
	);

	// Accept string based on model capabilities and protocol
	const acceptedFormats = $derived(
		isMultimodal
			? (protocol === 'anthropic'
				? 'image/*,.txt,.md,.csv,.json,.xml,.yaml,.yml,.log,.py,.js,.ts,.html,.css,.sh,.sql,.conf,.ini,.toml'
				: 'image/*,video/*,.txt,.md,.csv,.json,.xml,.yaml,.yml,.log,.py,.js,.ts,.html,.css,.sh,.sql,.conf,.ini,.toml')
			: '.txt,.md,.csv,.json,.xml,.yaml,.yml,.log,.py,.js,.ts,.html,.css,.sh,.sql,.conf,.ini,.toml'
	);

	// Handle URL query parameter reactively (not just on mount)
	// This ensures we find the profile even if stores aren't loaded yet
	let urlProfileId = $state<number | null>(null);

	onMount(async () => {
		// Initial data load - polling is handled globally by +layout.svelte
		profileStore.refresh();
		serverStore.refresh();

		// Get profile from URL query param
		const profileParam = $page.url.searchParams.get('profile');
		if (profileParam) {
			urlProfileId = parseInt(profileParam, 10);
		}

		// Load available MCP tools
		try {
			availableTools = await mcp.listTools();
			toolsLoaded = true;
		} catch {
			// MCP tools not available - toggle won't show
		}
	});

	// Reactively set selectedProfileId when stores load and URL param is present
	$effect(() => {
		if (urlProfileId !== null && selectedProfileId === null) {
			// With embedded server, any profile can be used (model loads on-demand)
			const profileExists = availableProfiles.some((p) => p.id === urlProfileId);
			if (profileExists) {
				selectedProfileId = urlProfileId;
			}
		}
	});

	// Reset textarea height when input becomes empty
	$effect(() => {
		if (!input && textareaRef) {
			textareaRef.style.height = 'auto';
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

	// Add attachment with validation
	async function addAttachment(file: File): Promise<void> {
		if (attachments.length >= 3) {
			chatError = { summary: 'Maximum 3 attachments per message' };
			return;
		}

		const isVideo = file.type.startsWith('video/');
		const isImage = file.type.startsWith('image/');

		// Determine if file has an extension
		const nameParts = file.name.split('.');
		const hasExtension = nameParts.length > 1;
		const ext = hasExtension ? (nameParts.pop()?.toLowerCase() || '') : '';

		// Check extensionless filename allowlist OR extension-based detection
		const isText = hasExtension
			? TEXT_EXTENSIONS.has(ext)
			: KNOWN_TEXT_FILENAMES.has(file.name.toLowerCase());

		// Validate based on model type
		if (!isText && !isVideo && !isImage) {
			const supported = isMultimodal
				? 'images, videos, and text files (.txt, .md, .py, .js, .json, .csv, .log, etc.)'
				: 'text files (.txt, .md, .py, .js, .json, .csv, .log, .yaml, .xml, etc.)';
			chatError = { summary: `Unsupported file type. Accepted: ${supported}` };
			return;
		}

		// Text-only models: reject media files
		if (!isMultimodal && (isImage || isVideo)) {
			chatError = { summary: 'This model only supports text file attachments' };
			return;
		}

		// Anthropic protocol: reject video files
		if (isVideo && protocol === 'anthropic') {
			chatError = { summary: 'Video attachments are not supported with the Anthropic protocol' };
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

	// Remove attachment and cleanup object URL
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

	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	async function buildMessageContent(text: string, currentAttachments: Attachment[]): Promise<string | any[]> {
		if (currentAttachments.length === 0) {
			return text;
		}

		if (protocol === 'anthropic') {
			// Anthropic content format
			// eslint-disable-next-line @typescript-eslint/no-explicit-any
			const parts: any[] = [{ type: 'text', text }];

			for (const attachment of currentAttachments) {
				if (attachment.type === 'text') {
					const fileContent = await readFileAsText(attachment.file);
					parts.push({
						type: 'text',
						text: `[File: ${attachment.file.name}]\n${fileContent}`
					});
				} else if (attachment.type === 'image') {
					// Anthropic expects raw base64 with media_type
					const dataUrl = await encodeFileAsBase64(attachment.file);
					// Strip "data:image/png;base64," prefix to get raw base64
					const commaIdx = dataUrl.indexOf(',');
					const rawBase64 = commaIdx >= 0 ? dataUrl.slice(commaIdx + 1) : dataUrl;
					// Extract media type from data URL
					const mediaTypeMatch = dataUrl.match(/^data:([^;]+);/);
					const mediaType = mediaTypeMatch ? mediaTypeMatch[1] : 'image/png';
					parts.push({
						type: 'image',
						source: {
							type: 'base64',
							media_type: mediaType,
							data: rawBase64
						}
					});
				}
				// Video attachments are blocked for Anthropic in addAttachment()
			}

			return parts;
		} else {
			// OpenAI content format
			const parts: ContentPart[] = [{ type: 'text', text }];

			for (const attachment of currentAttachments) {
				if (attachment.type === 'text') {
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
	}

	interface StreamResult {
		content: string;
		thinking: string;
		thinkingDur: number | undefined;
		toolCalls: Array<{ id: string; name: string; arguments: string }>;
		toolCallsDone: boolean;
		error: { summary: string; details?: string } | null;
	}

	// OpenAI SSE stream parser
	async function processOpenAISSEStream(response: Response): Promise<StreamResult> {
		const result: StreamResult = {
			content: '',
			thinking: '',
			thinkingDur: undefined,
			toolCalls: [],
			toolCallsDone: false,
			error: null,
		};

		const reader = response.body?.getReader();
		if (!reader) return result;

		const decoder = new TextDecoder();
		let buffer = '';
		let modelLoadNotified = false;
		let thinkingStartTime: number | null = null;
		// Track tool calls by index for accumulation
		const toolCallsByIndex: Record<number, { id: string; name: string; arguments: string }> = {};

		while (true) {
			const { done, value } = await reader.read();
			if (done) break;

			buffer += decoder.decode(value, { stream: true });
			const lines = buffer.split('\n');
			buffer = lines.pop() || '';

			for (const line of lines) {
				const trimmed = line.trim();
				if (!trimmed.startsWith('data: ')) continue;

				const dataStr = trimmed.slice(6);
				if (dataStr === '[DONE]') {
					// Stream complete - finalize tool calls if any
					const toolValues = Object.values(toolCallsByIndex);
					if (toolValues.length > 0) {
						result.toolCalls = toolValues;
						result.toolCallsDone = true;
					}
					// Finalize thinking duration if we were still thinking
					if (thinkingStartTime !== null && result.thinkingDur === undefined) {
						result.thinkingDur = (Date.now() - thinkingStartTime) / 1000;
						thinkingDuration = result.thinkingDur;
					}
					continue;
				}

				try {
					const data = JSON.parse(dataStr);

					// Check for error responses
					if (data.error) {
						const errMsg = typeof data.error === 'string' ? data.error : data.error.message || JSON.stringify(data.error);
						result.error = { summary: 'Inference error', details: errMsg };
						continue;
					}

					const choice = data.choices?.[0];
					if (!choice) continue;

					const delta = choice.delta || {};
					const finishReason = choice.finish_reason;

					// Handle reasoning_content (thinking)
					if (delta.reasoning_content) {
						if (thinkingStartTime === null) {
							thinkingStartTime = Date.now();
						}
						result.thinking += delta.reasoning_content;
						streamingThinking += delta.reasoning_content;
						if (!modelLoadNotified) {
							modelLoadNotified = true;
							serverStore.refresh();
						}
					}

					// Handle content
					if (delta.content) {
						// If we were in thinking mode, compute thinking duration on first content chunk
						if (thinkingStartTime !== null && result.thinkingDur === undefined) {
							result.thinkingDur = (Date.now() - thinkingStartTime) / 1000;
							thinkingDuration = result.thinkingDur;
						}
						result.content += delta.content;
						streamingResponse += delta.content;
						if (!modelLoadNotified) {
							modelLoadNotified = true;
							serverStore.refresh();
						}
					}

					// Handle tool_calls
					if (delta.tool_calls) {
						for (const tc of delta.tool_calls) {
							const idx = tc.index ?? 0;
							const existing = toolCallsByIndex[idx];
							if (existing) {
								if (tc.function?.arguments) {
									existing.arguments += tc.function.arguments;
								}
							} else {
								toolCallsByIndex[idx] = {
									id: tc.id || '',
									name: tc.function?.name || '',
									arguments: tc.function?.arguments || '',
								};
							}
						}
					}

					// Handle finish_reason
					if (finishReason === 'tool_calls') {
						result.toolCalls = Object.values(toolCallsByIndex);
						result.toolCallsDone = true;
					}

				} catch {
					// Ignore parse errors on individual lines
				}
			}
		}

		// Ensure tool calls are captured even without explicit finish_reason
		const finalToolValues = Object.values(toolCallsByIndex);
		if (finalToolValues.length > 0 && result.toolCalls.length === 0) {
			result.toolCalls = finalToolValues;
			result.toolCallsDone = true;
		}

		return result;
	}

	// Anthropic SSE stream parser
	async function processAnthropicSSEStream(response: Response): Promise<StreamResult> {
		const result: StreamResult = {
			content: '',
			thinking: '',
			thinkingDur: undefined,
			toolCalls: [],
			toolCallsDone: false,
			error: null,
		};

		const reader = response.body?.getReader();
		if (!reader) return result;

		const decoder = new TextDecoder();
		let buffer = '';
		let modelLoadNotified = false;
		// Track current event type from "event:" lines
		let currentEventType = '';
		// Track content blocks for tool use accumulation
		const toolBlocks: Record<number, { id: string; name: string; arguments: string }> = {};

		while (true) {
			const { done, value } = await reader.read();
			if (done) break;

			buffer += decoder.decode(value, { stream: true });
			const lines = buffer.split('\n');
			buffer = lines.pop() || '';

			for (const line of lines) {
				const trimmed = line.trim();

				// Track event type
				if (trimmed.startsWith('event: ')) {
					currentEventType = trimmed.slice(7).trim();
					continue;
				}

				// Skip empty lines and comments
				if (!trimmed.startsWith('data: ')) continue;

				const dataStr = trimmed.slice(6);

				try {
					const data = JSON.parse(dataStr);

					// Check for error events
					if (data.type === 'error' || data.error) {
						const errMsg = data.error?.message || data.message || JSON.stringify(data);
						result.error = { summary: 'Inference error', details: errMsg };
						continue;
					}

					switch (currentEventType) {
						case 'message_start': {
							// Extract metadata if needed
							if (!modelLoadNotified) {
								modelLoadNotified = true;
								serverStore.refresh();
							}
							break;
						}

						case 'content_block_start': {
							const block = data.content_block;
							if (block?.type === 'tool_use') {
								toolBlocks[data.index] = {
									id: block.id || '',
									name: block.name || '',
									arguments: '',
								};
							}
							break;
						}

						case 'content_block_delta': {
							const delta = data.delta;
							if (delta?.type === 'text_delta') {
								result.content += delta.text || '';
								streamingResponse += delta.text || '';
								if (!modelLoadNotified) {
									modelLoadNotified = true;
									serverStore.refresh();
								}
							} else if (delta?.type === 'input_json_delta') {
								const block = toolBlocks[data.index];
								if (block) {
									block.arguments += delta.partial_json || '';
								}
							}
							break;
						}

						case 'content_block_stop': {
							// Finalize tool block if it was a tool_use block
							// (nothing special needed - data is already accumulated)
							break;
						}

						case 'message_delta': {
							const stopReason = data.delta?.stop_reason;
							if (stopReason === 'tool_use') {
								result.toolCalls = Object.values(toolBlocks);
								result.toolCallsDone = true;
							}
							break;
						}

						case 'message_stop': {
							// Stream complete
							const remainingTools = Object.values(toolBlocks);
							if (remainingTools.length > 0 && result.toolCalls.length === 0) {
								result.toolCalls = remainingTools;
								result.toolCallsDone = true;
							}
							break;
						}
					}

				} catch {
					// Ignore parse errors on individual lines
				}
			}
		}

		return result;
	}

	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	function buildOpenAIRequest(apiMessages: any[]): { url: string; body: Record<string, unknown> } {
		return {
			url: '/v1/chat/completions',
			body: {
				model: selectedProfile!.model_repo_id,
				messages: apiMessages,
				stream: true,
				temperature: selectedProfile!.inference?.temperature ?? 0.7,
				max_tokens: selectedProfile!.inference?.max_tokens ?? 4096,
				top_p: selectedProfile!.inference?.top_p ?? 1.0,
				...(toolsEnabled && availableTools.length > 0
					? { tools: availableTools, tool_choice: 'auto' }
					: {}),
			},
		};
	}

	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	function buildAnthropicRequest(apiMessages: any[]): { url: string; body: Record<string, unknown> } {
		const systemPrompt = selectedProfile!.context?.system_prompt;
		// Filter system messages out for Anthropic (system goes in separate field)
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		const anthropicMessages = apiMessages.filter((m: any) => m.role !== 'system');
		return {
			url: '/v1/messages',
			body: {
				model: selectedProfile!.model_repo_id,
				max_tokens: selectedProfile!.inference?.max_tokens ?? 4096,
				...(systemPrompt ? { system: systemPrompt } : {}),
				messages: anthropicMessages,
				stream: true,
				temperature: Math.min(selectedProfile!.inference?.temperature ?? 0.7, 1.0),
				top_p: selectedProfile!.inference?.top_p ?? 1.0,
				...(toolsEnabled && availableTools.length > 0
					? {
						tools: availableTools.map(t => ({
							name: t.function.name,
							description: t.function.description,
							input_schema: t.function.parameters,
						})),
					}
					: {}),
			},
		};
	}

	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	async function sendWithRetry(userContent: string | any[], userAttachments: Attachment[], attempt: number = 1): Promise<boolean> {
		retryAttempt = attempt;
		isRetrying = attempt > 1;

		if (!selectedProfile) return false;

		// Build messages array for API
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		const apiMessages: any[] = [];

		// Add system prompt as first message (OpenAI puts it in messages; Anthropic filters it out later)
		if (selectedProfile.context?.system_prompt) {
			apiMessages.push({ role: 'system', content: selectedProfile.context.system_prompt });
		}

		// Add conversation history (exclude last user message since we're about to add it)
		for (const m of messages.slice(0, -1)) {
			apiMessages.push({ role: m.role, content: m.content });
		}

		// Add current user message
		apiMessages.push({ role: 'user', content: userContent });

		try {
			// Create abort controller for this request
			abortController = new AbortController();

			const { url, body: requestBody } = protocol === 'anthropic'
				? buildAnthropicRequest(apiMessages)
				: buildOpenAIRequest(apiMessages);

			const response = await fetch(url, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					'Authorization': `Bearer ${authStore.token}`,
				},
				body: JSON.stringify(requestBody),
				signal: abortController.signal,
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

			// Process the SSE stream based on protocol
			const streamResult = protocol === 'anthropic'
				? await processAnthropicSSEStream(response)
				: await processOpenAISSEStream(response);

			if (streamResult.error) {
				chatError = streamResult.error;
				return false;
			}

			// Handle tool-use loop (max 3 rounds to prevent infinite loops)
			const MAX_TOOL_DEPTH = 3;
			let currentResult = streamResult;
			let toolDepth = 0;
			let assistantContent = currentResult.content;

			while (currentResult.toolCallsDone && currentResult.toolCalls.length > 0 && toolDepth < MAX_TOOL_DEPTH) {
				toolDepth++;

				// Execute tool calls and store structured results
				for (const toolCall of currentResult.toolCalls) {
					const callData: ToolCallData = {
						id: toolCall.id,
						name: toolCall.name,
						arguments: toolCall.arguments,
					};

					try {
						const parsedArgs = JSON.parse(toolCall.arguments);
						const toolResult = await mcp.executeTool(toolCall.name, parsedArgs);
						callData.result = JSON.stringify(toolResult);
					} catch {
						callData.error = 'Tool execution failed';
					}

					streamingToolCalls = [...streamingToolCalls, callData];

					// Add assistant tool_calls message and tool result to apiMessages
					// Format differs by protocol
					if (protocol === 'anthropic') {
						apiMessages.push({
							role: 'assistant',
							content: [{
								type: 'tool_use',
								id: toolCall.id,
								name: toolCall.name,
								input: JSON.parse(toolCall.arguments || '{}'),
							}]
						});
						apiMessages.push({
							role: 'user',
							content: [{
								type: 'tool_result',
								tool_use_id: toolCall.id,
								content: callData.result || callData.error || '',
							}]
						});
					} else {
						apiMessages.push({
							role: 'assistant',
							content: '',
							tool_calls: [{
								id: toolCall.id,
								type: 'function',
								function: { name: toolCall.name, arguments: toolCall.arguments }
							}]
						});
						apiMessages.push({
							role: 'tool',
							tool_call_id: toolCall.id,
							content: callData.result || callData.error || ''
						});
					}
				}

				// Send follow-up request with tool results
				const { url: followUpUrl, body: followUpBody } = protocol === 'anthropic'
					? buildAnthropicRequest(apiMessages)
					: buildOpenAIRequest(apiMessages);

				const followUpRes = await fetch(followUpUrl, {
					method: 'POST',
					headers: {
						'Content-Type': 'application/json',
						'Authorization': `Bearer ${authStore.token}`,
					},
					body: JSON.stringify(followUpBody),
				});

				if (!followUpRes.ok) {
					console.error('Follow-up request failed:', followUpRes.status, followUpRes.statusText);
					chatError = { summary: 'Tool follow-up failed', details: `Status: ${followUpRes.status}` };
					break;
				}

				// Process follow-up stream (may contain more tool calls)
				currentResult = protocol === 'anthropic'
					? await processAnthropicSSEStream(followUpRes)
					: await processOpenAISSEStream(followUpRes);

				// Debug: log what we got from follow-up
				console.log('Follow-up result:', {
					content: currentResult.content?.slice(0, 100),
					thinking: currentResult.thinking?.slice(0, 100),
					toolCalls: currentResult.toolCalls.length,
					error: currentResult.error
				});

				// Accumulate both content and any additional thinking from follow-up
				assistantContent += currentResult.content;
				streamingResponse = assistantContent;

				if (currentResult.error) {
					chatError = currentResult.error;
					break;
				}
			}

			// If we hit max depth and model still wants tools, show warning
			if (toolDepth >= MAX_TOOL_DEPTH && currentResult.toolCallsDone) {
				assistantContent += '\n\n*[Tool call limit reached (3 rounds). Stopping automatic execution.]*';
				streamingResponse = assistantContent;
			}

			// Finalize message - clean the response to remove tool call JSON, special tokens
			if (streamingResponse || streamingThinking || streamingToolCalls.length > 0) {
				const cleanedResponse = cleanResponse(streamingResponse);
				const finalContent = streamingThinking
					? `<think>${streamingThinking}</think>${cleanedResponse}`
					: cleanedResponse;
				messages.push({
					role: 'assistant',
					content: finalContent,
					toolCalls: streamingToolCalls.length > 0 ? [...streamingToolCalls] : undefined,
					thinkingDuration: thinkingDuration
				});
			}

			// On success, reset retry state
			retryAttempt = 0;
			isRetrying = false;
			lastFailedMessage = null;
			return true;

		} catch (error) {
			// Check if request was aborted (user clicked Stop)
			if (error instanceof DOMException && error.name === 'AbortError') {
				// User stopped generation - not an error
				return true; // Don't remove user message, handleStop already added partial response
			}

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
				summary: 'Failed to get response after 3 attempts',
				details: `The model may still be loading or the request may have failed.\n\nError: ${errorMsg}`
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

		// Clear attachments
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
		streamingToolCalls = [];
		loading = true;

		try {
			const success = await sendWithRetry(content, currentAttachments, 1);
			if (!success) {
				// Remove the user message on failure
				messages.pop();
			}
		} finally {
			loading = false;
			abortController = null;
		}
	}

	function handleClear() {
		messages = [];
		chatError = null;
	}

	async function handleCopyChat() {
		if (messages.length === 0) return;

		const parts: string[] = [];
		parts.push(`# Chat Transcript`);
		parts.push(`Model: ${selectedProfile?.model_repo_id || 'Unknown'}`);
		parts.push(`Protocol: ${protocol === 'openai' ? 'OpenAI' : 'Anthropic'}`);
		parts.push(`Date: ${new Date().toISOString()}`);
		parts.push('');

		for (const message of messages) {
			if (message.role === 'user') {
				parts.push(`## User`);
				parts.push(typeof message.content === 'string' ? message.content : '[multipart content]');
			} else {
				parts.push(`## Assistant`);
				const parsed = parseThinking(message.content);
				if (parsed.thinking) {
					parts.push(`### Thinking${message.thinkingDuration ? ` (${message.thinkingDuration.toFixed(1)}s)` : ''}`);
					parts.push(parsed.thinking);
					parts.push('');
				}
				if (message.toolCalls && message.toolCalls.length > 0) {
					parts.push(`### Tool Calls`);
					for (const tc of message.toolCalls) {
						parts.push(`**${tc.name}**(${tc.arguments})`);
						if (tc.result) {
							parts.push(`Result: ${tc.result}`);
						}
						if (tc.error) {
							parts.push(`Error: ${tc.error}`);
						}
					}
					parts.push('');
				}
				if (parsed.response) {
					parts.push(`### Response`);
					parts.push(parsed.response);
				}
			}
			parts.push('');
		}

		const transcript = parts.join('\n');
		await navigator.clipboard.writeText(transcript);
		copyFeedback = true;
		setTimeout(() => copyFeedback = false, 2000);
	}

	function handleStop() {
		if (abortController) {
			abortController.abort();
			abortController = null;
		}
		loading = false;
		// Keep any partial response that was streamed
		if (streamingResponse || streamingThinking) {
			const cleanedResponse = cleanResponse(streamingResponse);
			const finalContent = streamingThinking
				? `<think>${streamingThinking}</think>${cleanedResponse}\n\n*[Generation stopped by user]*`
				: cleanedResponse + '\n\n*[Generation stopped by user]*';
			messages.push({
				role: 'assistant',
				content: finalContent,
				toolCalls: streamingToolCalls.length > 0 ? [...streamingToolCalls] : undefined,
				thinkingDuration: thinkingDuration
			});
		}
		// Reset streaming state
		streamingThinking = '';
		streamingResponse = '';
		thinkingDuration = undefined;
		streamingToolCalls = [];
	}

	function handleProfileChange(e: Event) {
		const target = e.target as HTMLSelectElement;
		selectedProfileId = target.value ? parseInt(target.value, 10) : null;
		messages = [];
		chatError = null;
		// Clear attachments when switching profiles
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

	// Clean response text by removing tool call JSON, special tokens, etc.
	// This handles cases where models output raw JSON tool calls inline
	function cleanResponse(text: string): string {
		let cleaned = text;

		// Remove raw JSON tool calls: {"name": "...", "arguments": {...}}
		cleaned = cleaned.replace(
			/\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}\s*\}/g,
			''
		);

		// Remove tagged tool calls: <tool_call>...</tool_call>
		cleaned = cleaned.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, '');

		// Remove special tokens
		const specialTokens = [
			'<|endoftext|>',
			'<|im_end|>',
			'<|im_start|>',
			'<|end|>',
			'<|eot_id|>',
			'<|start_header_id|>',
			'<|end_header_id|>',
		];
		for (const token of specialTokens) {
			cleaned = cleaned.replaceAll(token, '');
		}

		// Clean up excessive whitespace from removals
		cleaned = cleaned.replace(/\n{3,}/g, '\n\n');
		cleaned = cleaned.trim();

		return cleaned;
	}
</script>

<div class="space-y-6 h-[calc(100vh-8rem)] flex flex-col">
	<div class="flex items-center justify-between">
		<h1 class="text-2xl font-bold">Chat</h1>
		<div class="flex items-center gap-4">
			<Select onchange={handleProfileChange} value={selectedProfileId?.toString() ?? ''}>
				<option value="">Select a profile...</option>
				{#each availableProfiles as profile (profile.id)}
					<option value={profile.id.toString()}>
						{profile.name} ({getModelShortName(profile.model_repo_id ?? profile.name)})
					</option>
				{/each}
			</Select>
			<div class="flex items-center rounded-lg border bg-muted p-0.5">
				<button
					type="button"
					class="px-3 py-1 text-xs font-medium rounded-md transition-colors {protocol === 'openai' ? 'bg-background shadow-sm text-foreground' : 'text-muted-foreground hover:text-foreground'}"
					onclick={() => { protocol = 'openai'; messages = []; chatError = null; }}
				>
					OpenAI
				</button>
				<button
					type="button"
					class="px-3 py-1 text-xs font-medium rounded-md transition-colors {protocol === 'anthropic' ? 'bg-background shadow-sm text-foreground' : 'text-muted-foreground hover:text-foreground'}"
					onclick={() => { protocol = 'anthropic'; messages = []; chatError = null; }}
				>
					Anthropic
				</button>
			</div>
			{#if messages.length > 0}
				<Button variant="outline" onclick={handleCopyChat} title="Copy chat transcript to clipboard">
					<Copy class="w-4 h-4 mr-2" />
					{copyFeedback ? 'Copied!' : 'Copy'}
				</Button>
				<Button variant="outline" onclick={handleClear}>
					Clear Chat
				</Button>
			{/if}
		</div>
	</div>

	{#if availableProfiles.length === 0}
		<Card class="flex-1 flex items-center justify-center">
			<div class="text-center">
				<Bot class="w-16 h-16 mx-auto text-muted-foreground mb-4" />
				<p class="text-muted-foreground mb-4">No running chat-capable profiles.</p>
				<Button href="/profiles">Manage Profiles</Button>
			</div>
		</Card>
	{:else if !selectedProfile}
		<Card class="flex-1 flex items-center justify-center">
			<div class="text-center">
				<Bot class="w-16 h-16 mx-auto text-muted-foreground mb-4" />
				<p class="text-muted-foreground">Select a profile to start chatting.</p>
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
							<p class="text-sm mt-1">Model: {selectedProfile.model_repo_id ?? 'Unknown'} ({protocol === 'openai' ? 'OpenAI' : 'Anthropic'} API)</p>
						</div>
					</div>
				{:else}
					<!-- System Prompt (pinned first message) -->
					{#if selectedProfile?.context?.system_prompt}
						<div class="flex gap-3 opacity-60">
							<div class="flex-shrink-0 w-8 h-8 rounded-full bg-muted flex items-center justify-center">
								<Bot class="w-4 h-4 text-muted-foreground" />
							</div>
							<div class="flex-1 min-w-0">
								<p class="text-xs text-muted-foreground mb-1 italic">System Prompt</p>
								<div class="text-sm text-muted-foreground italic whitespace-pre-wrap">
									{selectedProfile.context.system_prompt}
								</div>
							</div>
						</div>
					{:else if selectedProfile && !selectedProfile.context?.system_prompt}
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
										<ThinkingBubble content={parsed.thinking} duration={message.thinkingDuration} />
									{/if}
									{#if message.toolCalls && message.toolCalls.length > 0}
										<ToolCallBubble calls={message.toolCalls} />
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
								{#if streamingToolCalls.length > 0}
									<ToolCallBubble calls={streamingToolCalls} />
								{/if}
								{#if streamingResponse}
									<Markdown content={streamingResponse} />
								{:else if !streamingThinking && streamingToolCalls.length === 0}
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
											: lastFailedMessage.content.find((p: ContentPart) => p.type === 'text')?.text || '';
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
					{#if toolsLoaded}
						<Button
							type="button"
							variant={toolsEnabled ? "default" : "ghost"}
							size="icon"
							onclick={() => toolsEnabled = !toolsEnabled}
							title={toolsEnabled ? `Tools enabled (${availableTools.length})` : "Enable tools"}
						>
							<Wrench class="w-4 h-4" />
						</Button>
					{/if}
					<Button
						type="button"
						variant="ghost"
						size="icon"
						onclick={() => fileInputRef?.click()}
						disabled={loading || attachments.length >= 3}
						title={protocol === 'anthropic' && isMultimodal ? 'Attach files (video not supported with Anthropic)' : 'Attach files'}
					>
						<Paperclip class="w-4 h-4" />
					</Button>
					<textarea
						bind:this={textareaRef}
						bind:value={input}
						placeholder="Type a message..."
						disabled={loading}
						class="flex-1 resize-none overflow-hidden rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
						rows="1"
						style="max-height: 150px; overflow-y: {input.split('\n').length > 4 ? 'auto' : 'hidden'}"
						oninput={(e) => { const el = e.currentTarget; el.style.height = 'auto'; el.style.height = Math.min(el.scrollHeight, 150) + 'px'; }}
						onkeydown={(e) => {
							if (e.key === 'Enter' && !e.shiftKey) {
								e.preventDefault();
								const form = e.currentTarget.closest('form');
								if (form && input.trim()) form.requestSubmit();
							}
						}}
					></textarea>
					{#if loading}
						<Button type="button" variant="destructive" onclick={handleStop} title="Stop generation">
							<Square class="w-4 h-4" />
						</Button>
					{:else}
						<Button type="submit" disabled={!input.trim()}>
							<Send class="w-4 h-4" />
						</Button>
					{/if}
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
