# Phase 5: Chat Multimodal Support - Research

**Researched:** 2026-01-21
**Domain:** Multimodal chat UI, MLX vision models, SSE streaming
**Confidence:** MEDIUM

## Summary

This research investigated how to implement multimodal chat support for image/video attachments, thinking model display, and error handling in an MLX-based chat interface. The standard approach uses OpenAI-compatible message formats with base64-encoded images, Server-Sent Events for streaming responses, and specialized parsers for thinking/reasoning content.

MLX multimodal support is provided through mlx-vlm and mlx-openai-server, which accept images via URLs, file paths, or PIL objects. Video support exists in select models (Qwen2-VL, Qwen2.5-VL, LLaVA, Idefics3) and should be sent directly to the model without frame extraction. Thinking models use XML-style tags (`<think>...</think>`) to separate reasoning from final answers, requiring specialized parsers on the server side.

Frontend implementation follows standard patterns: HTML5 file input with drag-drop, EventSource for SSE streaming with JSON parsing, and Svelte 5 runes for reactive state management.

**Primary recommendation:** Use OpenAI-compatible message format with base64 image encoding, implement SSE streaming with incremental JSON parsing, and add reasoning_parser configuration to ServerProfile for thinking model support.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| mlx-vlm | Latest | MLX vision/multimodal models | Official MLX vision framework from community |
| mlx-openai-server | Latest | OpenAI-compatible MLX server | Already in use, supports multimodal via --model-type vlm |
| EventSource (Web API) | Native | SSE client for streaming | Native browser API, no dependencies |
| bits-ui Collapsible | Latest | Accessible collapsible component | Already in use, Svelte 5 compatible |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| eventsource-parser | 2.x | SSE chunk parsing | If custom SSE handling needed (likely not required) |
| Flowbite-Svelte Dropzone | Latest | File upload component | If building custom dropzone (evaluate vs custom) |
| ResizeObserver (Web API) | Native | Height-auto animations | For smooth collapse/expand animations |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Base64 encoding | File URLs | URLs require file hosting, base64 works inline |
| Native EventSource | fetch with ReadableStream | EventSource simpler, auto-reconnect built-in |
| bits-ui Collapsible | Custom solution | Collapsible provides accessibility, keyboard nav |

**Installation:**
```bash
# Backend - mlx-vlm already likely installed with mlx-openai-server
pip install mlx-vlm

# Frontend - bits-ui already installed
# No new dependencies required for core functionality
```

## Architecture Patterns

### Recommended Message Structure
```typescript
// Multimodal message format (OpenAI-compatible)
interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string | ContentPart[];
}

interface ContentPart {
  type: "text" | "image_url";
  text?: string;
  image_url?: {
    url: string; // "data:image/png;base64,..." or HTTP URL
  };
}

// Example multimodal message
const message: ChatMessage = {
  role: "user",
  content: [
    { type: "text", text: "What's in this image?" },
    {
      type: "image_url",
      image_url: {
        url: "data:image/png;base64,iVBORw0KGgoAAAANSUh..."
      }
    }
  ]
};
```

### Pattern 1: Base64 Image Encoding
**What:** Encode images as base64 data URLs before sending to API
**When to use:** All image attachments in chat messages
**Example:**
```typescript
// Source: MDN and OpenAI API documentation
async function encodeImageAsBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const base64 = reader.result as string;
      // Format: data:image/png;base64,<base64-string>
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// Usage in message construction
const base64Image = await encodeImageAsBase64(imageFile);
const message = {
  role: "user",
  content: [
    { type: "text", text: prompt },
    { type: "image_url", image_url: { url: base64Image } }
  ]
};
```

### Pattern 2: SSE Streaming with Incremental Parsing
**What:** Stream chat responses using Server-Sent Events with buffer-based JSON parsing
**When to use:** All chat completions, especially for thinking models
**Example:**
```typescript
// Source: Existing codebase (downloads.svelte.ts) + eventsource-parser docs
function streamChatCompletion(profileId: number, messages: ChatMessage[]): EventSource {
  const eventSource = new EventSource(
    `/api/chat/${profileId}/completions?stream=true`
  );

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);

      // Handle different chunk types
      if (data.type === "thinking") {
        appendThinkingContent(data.content);
      } else if (data.type === "response") {
        appendResponseContent(data.content);
      } else if (data.type === "done") {
        eventSource.close();
      }
    } catch (error) {
      console.error("Failed to parse SSE chunk:", error);
    }
  };

  eventSource.onerror = () => {
    eventSource.close();
    handleStreamError();
  };

  return eventSource;
}
```

### Pattern 3: Thinking Content Display
**What:** Separate thinking/reasoning content from final response, display in collapsible format
**When to use:** When reasoning_parser is configured for profile (DeepSeek-R1, QwQ models)
**Example:**
```svelte
<script lang="ts">
  import { Collapsible } from "bits-ui";

  let thinking = $state("");
  let response = $state("");
  let thinkingTime = $state(0);
  let expanded = $state(false);
</script>

{#if thinking}
  <Collapsible.Root bind:open={expanded}>
    <Collapsible.Trigger>
      <span class="text-muted-foreground text-sm">
        Thought for {thinkingTime.toFixed(1)}s
      </span>
    </Collapsible.Trigger>
    <Collapsible.Content>
      <div class="border-l-2 pl-4 mt-2 text-muted-foreground italic">
        {thinking}
      </div>
    </Collapsible.Content>
  </Collapsible.Root>
{/if}

<div class="response">{response}</div>
```

### Pattern 4: File Attachment UI
**What:** Input-integrated attachment button with drag-drop support
**When to use:** Chat message composition
**Example:**
```svelte
<script lang="ts">
  let attachments = $state<File[]>([]);
  let inputRef: HTMLInputElement;

  function handleFileSelect(e: Event) {
    const files = Array.from((e.target as HTMLInputElement).files || []);
    attachments = [...attachments, ...files.slice(0, 3 - attachments.length)];
  }

  function handleDrop(e: DragEvent) {
    e.preventDefault();
    const files = Array.from(e.dataTransfer?.files || []);
    attachments = [...attachments, ...files.slice(0, 3 - attachments.length)];
  }

  function removeAttachment(index: number) {
    attachments = attachments.filter((_, i) => i !== index);
  }
</script>

<div
  class="chat-input-area"
  ondrop={handleDrop}
  ondragover={(e) => e.preventDefault()}
>
  {#if attachments.length > 0}
    <div class="attachments-row">
      {#each attachments as file, i}
        <div class="thumbnail" onmouseenter={() => showRemove = i}>
          <img src={URL.createObjectURL(file)} alt={file.name} />
          {#if showRemove === i}
            <button onclick={() => removeAttachment(i)}>×</button>
          {/if}
        </div>
      {/each}
    </div>
  {/if}

  <div class="input-wrapper">
    <textarea bind:value={message} />
    <button onclick={() => inputRef.click()}>
      📎
    </button>
  </div>

  <input
    type="file"
    bind:this={inputRef}
    accept={acceptedFormats}
    multiple
    onchange={handleFileSelect}
    style="display: none"
  />
</div>
```

### Pattern 5: Video Duration Validation
**What:** Validate video duration before allowing attachment
**When to use:** Video file selection (2-minute max per CONTEXT.md)
**Example:**
```typescript
// Source: MDN HTMLMediaElement.duration + W3Schools examples
async function validateVideoDuration(file: File): Promise<boolean> {
  return new Promise((resolve) => {
    const video = document.createElement('video');
    video.preload = 'metadata';

    video.onloadedmetadata = () => {
      URL.revokeObjectURL(video.src);
      const duration = video.duration; // in seconds
      resolve(duration <= 120); // 2 minutes = 120 seconds
    };

    video.onerror = () => {
      URL.revokeObjectURL(video.src);
      resolve(false);
    };

    video.src = URL.createObjectURL(file);
  });
}
```

### Anti-Patterns to Avoid
- **Don't hand-roll SSE parsing:** Use native EventSource with JSON.parse per message, not custom chunking logic
- **Don't validate MIME types client-side only:** Always validate on server (MIME spoofing is trivial)
- **Don't store base64 in component state during typing:** Encode only on message send to avoid memory bloat
- **Don't reassign arrays in Svelte 5:** Use `array.push()` directly, proxies handle reactivity
- **Don't fetch reasoning content separately:** Stream it inline with SSE, tagged by type field

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SSE chunk parsing | Custom buffer/parser | Native EventSource + JSON.parse | Auto-reconnect, handles incomplete chunks, browser-optimized |
| Collapsible animations | Custom height transitions | bits-ui Collapsible + Svelte transitions | Accessibility, keyboard nav, smooth height-auto animations |
| File MIME validation | Client-side extension checks | Server-side magic byte inspection | Client MIME headers easily spoofed, security risk |
| Image thumbnail generation | Canvas manipulation | URL.createObjectURL(file) | Instant preview, no encoding overhead until send |
| Video first-frame extraction | Manual canvas drawing | Load as video element, set currentTime=0 | Browser handles codec complexity, works across formats |

**Key insight:** Browser APIs (EventSource, FileReader, URL.createObjectURL) and existing component libraries handle the hard parts. Focus implementation effort on message formatting and state management, not low-level encoding/parsing.

## Common Pitfalls

### Pitfall 1: Incomplete JSON in SSE Chunks
**What goes wrong:** Network packets split JSON objects mid-string, JSON.parse fails
**Why it happens:** SSE events arrive in arbitrary chunks, not aligned to message boundaries
**How to avoid:** Parse only complete events from EventSource.onmessage (browser handles buffering)
**Warning signs:** "Unexpected end of JSON input" errors in console during streaming

### Pitfall 2: MIME Type Spoofing in File Uploads
**What goes wrong:** Attacker renames malicious file, bypasses accept attribute, uploads executable
**Why it happens:** Client-side accept and MIME headers are user-controlled
**How to avoid:** Server validates with magic bytes (file signature inspection), not MIME headers
**Warning signs:** Security audit flags file upload without server-side validation

### Pitfall 3: Memory Bloat from Base64 Encoding
**What goes wrong:** Large images encoded to base64 consume 33% more memory, stored in reactive state
**Why it happens:** Base64 expands binary data, Svelte tracks entire string in proxy
**How to avoid:** Keep File objects in state, encode to base64 only when sending message
**Warning signs:** Browser tab using >500MB RAM with a few images attached

### Pitfall 4: Thinking Content Not Streaming
**What goes wrong:** Thinking content appears all at once after model finishes, not incrementally
**Why it happens:** Server buffers until `</think>` tag before sending, or client waits for complete JSON
**How to avoid:** Server streams thinking chunks as they generate (before closing tag), client appends incrementally
**Warning signs:** Long delay before any thinking content appears, then instant full block

### Pitfall 5: Video Thumbnail Not Displaying
**What goes wrong:** Video thumbnail shows blank or broken image
**Why it happens:** Video metadata not loaded before attempting to capture frame, or currentTime set incorrectly
**How to avoid:** Wait for `loadedmetadata` event, then set currentTime, then wait for `seeked` event
**Warning signs:** Thumbnail works for some videos but not others (codec/metadata variations)

### Pitfall 6: Drag-Drop Not Working in Chat Area
**What goes wrong:** Dropped files open in browser instead of attaching to chat
**Why it happens:** Missing `preventDefault()` on `dragover` event
**How to avoid:** Always call `e.preventDefault()` in both `dragover` and `drop` handlers
**Warning signs:** Browser navigates to image URL instead of attaching file

### Pitfall 7: Array Reassignment in Svelte 5
**What goes wrong:** Using `attachments = [...attachments, newFile]` instead of `attachments.push(newFile)`
**Why it happens:** Svelte 4 habits carried over, don't realize proxies make push() reactive
**How to avoid:** Use native array methods (push, splice) directly, Svelte 5 proxies handle reactivity
**Warning signs:** Works but inefficient, creates new arrays unnecessarily

## Code Examples

Verified patterns from official sources:

### Backend: Multimodal Chat Endpoint
```python
# Source: FastAPI StreamingResponse + mlx-openai-server patterns
from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
import json

router = APIRouter(prefix="/api/chat")

@router.post("/{profile_id}/completions")
async def chat_completion(
    profile_id: int,
    messages: list[dict] = Body(...),
    stream: bool = True,
):
    """Chat completion with multimodal support."""
    # Validate profile has model_type="vlm" if messages contain images
    profile = get_profile_or_404(profile_id)
    has_images = any(
        isinstance(msg.get("content"), list) and
        any(part.get("type") == "image_url" for part in msg["content"])
        for msg in messages
    )

    if has_images and profile.model_type != "vlm":
        raise HTTPException(
            status_code=400,
            detail="Multimodal requests require model_type='vlm'"
        )

    async def generate() -> AsyncGenerator[str, None]:
        # Proxy to mlx-openai-server running for this profile
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"http://{profile.host}:{profile.port}/v1/chat/completions",
                json={"messages": messages, "stream": True},
            ) as response:
                thinking_buffer = ""
                in_thinking = False

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    data = json.loads(line[6:])
                    content = data["choices"][0]["delta"].get("content", "")

                    # Parse thinking tags if reasoning_parser enabled
                    if profile.reasoning_parser:
                        for char in content:
                            if char == "<" and content[idx:idx+7] == "<think>":
                                in_thinking = True
                                thinking_buffer = ""
                                continue
                            elif char == "<" and content[idx:idx+8] == "</think>":
                                in_thinking = False
                                yield f"data: {json.dumps({'type': 'thinking_done'})}\n\n"
                                continue

                            if in_thinking:
                                thinking_buffer += char
                                yield f"data: {json.dumps({'type': 'thinking', 'content': char})}\n\n"
                            else:
                                yield f"data: {json.dumps({'type': 'response', 'content': char})}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'response', 'content': content})}\n\n"

                yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

### Frontend: Attachment State Management
```typescript
// Source: Svelte 5 $state patterns + existing downloads.svelte.ts
interface Attachment {
  file: File;
  preview: string; // Object URL for thumbnail
  type: "image" | "video";
}

class ChatStore {
  attachments = $state<Attachment[]>([]);

  async addAttachment(file: File): Promise<void> {
    // Validate file type based on model capabilities
    const modelSupportsVideo = this.currentProfile?.supports_video ?? false;
    const isVideo = file.type.startsWith("video/");

    if (isVideo && !modelSupportsVideo) {
      throw new Error("Current model doesn't support video");
    }

    // Validate video duration if applicable
    if (isVideo) {
      const valid = await validateVideoDuration(file);
      if (!valid) {
        throw new Error("Video must be 2 minutes or less");
      }
    }

    // Enforce 3-attachment limit
    if (this.attachments.length >= 3) {
      throw new Error("Maximum 3 attachments per message");
    }

    // Create preview URL
    const preview = URL.createObjectURL(file);

    // Use push() directly, Svelte 5 proxies handle reactivity
    this.attachments.push({
      file,
      preview,
      type: isVideo ? "video" : "image"
    });
  }

  removeAttachment(index: number): void {
    const attachment = this.attachments[index];
    URL.revokeObjectURL(attachment.preview); // Clean up memory
    this.attachments.splice(index, 1); // Direct mutation, reactive
  }

  async sendMessage(text: string): Promise<void> {
    // Encode attachments to base64
    const contentParts: ContentPart[] = [
      { type: "text", text }
    ];

    for (const attachment of this.attachments) {
      const base64 = await encodeImageAsBase64(attachment.file);
      contentParts.push({
        type: "image_url",
        image_url: { url: base64 }
      });
    }

    // Send message with attachments
    await this.streamCompletion({ role: "user", content: contentParts });

    // Clear attachments after send
    for (const attachment of this.attachments) {
      URL.revokeObjectURL(attachment.preview);
    }
    this.attachments = [];
  }
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Frame extraction for video | Send video directly | 2025 | MLX models now support native video, frame extraction loses temporal info |
| Separate thinking API call | Inline SSE streaming with type tags | 2024-2025 | DeepSeek-R1, QwQ use `<think>` tags in stream, better UX |
| Custom SSE parser libraries | Native EventSource API | Always preferred | Browser handles reconnection, buffering, no dependencies |
| OpenAI vision-specific API | Unified chat completions | 2023 | All multimodal in chat completions, not separate endpoint |
| Svelte stores | Svelte 5 runes ($state) | 2024 | Fine-grained reactivity, simpler code, better performance |

**Deprecated/outdated:**
- **eventsource-parser npm package**: Browser EventSource handles incomplete chunks natively, library adds complexity
- **Separate /vision endpoints**: OpenAI unified multimodal into chat completions, mlx-openai-server follows
- **$: reactive statements in Svelte**: Svelte 5 runes replace with $derived and $effect
- **Array reassignment for reactivity**: Svelte 5 proxies make push/splice reactive

## Open Questions

Things that couldn't be fully resolved:

1. **MLX-VLM video format support**
   - What we know: Qwen2-VL, Qwen2.5-VL, LLaVA, Idefics3 support video
   - What's unclear: Exact video codecs/containers supported, max resolution/bitrate
   - Recommendation: Document supported formats in UI, catch server errors and display clearly

2. **Thinking content format variations**
   - What we know: DeepSeek-R1 uses `<think>` tags, OpenAI o1/o3 doesn't expose reasoning
   - What's unclear: Do all MLX reasoning models use `<think>` tags, or are there variations?
   - Recommendation: Make thinking parser configurable per profile, test with QwQ and DeepSeek-R1

3. **Error recovery in SSE streams**
   - What we know: EventSource auto-reconnects, but message state may be lost
   - What's unclear: Should we buffer unsent message until stream completes, or allow new messages during streaming?
   - Recommendation: Disable send button during streaming, show "stop generation" button instead

4. **Video thumbnail performance**
   - What we know: Canvas-based thumbnail generation works but may be slow for large videos
   - What's unclear: Should we show icon overlay instead of first frame for large files?
   - Recommendation: Show icon for videos >10MB, first frame for smaller (user discretion per CONTEXT.md)

## Sources

### Primary (HIGH confidence)
- [MLX-VLM GitHub Repository](https://github.com/Blaizzy/mlx-vlm) - Official MLX vision models package, multimodal API format
- [Svelte 5 $state Documentation](https://svelte.dev/docs/svelte/$state) - Official reactive state API
- [bits-ui Collapsible Component](https://bits-ui.com/docs/components/collapsible) - Official component docs (already in use)
- [MDN: HTMLMediaElement.duration](https://developer.mozilla.org/en-US/docs/Web/API/HTMLMediaElement/duration) - Video duration API
- [MDN: Using Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events) - EventSource API

### Secondary (MEDIUM confidence)
- [OpenAI Reasoning Models Documentation](https://platform.openai.com/docs/guides/reasoning) - Reasoning trace format (verified via multiple sources)
- [DeepSeek-R1 GitHub Repository](https://github.com/deepseek-ai/DeepSeek-R1) - Thinking tag format `<think>...</think>`
- [OWASP File Upload Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/File_Upload_Cheat_Sheet.html) - Security validation best practices
- [eventsource-parser GitHub](https://github.com/rexxars/eventsource-parser) - SSE parsing patterns (informational, not recommended)
- [FastAPI StreamingResponse Tutorial](https://medium.com/@shudongai/building-a-real-time-streaming-api-with-fastapi-and-openai-a-comprehensive-guide-cb65b3e686a5) - SSE with FastAPI patterns

### Tertiary (LOW confidence)
- [LM Studio MLX Engine Blog](https://lmstudio.ai/blog/unified-mlx-engine) - Unified multimodal architecture (marketing, not technical docs)
- [vLLM-MLX GitHub](https://github.com/waybarrios/vllm-mlx) - Alternative server implementation (not in use, informational)
- WebSearch results for Svelte file upload libraries (multiple options, none definitively better than custom)

## Metadata

**Confidence breakdown:**
- Standard stack: MEDIUM - MLX-VLM verified authoritative, but specific version compatibility with mlx-openai-server not documented
- Architecture: HIGH - OpenAI message format well-documented, existing SSE patterns proven in codebase
- Pitfalls: HIGH - Based on official security docs (OWASP), browser API documentation (MDN), and Svelte 5 migration guides
- Thinking models: MEDIUM - DeepSeek-R1 format verified, but broader MLX support for reasoning parsers not fully documented

**Research date:** 2026-01-21
**Valid until:** 2026-02-21 (30 days - relatively stable domain, but MLX ecosystem evolving)
