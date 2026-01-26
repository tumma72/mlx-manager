---
status: resolved
trigger: "Investigate why MCP tool calls are displayed as large bold markdown text instead of a proper collapsible UI component."
created: 2026-01-24T10:30:00Z
updated: 2026-01-24T10:36:00Z
---

## Current Focus

hypothesis: CONFIRMED - Tool calls are appended as markdown strings to assistantContent which is then rendered by the Markdown component (marked library), causing bold text to appear large
test: Confirmed by reading chat page lines 413-423 and markdown.svelte implementation
expecting: Need to extract tool calls from message content and render them with a dedicated ToolCallBubble component
next_action: Document root cause

## Symptoms

expected: Tool calls should appear in a collapsible panel similar to ThinkingBubble component showing "Tool Calls: N" header
actual: Tool calls are displayed as large bold markdown text in the message content
errors: None reported
reproduction: Use chat with MCP tools enabled, observe tool call display
started: After implementing plan 06-13 (tool-use execution loop)

## Eliminated

## Evidence

- timestamp: 2026-01-24T10:35:00Z
  checked: frontend/src/routes/(protected)/chat/+page.svelte lines 408-446
  found: Tool calls are appended as markdown strings to assistantContent: `**Tool call:** \`${toolCall.name}(${toolCall.arguments})\`\n` and `**Result:** \`${JSON.stringify(toolResult)}\`\n`
  implication: These strings are part of message.content and rendered by the Markdown component

- timestamp: 2026-01-24T10:35:00Z
  checked: frontend/src/lib/components/ui/markdown.svelte
  found: Uses marked library to parse markdown content with @html directive, which renders **bold** text as large heading-like text in prose styling
  implication: The markdown renderer is correctly parsing the bold syntax, but prose CSS makes it visually prominent

- timestamp: 2026-01-24T10:35:00Z
  checked: frontend/src/lib/components/ui/thinking-bubble.svelte
  found: Uses Collapsible component from bits-ui with chevron icon, label showing status/duration, and content in bordered/indented section
  implication: This pattern should be replicated for tool calls - collapsible panel with tool count, expandable to show formatted details

- timestamp: 2026-01-24T10:35:00Z
  checked: Message rendering in chat page lines 725-730
  found: Assistant messages parse thinking content with parseThinking() function, then render ThinkingBubble for thinking and Markdown for response
  implication: Need similar pattern - parse tool calls from content, render with ToolCallBubble component, show remaining content in Markdown

## Resolution

root_cause: Tool calls are concatenated as markdown strings into message.content and rendered by Markdown component, which uses prose CSS that makes bold text (**) appear large and prominent instead of being displayed in a dedicated collapsible UI component
fix: Create ToolCallBubble component (similar to ThinkingBubble) and parseToolCalls() function (similar to parseThinking()) to extract tool call data from message content and render it separately
verification: Tool calls should appear in collapsible panel with "Tool Calls: N" header, expandable to show code-formatted details
files_changed:
  - path: frontend/src/routes/(protected)/chat/+page.svelte
    issue: Lines 413-423 append tool calls as markdown strings; lines 725-730 render with Markdown component
  - path: frontend/src/lib/components/ui/tool-call-bubble.svelte
    issue: Missing - needs to be created
  - path: frontend/src/lib/components/ui/index.ts
    issue: Needs to export ToolCallBubble component
