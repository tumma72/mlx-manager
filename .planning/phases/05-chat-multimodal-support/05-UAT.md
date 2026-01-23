---
status: complete
phase: 05-chat-multimodal-support
source: [05-01-SUMMARY.md, 05-02-SUMMARY.md, 05-03-SUMMARY.md, 05-04-SUMMARY.md]
started: 2026-01-23T12:00:00Z
updated: 2026-01-23T12:10:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Attachment Button Visibility
expected: Attachment button always visible regardless of model type. Text models allow text-based formats; multimodal models also allow images/videos.
result: issue
reported: "Button should always appear. Text models allow text-based formats only. Multimodal models allow images and videos. Max 3 attachments always."
severity: major

### 2. File Upload via Button
expected: Clicking the paperclip button opens a file picker. For text-only models, only text formats selectable. For multimodal, images/videos also selectable. Selecting files shows thumbnail previews above input.
result: issue
reported: "I can't see any attachment button in the text input of the chat"
severity: major

### 3. Drag-Drop Upload
expected: Dragging an image over the chat area shows a visual ring border. Dropping the file adds a thumbnail preview above the input.
result: issue
reported: "Images can be dropped at all times, but there is still no button for attachment, and text files can't be dropped at all"
severity: major

### 4. Attachment Validation (Max 3)
expected: Attempting to add a 4th attachment shows an error message. Only 3 attachments remain.
result: pass

### 5. Attachment Removal
expected: Hovering over a thumbnail shows an X button. Clicking it removes that attachment from the preview row.
result: pass

### 6. Streaming Text Response
expected: Sending a text message to a running server streams the response character by character in real-time (not all at once after completion).
result: pass

### 7. Thinking Model Display
expected: When chatting with a reasoning model (e.g., Qwen3), a ThinkingBubble appears showing live thinking content with a spinner. When done, it collapses and shows "Thought for Xs" with the duration.
result: pass

### 8. Multimodal Message Sending
expected: Attaching an image and typing a question, then sending, delivers both the image and text to the model. The model responds acknowledging the image content.
result: skipped
reason: Upstream mlx-openai-server v1.5.0 regression prevents loading VLM models (VisionConfig missing args). Known issue, not our code.

### 9. Error Display (Connection Failed)
expected: Sending a message when the server is not running shows an inline error in the chat area (not a banner) with a collapsible details section and copy-to-clipboard button.
result: pass

### 10. Attachments Clear After Send
expected: After sending a message with attachments, the thumbnail preview row clears and the input area resets.
result: pass

## Summary

total: 10
passed: 6
issues: 3
pending: 0
skipped: 1

## Gaps

- truth: "Attachment button always visible regardless of model type, with format filtering based on model capabilities"
  status: failed
  reason: "User reported: Button should always appear. Text models allow text-based formats only. Multimodal models allow images and videos. Max 3 attachments always."
  severity: major
  test: 1
  root_cause: ""
  artifacts: []
  missing: []
  debug_session: ""
- truth: "File picker opens from attachment button with format filtering based on model type"
  status: failed
  reason: "User reported: I can't see any attachment button in the text input of the chat"
  severity: major
  test: 2
  root_cause: ""
  artifacts: []
  missing: []
  debug_session: ""
- truth: "Drag-drop supports both images and text files based on model capabilities"
  status: failed
  reason: "User reported: Images can be dropped at all times, but there is still no button for attachment, and text files can't be dropped at all"
  severity: major
  test: 3
  root_cause: ""
  artifacts: []
  missing: []
  debug_session: ""
