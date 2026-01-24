---
phase: 06-bug-fixes-stability
plan: 10
subsystem: chat-multimodal
tags: [chat, multimodal, text-attachments, file-handling, content-parts]
dependency_graph:
  requires: [05-02]
  provides: [text-file-reading]
  affects: [chat-ux]
tech_stack:
  added: []
  patterns: [dual-file-reader-strategy, content-type-branching]
key_files:
  created: []
  modified:
    - frontend/src/routes/(protected)/chat/+page.svelte
decisions:
  - id: text-vs-image-file-readers
    choice: FileReader.readAsText() for text files, readAsDataURL() for images/videos
    rationale: Models cannot interpret base64-encoded text files - they need plain text content
    alternatives: [send-all-as-base64, convert-server-side]
  - id: text-file-labeling
    choice: Prefix text content with "[File: filename]" header
    rationale: Model needs context about which file the content came from when multiple text files attached
    alternatives: [no-prefix, json-structure, xml-wrapper]
metrics:
  duration: 51s
  completed: 2026-01-24
---

# Phase 06 Plan 10: Text File Attachments Summary

**One-liner:** Text files sent as readable text content parts instead of unreadable base64 image_url

## Objective

Fix text file attachments (.txt, .py, .json, etc.) being sent to the model as base64 image_url (which the model cannot interpret) by reading them as plain text and including as text content parts.

## What Was Built

### Text File Reading Infrastructure

**1. readFileAsText() Helper (chat/+page.svelte)**
- Async function using FileReader.readAsText()
- Returns Promise<string> with file contents
- Parallel to existing encodeFileAsBase64()

**2. Content Type Branching (buildMessageContent)**
- Checks `attachment.type` to determine handling strategy
- Text files (`type: 'text'`): read as plain text, create `{type: 'text', text: '[File: name]\ncontent'}` part
- Images (`type: 'image'`): encode as base64, create `{type: 'image_url', image_url: {url: base64}}` part
- Videos (`type: 'video'`): encode as base64, create `{type: 'image_url', image_url: {url: base64}}` part (model extracts frames)

**3. File Name Context**
- Text content prefixed with `[File: filename]` header
- Allows model to distinguish between multiple text files in same message
- Preserves file identity in conversation context

## Technical Approach

**File Reading Strategy:**
```typescript
// Text files: read as UTF-8 strings
async function readFileAsText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsText(file);
  });
}

// Images/videos: read as base64 data URLs
async function encodeFileAsBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}
```

**Content Part Construction:**
```typescript
if (attachment.type === 'text') {
  const fileContent = await readFileAsText(attachment.file);
  parts.push({
    type: 'text',
    text: `[File: ${attachment.file.name}]\n${fileContent}`
  });
} else {
  const base64 = await encodeFileAsBase64(attachment.file);
  parts.push({
    type: 'image_url',
    image_url: { url: base64 }
  });
}
```

## Key Decisions

**1. Dual File Reader Strategy**
- **Decision:** Use FileReader.readAsText() for text files, readAsDataURL() for media
- **Rationale:** Models cannot interpret base64-encoded text content - they need raw UTF-8 strings
- **Impact:** Text file contents now readable by model, enabling code review, document Q&A, etc.

**2. File Name Labeling**
- **Decision:** Prefix each text file's content with `[File: filename]` header
- **Rationale:** When multiple text files attached, model needs to distinguish which content came from which file
- **Impact:** Preserves file identity in model's context window

**3. ContentPart Type Reuse**
- **Decision:** Use existing `ContentPart` type from types.ts (already supports text and image_url)
- **Rationale:** Type already designed for multimodal content - no changes needed
- **Impact:** Zero type modifications, immediate implementation

## Verification Results

**TypeScript Check:** ✅ Pass (0 errors, 0 warnings)
**ESLint:** ✅ Pass (0 errors in source, 2 warnings in coverage files only)
**grep readAsText:** ✅ Found at line 199
**grep attachment.type === 'text':** ✅ Found at lines 211, 697

## Deviations from Plan

None - plan executed exactly as written.

## Impact Assessment

**User-Facing Changes:**
- Text files (.txt, .py, .json, .xml, .yaml, .log, .sh, .sql, etc.) now sent as readable text
- Model can analyze code, answer questions about documents, review configurations
- No change to image/video handling

**Technical Debt:**
- None introduced

**Performance:**
- FileReader.readAsText() faster than base64 encoding for text files (no encoding overhead)
- Smaller payload for text files (raw text vs base64 encoding = ~25% smaller)

## Next Phase Readiness

**Blockers:** None

**Dependencies Satisfied:**
- Builds on Phase 05-02 (multimodal chat infrastructure)
- Attachment type detection (text vs image vs video) already in place
- ContentPart type already supports text parts

**Handoff Notes:**
- Text file handling now complete end-to-end
- Future enhancement: syntax highlighting in attachment preview
- Future enhancement: file size limits for text attachments

## Testing Recommendations

**Manual Testing:**
1. Attach .txt file → verify model reads content
2. Attach .py file → verify model can review code
3. Attach .json file → verify model can parse structure
4. Attach multiple text files → verify [File: name] headers distinguish them
5. Attach text + image → verify both rendered correctly
6. Attach text to text-only model → verify accepted
7. Attach image to text-only model → verify rejected

**UAT Scenarios:**
- Code review: attach Python file, ask for improvements
- Document Q&A: attach .txt file with specs, ask questions
- Config review: attach .json/.yaml file, ask for validation
- Multi-file context: attach multiple .py files, ask about interactions

## Lessons Learned

**What Worked Well:**
- Existing ContentPart type was perfectly suited - no type changes needed
- Dual FileReader strategy cleanly separates text vs binary handling
- File name prefix simple but effective for multi-file context

**What Could Be Improved:**
- Could add file size validation for text files (prevent 10MB text file crash)
- Could add character encoding detection (currently assumes UTF-8)
- Could add syntax highlighting in attachment preview thumbnails

**Knowledge Captured:**
- FileReader.readAsText() vs readAsDataURL() performance characteristics
- Models cannot interpret base64-encoded text (need raw strings)
- File name context critical for multi-file attachments
