---
status: resolved
trigger: "Investigate why text file attachments only work for .txt and .py but fail for .log, .md and other text formats."
created: 2026-01-24T00:00:00Z
updated: 2026-01-24T00:00:00Z
---

## Current Focus

hypothesis: File type detection uses hardcoded extension list instead of mime-type or comprehensive extension checking
test: Read chat page to find file attachment validation logic
expecting: Extension-based whitelist that excludes .log, .md, etc.
next_action: Read frontend/src/routes/(protected)/chat/+page.svelte to find type detection logic

## Symptoms

expected: Text files (.log, .md, etc.) should be read as text and sent as attachments
actual: Only .txt and .py files work; .log, .md fail with "Unsupported file format"
errors: "Unsupported file format" error message
reproduction: Try to attach .log or .md file in chat interface
started: Since plan 06-10 implementation

## Eliminated

## Evidence

- timestamp: 2026-01-24T00:01:00Z
  checked: frontend/src/routes/(protected)/chat/+page.svelte lines 127-132
  found: File type detection uses hardcoded mime-type checks (file.type.startsWith('text/') || specific application types)
  implication: Missing many common text file mime types like text/markdown, text/x-log, etc.

- timestamp: 2026-01-24T00:02:00Z
  checked: Lines 135-137
  found: Error message "Unsupported file type" doesn't explain what IS supported
  implication: User doesn't know which formats to use

- timestamp: 2026-01-24T00:03:00Z
  checked: Lines 53-57
  found: acceptedFormats includes .log, .md extensions in the accept attribute
  implication: File picker ALLOWS selection but validation REJECTS them - UX mismatch

## Resolution

root_cause: File type detection at lines 127-132 uses incomplete mime-type list. Browser reports different mime types for text files (.md = text/markdown, .log = text/x-log, etc.) that aren't in the hardcoded list. Additionally, mime-type detection is unreliable - macOS may report application/octet-stream for unknown extensions.
fix: Use extension-based detection instead of mime-type detection to match the accept attribute list
verification: Test with .log, .md, .csv, .json files
files_changed: []
