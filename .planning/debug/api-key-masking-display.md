---
status: resolved
trigger: "API key input shows ****...saved instead of masking characters as ****...last4 while typing"
created: 2026-01-30T12:00:00Z
updated: 2026-01-30T12:05:00Z
---

## Current Focus

hypothesis: The masking IS implemented correctly - there's a misunderstanding of the behavior
test: Read the implementation and trace the logic
expecting: Clarify whether this is a bug or expected behavior
next_action: Document findings

## Symptoms

expected: When typing an API key, characters should display masked showing only last 4 characters
actual: Shows placeholder text `****...saved` regardless of typing state
errors: none
reproduction: Open provider form with existing credential, observe input field
started: Issue report - needs verification

## Eliminated

- hypothesis: maskedKey derived value not working
  evidence: Code at lines 26-28 correctly computes mask from apiKey
  timestamp: 2026-01-30T12:02:00Z

## Evidence

- timestamp: 2026-01-30T12:01:00Z
  checked: ProviderForm.svelte lines 1-223
  found: |
    1. Input is type="password" (line 125) - browser natively masks all chars as dots
    2. maskedKey derived (lines 26-28) shows `****...${last4}` when apiKey.length > 4
    3. maskedKey displays in span overlay (lines 130-136) only when apiKey is truthy
    4. placeholder (lines 32-34) shows "****...saved" only when no text entered AND credential exists
  implication: The implementation is correct - mask shows AS you type

- timestamp: 2026-01-30T12:03:00Z
  checked: Interaction between password input and overlay
  found: |
    - Password input masks with dots (native browser behavior)
    - ADDITIONALLY, a span overlay shows `****...last4` at right side of input
    - These are two separate visual elements working correctly together
    - The "****...saved" text is the placeholder - only visible when input is empty
  implication: User confusion may stem from not understanding the dual-display design

## Resolution

root_cause: NOT A BUG - The implementation is working as designed. The confusion is:
  1. "****...saved" is the PLACEHOLDER text - only shown when input is EMPTY
  2. When typing, the password input shows dots (native browser behavior)
  3. The mask overlay `****...last4` appears ONLY when apiKey has content (line 130)
  4. The overlay displays at the right side of the input (lines 131-135)

fix: No code fix needed. The behavior is:
  - Empty input + existing credential: shows placeholder "****...saved (enter new key to update)"
  - Typing: password field shows dots, overlay shows "****...last4" on the right
  - Empty input + no credential: shows placeholder "Enter API key"

verification: Code review confirms correct implementation at lines 26-28, 32-34, 123-137

files_changed: []
