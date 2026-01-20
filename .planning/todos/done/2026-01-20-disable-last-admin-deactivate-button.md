---
created: 2026-01-20T17:27
title: Disable deactivate button for last admin in users panel
area: frontend/ui
files:
  - frontend/src/routes/(protected)/users/+page.svelte
  - backend/mlx_manager/routers/auth.py
---

## Problem

The users panel has 4 action buttons per user. Backend already prevents the last admin from deleting or disabling themselves (returns error to prevent system lockout with no admin to approve users). The UI partially reflects this:

- **Delete button:** Already disabled for the last admin ✓
- **Deactivate button:** Still active, shows error dialog with unhelpful "retry later" message when clicked ✗

This is a poor UX — users shouldn't be able to click a button that will always fail. The deactivate button should also be disabled when the logged-in user is the only admin.

## Solution

1. Disable the deactivate button for the current user if they are the only admin (same logic as delete button)
2. Optionally: Add tooltip explaining why the button is disabled ("Cannot deactivate: you are the only administrator")
3. Consider: API could return admin count to make frontend logic cleaner
