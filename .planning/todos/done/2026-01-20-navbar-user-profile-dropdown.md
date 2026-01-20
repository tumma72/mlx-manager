---
created: 2026-01-20T17:25
title: Navbar user profile dropdown and extended user model
area: frontend/ui
files:
  - frontend/src/lib/components/Navbar.svelte
  - backend/mlx_manager/models.py
  - backend/mlx_manager/routers/auth.py
---

## Problem

The navbar is too crowded when displaying user info. Currently shows email + logout button with icon, which can overlap with long email addresses. Additionally, the user model was originally discussed to include username and full name fields for a proper user profile, allowing login with either username or email — this was never implemented.

**Two related issues:**
1. **UI crowding** — Email + logout in navbar causes layout issues with long emails
2. **Missing user profile fields** — No username/full_name fields on User model, no profile editing

## Solution

1. **Navbar refactor:**
   - Replace email + logout with a single user profile icon (right side of navbar)
   - Add dropdown menu on click showing:
     - User's email address (and name if added)
     - Logout button
   - Possibly link to profile settings page

2. **User model extension (optional/future):**
   - Add `username` and `full_name` fields to User model
   - Allow login with email OR username
   - Add profile settings page for users to edit their info
