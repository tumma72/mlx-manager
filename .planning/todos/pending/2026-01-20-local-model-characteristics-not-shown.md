---
created: 2026-01-20T12:00
title: Local models don't show characteristics in model list
area: frontend
files:
  - frontend/src/routes/(protected)/models/+page.svelte
  - frontend/src/lib/components/models/*
  - backend/mlx_manager/routers/models.py
---

## Problem

In the model list view, model characteristics (size, quantization, downloads, etc.) are only shown when searching online on HuggingFace. Locally downloaded models don't display these characteristics.

## Context

User reported: "there is a bug in the model visualization in the model list, they only show the characteristics of the model when searching online on HuggingFace but they do not show them for locally downloaded models"

## Solution

1. Investigate what metadata is available for local models via `list_local_models` API
2. Ensure the frontend displays available characteristics for local models
3. May need to store/retrieve additional metadata when models are downloaded
