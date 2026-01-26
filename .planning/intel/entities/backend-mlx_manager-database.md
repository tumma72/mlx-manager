---
path: /Users/atomasini/Development/mlx-manager/backend/mlx_manager/database.py
type: service
updated: 2026-01-21
status: active
---

# database.py

## Purpose

Database setup and session management for async SQLite via aiosqlite. Creates the async engine and session factory, handles database initialization with table creation, performs schema migrations for adding columns to existing databases, recovers incomplete downloads on startup, and provides session dependency injection for FastAPI routes.

## Exports

- `engine` - SQLAlchemy async engine
- `async_session` - Async session factory
- `migrate_schema()` - Add missing columns to existing tables
- `recover_incomplete_downloads() -> list[tuple[int, str]]` - Resume interrupted downloads
- `init_db()` - Initialize database and default settings
- `get_session()` - Async context manager for sessions
- `get_db()` - FastAPI dependency for route injection

## Dependencies

- [[backend-mlx_manager-config]] - Database path settings
- [[backend-mlx_manager-models]] - SQLModel table definitions
- sqlalchemy - Async engine and session
- aiosqlite - Async SQLite driver
- sqlmodel - ORM

## Used By

TBD

## Notes

Migration system adds new columns to existing tables since SQLite doesn't support them in CREATE TABLE IF NOT EXISTS. Default settings are inserted on first run.
