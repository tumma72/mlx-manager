# Contributing to MLX Model Manager

Thank you for your interest in contributing to MLX Model Manager! This guide will help you get started with the development workflow.

## Development Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- macOS (Apple Silicon recommended for testing MLX features)

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Install Playwright browsers (for E2E tests)
npx playwright install
```

## Quality Standards

This project enforces quality through automated checks. All code must pass these checks before merging.

### Backend Quality Commands

```bash
cd backend
source .venv/bin/activate

# Linting (auto-fixes issues)
ruff check . --fix

# Formatting
ruff format .

# Type checking
mypy mlx_manager

# Run tests
pytest -v

# Run tests with coverage
pytest --cov=mlx_manager --cov-report=term-missing

# Run specific test file
pytest tests/test_profiles.py -v

# Run specific test
pytest tests/test_profiles.py::test_create_profile -v
```

### Frontend Quality Commands

```bash
cd frontend

# Type checking
npm run check

# Linting
npm run lint

# Formatting
npm run format

# Unit tests
npm run test

# Unit tests with watch mode
npm run test:watch

# E2E tests (requires dev server running)
npm run test:e2e
```

## Pre-commit Hooks

Pre-commit hooks automatically run on every commit to ensure code quality:

- **On commit**: Linting, formatting, type checking
- **On push**: Full test suites

To manually run all hooks:

```bash
pre-commit run --all-files
```

To skip hooks (not recommended):

```bash
git commit --no-verify
```

## Testing Guidelines

### Backend Tests

- Tests are located in `backend/tests/`
- Use pytest with async support (`pytest-asyncio`)
- Use fixtures from `conftest.py` for common setup
- Mock external services (HuggingFace Hub, subprocesses, launchd)

**Test Categories:**
- `test_profiles.py` - Profile CRUD API tests
- `test_models.py` - Model search/download API tests
- `test_servers.py` - Server lifecycle API tests
- `test_system.py` - System info and launchd API tests
- `test_services_*.py` - Unit tests for service classes

**Writing Tests:**

```python
@pytest.mark.asyncio
async def test_example(client, sample_profile_data):
    """Test description."""
    # Arrange
    response = await client.post("/api/profiles", json=sample_profile_data)

    # Assert
    assert response.status_code == 201
    assert response.json()["name"] == sample_profile_data["name"]
```

### Frontend Tests

- Unit tests use Vitest and Testing Library
- Tests are colocated with source: `src/**/*.test.ts`
- E2E tests use Playwright in `e2e/`

**Writing Tests:**

```typescript
import { describe, it, expect, vi } from 'vitest';

describe('MyComponent', () => {
  it('does something', () => {
    // Test implementation
  });
});
```

## Code Style

### Python

- Follow PEP 8 (enforced by Ruff)
- Use type hints for all function signatures
- Docstrings for public functions and classes
- Max line length: 100 characters

### TypeScript/Svelte

- Use TypeScript strict mode
- Follow Svelte 5 runes patterns (`$state`, `$derived`, `$effect`)
- Use Prettier for formatting

## Pull Request Process

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make changes** with atomic commits

3. **Ensure quality checks pass**:
   ```bash
   # Backend
   cd backend && ruff check . && pytest

   # Frontend
   cd frontend && npm run check && npm run test
   ```

4. **Push and create PR**:
   ```bash
   git push -u origin feature/my-feature
   ```

5. **Address review feedback**

## Project Structure

```
mlx-model-manager/
├── backend/
│   ├── mlx_manager/
│   │   ├── routers/      # FastAPI route handlers
│   │   ├── services/     # Business logic
│   │   ├── models.py     # SQLModel entities
│   │   ├── database.py   # Database setup
│   │   ├── cli.py        # CLI entry point
│   │   └── menubar.py    # macOS menubar app
│   └── tests/            # pytest tests
├── frontend/
│   ├── src/
│   │   ├── lib/
│   │   │   ├── api/      # API client
│   │   │   ├── components/
│   │   │   └── stores/   # Svelte stores
│   │   └── routes/       # SvelteKit pages
│   └── e2e/              # Playwright tests
└── docs/
```

## Current Coverage

| Area | Coverage |
|------|----------|
| Backend Overall | 67% |
| Routers | 85%+ |
| Services | 80%+ |
| Frontend API/Utils | 100% |

## Getting Help

- Check existing issues on GitHub
- Review the codebase documentation in `docs/ARCHITECTURE.md`
- Ask questions via GitHub Discussions
