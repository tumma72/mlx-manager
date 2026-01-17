# MLX Model Manager - Build and Test Commands
# Run 'make help' for available commands

.PHONY: help install install-dev build test test-backend test-frontend \
        lint lint-backend lint-frontend format format-backend format-frontend \
        check check-backend check-frontend dev dev-offline check-offline clean

# Default target
help:
	@echo "MLX Model Manager - Build and Test Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install all dependencies (production)"
	@echo "  make install-dev    Install all dependencies (development)"
	@echo ""
	@echo "Build:"
	@echo "  make build          Build both backend and frontend for production"
	@echo "  make build-backend  Build backend wheel package"
	@echo "  make build-frontend Build frontend static files"
	@echo ""
	@echo "Test:"
	@echo "  make test           Run all tests (backend + frontend)"
	@echo "  make test-backend   Run backend tests with coverage"
	@echo "  make test-frontend  Run frontend unit tests"
	@echo "  make test-e2e       Run frontend E2E tests (requires dev server)"
	@echo ""
	@echo "Quality:"
	@echo "  make lint           Run all linters"
	@echo "  make lint-backend   Run Python linter (ruff)"
	@echo "  make lint-frontend  Run frontend linter (eslint)"
	@echo "  make format         Format all code"
	@echo "  make format-backend Format Python code"
	@echo "  make format-frontend Format frontend code"
	@echo "  make check          Run all type checks"
	@echo "  make check-backend  Run Python type checker (mypy)"
	@echo "  make check-frontend Run Svelte type checker"
	@echo ""
	@echo "Development:"
	@echo "  make dev            Start both backend and frontend in dev mode"
	@echo "  make dev-offline    Start development in offline mode"
	@echo "  make check-offline  Check if system is ready for offline development"
	@echo "  make clean          Remove build artifacts and caches"
	@echo ""
	@echo "CI/CD:"
	@echo "  make ci             Run full CI pipeline (lint, check, test)"
	@echo "  make pre-commit     Install and run pre-commit hooks"

# ============================================================================
# Setup
# ============================================================================

install:
	@echo "Installing backend dependencies..."
	cd backend && if command -v uv >/dev/null 2>&1; then uv pip install -e .; else pip install -e .; fi
	@echo "Installing frontend dependencies..."
	cd frontend && npm install
	@echo "✓ Installation complete"

install-dev:
	@echo "Installing backend development dependencies..."
	cd backend && if command -v uv >/dev/null 2>&1; then uv pip install -e ".[dev]"; else pip install -e ".[dev]"; fi
	@echo "Installing frontend dependencies..."
	cd frontend && npm install
	@echo "Installing Playwright browsers..."
	cd frontend && npx playwright install
	@echo "Installing pre-commit hooks..."
	pre-commit install || echo "pre-commit not installed, skipping hooks setup"
	@echo "✓ Development installation complete"

# ============================================================================
# Build
# ============================================================================

build: build-frontend build-backend
	@echo "✓ Build complete"

build-backend:
	@echo "Building backend wheel..."
	cd backend && if command -v uv >/dev/null 2>&1; then uv pip install build; else .venv/bin/pip install build; fi && .venv/bin/python -m build
	@echo "✓ Backend build complete: backend/dist/"

build-frontend:
	@echo "Building frontend..."
	cd frontend && npm run build
	@echo "✓ Frontend build complete: frontend/build/"

# ============================================================================
# Test
# ============================================================================

test: test-backend test-frontend
	@./scripts/coverage_summary.sh
	@echo "✓ All tests passed"

test-backend:
	@echo "Running backend tests (95% coverage required)..."
	cd backend && .venv/bin/pytest --cov=mlx_manager --cov-report=term-missing --cov-fail-under=95 -v

test-frontend:
	@echo "Running frontend unit tests (95% coverage required)..."
	cd frontend && npm run test:coverage

test-e2e:
	@echo "Running frontend E2E tests..."
	@echo "Note: Requires dev server running (make dev)"
	cd frontend && npm run test:e2e

# ============================================================================
# Quality - Linting
# ============================================================================

lint: lint-backend lint-frontend
	@echo "✓ All linting passed"

lint-backend:
	@echo "Linting backend..."
	cd backend && .venv/bin/ruff check .

lint-frontend:
	@echo "Linting frontend..."
	cd frontend && npm run lint

# ============================================================================
# Quality - Formatting
# ============================================================================

format: format-backend format-frontend
	@echo "✓ All formatting complete"

format-backend:
	@echo "Formatting backend..."
	cd backend && .venv/bin/ruff format .

format-frontend:
	@echo "Formatting frontend..."
	cd frontend && npm run format

# ============================================================================
# Quality - Type Checking
# ============================================================================

check: check-backend check-frontend
	@echo "✓ All type checks passed"

check-backend:
	@echo "Type checking backend..."
	cd backend && .venv/bin/mypy mlx_manager

check-frontend:
	@echo "Type checking frontend..."
	cd frontend && npm run check

# ============================================================================
# Development
# ============================================================================

dev:
	@echo "Starting development servers..."
	./scripts/dev.sh

dev-offline: ## Start development servers in offline mode
	@echo "Starting development in offline mode..."
	MLX_MANAGER_OFFLINE_MODE=true ./scripts/dev.sh --offline

check-offline: ## Check if system is ready for offline development
	@./scripts/check_deps.sh

# ============================================================================
# CI/CD
# ============================================================================

ci: lint check test
	@echo "✓ CI pipeline complete"

pre-commit:
	@echo "Running pre-commit hooks..."
	pre-commit install
	pre-commit run --all-files

# ============================================================================
# Cleanup
# ============================================================================

clean:
	@echo "Cleaning build artifacts..."
	rm -rf backend/dist backend/build backend/*.egg-info
	rm -rf frontend/build frontend/.svelte-kit
	rm -rf backend/.pytest_cache backend/.mypy_cache backend/.ruff_cache
	rm -rf frontend/node_modules/.cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ Clean complete"
