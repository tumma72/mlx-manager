# MLX Model Manager - Build and Test Commands
# Run 'make help' for available commands

.PHONY: help install install-dev build test test-backend test-frontend \
        lint lint-backend lint-frontend format format-backend format-frontend \
        check check-backend check-frontend dev dev-offline check-offline clean \
        version bump-patch bump-minor bump-major release probe

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
	@echo "Diagnostics:"
	@echo "  make probe MODEL=<id>  Probe model thinking/tool capabilities"
	@echo ""
	@echo "CI/CD:"
	@echo "  make ci             Run full CI pipeline (lint, check, test)"
	@echo "  make pre-commit     Install and run pre-commit hooks"
	@echo ""
	@echo "Release:"
	@echo "  make version              Show current version"
	@echo "  make bump-patch           Bump patch version (1.0.3 -> 1.0.4)"
	@echo "  make bump-minor           Bump minor version (1.0.3 -> 1.1.0)"
	@echo "  make bump-major           Bump major version (1.0.3 -> 2.0.0)"
	@echo "  make release              Release current version"
	@echo "  make release VERSION=x.y.z  Set version and release"

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
	cd backend && MLX_MANAGER_DISABLE_TELEMETRY=true .venv/bin/pytest --cov=mlx_manager --cov-report=term-missing -v

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

# ============================================================================
# Release Management
# ============================================================================

# Single source of truth: VERSION file
# Can be overridden: make release VERSION=1.2.3
VERSION ?= $(shell cat VERSION 2>/dev/null || echo "0.0.0")

version:
	@echo "Current version: $(VERSION)"

bump-patch:
	@echo "Bumping patch version..."
	@CURRENT=$$(cat VERSION) && \
	NEW_VERSION=$$(echo $$CURRENT | awk -F. '{print $$1"."$$2"."$$3+1}') && \
	./scripts/sync_version.sh $$NEW_VERSION

bump-minor:
	@echo "Bumping minor version..."
	@CURRENT=$$(cat VERSION) && \
	NEW_VERSION=$$(echo $$CURRENT | awk -F. '{print $$1"."$$2+1".0"}') && \
	./scripts/sync_version.sh $$NEW_VERSION

bump-major:
	@echo "Bumping major version..."
	@CURRENT=$$(cat VERSION) && \
	NEW_VERSION=$$(echo $$CURRENT | awk -F. '{print $$1+1".0.0"}') && \
	./scripts/sync_version.sh $$NEW_VERSION

# Main release target
# Usage:
#   make release              # Release current VERSION
#   make release VERSION=1.2.3  # Set version and release
release:
	@echo ""
	@echo "═══════════════════════════════════════════════════════"
	@echo " MLX Manager Release Process"
	@echo "═══════════════════════════════════════════════════════"
	@echo ""
	@# Step 1: Sync version (in case VERSION was overridden)
	@./scripts/sync_version.sh $(VERSION)
	@echo ""
	@# Step 2: Run CI checks
	@echo "Running CI checks..."
	@$(MAKE) ci
	@echo ""
	@# Step 3: Commit version changes if any
	@if [ -n "$$(git status --porcelain -- VERSION backend/ frontend/package.json Formula/)" ]; then \
		echo "Committing version changes..."; \
		git add VERSION backend/pyproject.toml backend/mlx_manager/__init__.py frontend/package.json Formula/mlx-manager.rb; \
		git commit -m "chore: bump version to $(VERSION)"; \
	fi
	@# Step 4: Create tag and push
	@echo ""
	@echo "Creating release v$(VERSION)..."
	git tag -a v$(VERSION) -m "Release v$(VERSION)"
	git push origin main
	git push origin v$(VERSION)
	@echo ""
	@echo "═══════════════════════════════════════════════════════"
	@echo " ✓ Release v$(VERSION) created and pushed!"
	@echo "═══════════════════════════════════════════════════════"
	@echo ""
	@echo "Next steps:"
	@echo "  1. GitHub Actions will build and publish to PyPI"
	@echo "  2. After PyPI release, update Formula/mlx-manager.rb:"
	@echo "     - Update 'url' to point to v$(VERSION) tarball"
	@echo "     - Update 'sha256' with: shasum -a 256 <tarball>"
	@echo "  3. Create GitHub release notes from CHANGELOG.md"

# ============================================================================
# Diagnostics
# ============================================================================

# Probe a model's thinking/tool calling capabilities
# Usage: make probe MODEL=mlx-community/Qwen3-0.6B-4bit-DWQ
#        make probe-all
PROBE_ARGS ?=
probe:
ifndef MODEL
	@echo "Usage: make probe MODEL=<model-id>"
	@echo "  Example: make probe MODEL=mlx-community/Qwen3-0.6B-4bit-DWQ"
	@echo "  Or: make probe-all"
else
	cd backend && .venv/bin/python -m mlx_manager.cli probe $(MODEL) $(PROBE_ARGS)
endif

probe-all:
	cd backend && .venv/bin/python -m mlx_manager.cli probe --all
