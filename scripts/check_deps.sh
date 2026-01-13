#!/bin/bash
# Check if MLX Manager is ready for offline development

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

EXIT_CODE=0

echo "Checking MLX Manager offline readiness..."
echo ""

# ============================================================================
# System Prerequisites
# ============================================================================

echo "System prerequisites:"

echo -n "  Python 3: "
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    echo -e "${GREEN}OK${NC} (v${PYTHON_VERSION})"
else
    echo -e "${RED}MISSING${NC}"
    EXIT_CODE=1
fi

echo -n "  Node.js: "
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version 2>&1)
    echo -e "${GREEN}OK${NC} (${NODE_VERSION})"
else
    echo -e "${RED}MISSING${NC}"
    EXIT_CODE=1
fi

echo -n "  npm: "
if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version 2>&1)
    echo -e "${GREEN}OK${NC} (v${NPM_VERSION})"
else
    echo -e "${RED}MISSING${NC}"
    EXIT_CODE=1
fi

echo ""

# ============================================================================
# Backend Dependencies
# ============================================================================

echo "Backend dependencies:"

echo -n "  Virtual environment: "
if [ -d "$ROOT_DIR/backend/.venv" ]; then
    echo -e "${GREEN}OK${NC}"

    # Check Python packages
    echo -n "  Core packages: "
    source "$ROOT_DIR/backend/.venv/bin/activate"
    if python -c "import fastapi, uvicorn, sqlmodel, httpx" 2>/dev/null; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}INCOMPLETE${NC}"
        echo "    Missing: Some core packages (fastapi, uvicorn, sqlmodel, httpx)"
        EXIT_CODE=1
    fi

    echo -n "  HuggingFace Hub: "
    if python -c "import huggingface_hub" 2>/dev/null; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${YELLOW}MISSING${NC} (optional for offline)"
    fi

    echo -n "  Dev tools: "
    if python -c "import pytest, ruff, mypy" 2>/dev/null; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${YELLOW}INCOMPLETE${NC} (optional)"
    fi
else
    echo -e "${RED}MISSING${NC}"
    echo "    Virtual environment not found at backend/.venv"
    EXIT_CODE=1
fi

echo ""

# ============================================================================
# Frontend Dependencies
# ============================================================================

echo "Frontend dependencies:"

echo -n "  node_modules: "
if [ -d "$ROOT_DIR/frontend/node_modules" ]; then
    echo -e "${GREEN}OK${NC}"

    echo -n "  Svelte: "
    if [ -d "$ROOT_DIR/frontend/node_modules/svelte" ]; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}MISSING${NC}"
        EXIT_CODE=1
    fi

    echo -n "  Vite: "
    if [ -d "$ROOT_DIR/frontend/node_modules/vite" ]; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}MISSING${NC}"
        EXIT_CODE=1
    fi

    echo -n "  SvelteKit: "
    if [ -d "$ROOT_DIR/frontend/node_modules/@sveltejs/kit" ]; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}MISSING${NC}"
        EXIT_CODE=1
    fi
else
    echo -e "${RED}MISSING${NC}"
    echo "    node_modules not found at frontend/node_modules"
    EXIT_CODE=1
fi

echo ""

# ============================================================================
# Summary
# ============================================================================

echo "----------------------------------------"
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}System ready for offline development${NC}"
    echo ""
    echo "Run development server:"
    echo "  ./scripts/dev.sh --offline"
    echo ""
    echo "Or use Make:"
    echo "  make dev-offline"
else
    echo -e "${RED}Dependencies missing${NC}"
    echo ""
    echo "To install dependencies while online:"
    echo "  make install-dev"
    echo ""
    echo "Or manually:"
    echo "  cd backend && pip install -e \".[dev]\""
    echo "  cd frontend && npm install"
fi

exit $EXIT_CODE
