#!/bin/bash
# MLX Model Manager Development Script
# Starts both backend and frontend in development mode
# Supports offline mode with smart dependency handling

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Flags
OFFLINE=false
SKIP_DEPS=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --offline)
            OFFLINE=true
            shift
            ;;
        --skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Start MLX Model Manager development environment."
            echo ""
            echo "Options:"
            echo "  --offline     Force offline mode (skip network operations)"
            echo "  --skip-deps   Skip dependency installation"
            echo "  -h, --help    Show this help"
            echo ""
            echo "Environment variables:"
            echo "  MLX_MANAGER_OFFLINE_MODE=true  Force offline mode"
            echo ""
            echo "Examples:"
            echo "  $0                    # Normal mode with auto network detection"
            echo "  $0 --offline          # Force offline mode"
            echo "  $0 --skip-deps        # Skip dependency checks (faster startup)"
            echo "  $0 --offline --skip-deps  # Fastest startup for offline work"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Run '$0 --help' for usage information"
            exit 1
            ;;
    esac
done

# Network detection (fast DNS check, 2s timeout)
detect_network() {
    # Check environment variable first
    if [ "$MLX_MANAGER_OFFLINE_MODE" = "true" ]; then
        OFFLINE=true
        return
    fi

    # Skip if already in offline mode
    if [ "$OFFLINE" = true ]; then
        return
    fi

    echo -n "Checking network connectivity... "

    # Try DNS resolution (faster than ping)
    if command -v host &> /dev/null; then
        if ! timeout 2 host -W 2 pypi.org &> /dev/null; then
            OFFLINE=true
            echo -e "${YELLOW}offline${NC}"
        else
            echo -e "${GREEN}online${NC}"
        fi
    elif ! ping -c 1 -W 2 8.8.8.8 &> /dev/null 2>&1; then
        OFFLINE=true
        echo -e "${YELLOW}offline${NC}"
    else
        echo -e "${GREEN}online${NC}"
    fi
}

# Check if Python dependencies are installed
check_python_deps() {
    [ -d "$ROOT_DIR/backend/.venv" ] || return 1
    source "$ROOT_DIR/backend/.venv/bin/activate"
    python -c "import fastapi, uvicorn, sqlmodel, httpx" 2>/dev/null || return 1
    return 0
}

# Check if Node dependencies are installed
check_node_deps() {
    [ -d "$ROOT_DIR/frontend/node_modules" ] || return 1
    [ -d "$ROOT_DIR/frontend/node_modules/svelte" ] || return 1
    [ -d "$ROOT_DIR/frontend/node_modules/vite" ] || return 1
    return 0
}

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down...${NC}"
    jobs -p | xargs -r kill 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# ============================================================================
# Prerequisite Checks
# ============================================================================

echo -e "${GREEN}Starting MLX Model Manager Development Environment${NC}"
echo ""

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

# Check for Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}Error: Node.js is not installed${NC}"
    exit 1
fi

# Check for npm
if ! command -v npm &> /dev/null; then
    echo -e "${RED}Error: npm is not installed${NC}"
    exit 1
fi

# ============================================================================
# Network Detection
# ============================================================================

if [ "$SKIP_DEPS" = false ]; then
    detect_network
fi

if [ "$OFFLINE" = true ]; then
    echo -e "${BLUE}Running in offline mode${NC}"
    export MLX_MANAGER_OFFLINE_MODE=true
fi

# ============================================================================
# Backend Setup
# ============================================================================

echo ""
echo -e "${GREEN}Setting up backend...${NC}"
cd "$ROOT_DIR/backend"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    if [ "$OFFLINE" = true ]; then
        echo -e "${RED}Error: Backend virtual environment not found${NC}"
        echo "Please run 'make install-dev' while online first"
        exit 1
    fi
    echo "Creating Python virtual environment..."
    if command -v uv &> /dev/null; then
        uv venv
    else
        python3 -m venv .venv
    fi
fi

# Activate virtual environment
source .venv/bin/activate

# Handle dependencies
if [ "$SKIP_DEPS" = false ]; then
    if check_python_deps; then
        echo -e "Python dependencies: ${GREEN}OK${NC}"
    else
        if [ "$OFFLINE" = true ]; then
            echo -e "${RED}Error: Python dependencies incomplete and offline${NC}"
            echo "Please run 'make install-dev' while online first"
            exit 1
        fi
        echo "Installing Python dependencies..."
        if command -v uv &> /dev/null; then
            uv pip install -q -e ".[dev]"
        else
            pip install -q -e ".[dev]"
        fi
    fi
else
    echo -e "Skipping dependency check ${YELLOW}(--skip-deps)${NC}"
fi

# Start uvicorn in background
echo "Starting backend server..."
uvicorn mlx_manager.main:app --reload --host 127.0.0.1 --port 8080 &
BACKEND_PID=$!

# Wait for backend to start
echo "Waiting for backend to start..."
sleep 3

# ============================================================================
# Frontend Setup
# ============================================================================

echo ""
echo -e "${GREEN}Setting up frontend...${NC}"
cd "$ROOT_DIR/frontend"

# Handle dependencies
if [ "$SKIP_DEPS" = false ]; then
    if check_node_deps; then
        echo -e "Node dependencies: ${GREEN}OK${NC}"
    else
        if [ "$OFFLINE" = true ]; then
            echo -e "${RED}Error: Node dependencies incomplete and offline${NC}"
            echo "Please run 'make install-dev' while online first"
            exit 1
        fi
        echo "Installing Node dependencies..."
        npm install
    fi
else
    echo -e "Skipping dependency check ${YELLOW}(--skip-deps)${NC}"
fi

# Ensure SvelteKit generated files exist (tsconfig, types, etc.)
if [ ! -f ".svelte-kit/tsconfig.json" ]; then
    echo "Generating SvelteKit files..."
    npx svelte-kit sync
fi

# Start Vite dev server in background
echo "Starting frontend server..."
npm run dev &
FRONTEND_PID=$!

# ============================================================================
# Ready
# ============================================================================

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}MLX Model Manager is running!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Frontend: ${YELLOW}http://localhost:5173${NC}"
echo -e "Backend:  ${YELLOW}http://localhost:8080${NC}"
echo -e "API Docs: ${YELLOW}http://localhost:8080/docs${NC}"
echo ""
if [ "$OFFLINE" = true ]; then
    echo -e "Mode:     ${BLUE}Offline${NC}"
    echo ""
fi
echo -e "Press ${YELLOW}Ctrl+C${NC} to stop all services"
echo ""

# Wait for processes
wait
