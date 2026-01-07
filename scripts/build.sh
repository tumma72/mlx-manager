#!/bin/bash

# MLX Model Manager Build Script
# Creates production builds of frontend and backend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building MLX Model Manager${NC}"
echo ""

# Build frontend
echo -e "${GREEN}Building frontend...${NC}"
cd "$ROOT_DIR/frontend"

# Install dependencies
npm install

# Build for production
npm run build

echo -e "${GREEN}Frontend build complete: frontend/build/${NC}"

# Package backend
echo -e "${GREEN}Preparing backend...${NC}"
cd "$ROOT_DIR/backend"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# Install dependencies
source .venv/bin/activate
pip install -e .

echo -e "${GREEN}Backend ready${NC}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Build complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "To run in production mode:"
echo ""
echo "  # Start backend"
echo "  cd backend && source .venv/bin/activate"
echo "  uvicorn app.main:app --host 127.0.0.1 --port 8080"
echo ""
echo "  # Serve frontend (use any static file server)"
echo "  cd frontend/build && python -m http.server 5173"
echo ""
