#!/bin/bash

# MLX Model Manager Build Script
# Creates production builds with embedded frontend

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
echo -e "${GREEN}Step 1: Building frontend...${NC}"
cd "$ROOT_DIR/frontend"

# Install dependencies
npm ci

# Build for production
npm run build

echo -e "${GREEN}Frontend build complete${NC}"

# Embed frontend in backend
echo -e "${GREEN}Step 2: Embedding frontend in backend...${NC}"
cd "$ROOT_DIR"

# Remove old static files
rm -rf "$ROOT_DIR/backend/mlx_manager/static"

# Copy frontend build to backend
cp -r "$ROOT_DIR/frontend/build" "$ROOT_DIR/backend/mlx_manager/static"

echo -e "${GREEN}Frontend embedded in backend/mlx_manager/static/${NC}"

# Build Python package
echo -e "${GREEN}Step 3: Building Python package...${NC}"
cd "$ROOT_DIR/backend"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# Activate and install build tools
source .venv/bin/activate
pip install -q build

# Build wheel and sdist
python -m build

echo -e "${GREEN}Python package built${NC}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Build complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Artifacts:"
echo "  - Wheel: backend/dist/mlx_manager-*.whl"
echo "  - Source: backend/dist/mlx-manager-*.tar.gz"
echo ""
echo "To install locally:"
echo "  pip install backend/dist/mlx_manager-*.whl"
echo ""
echo "To run:"
echo "  mlx-manager serve"
echo "  mlx-manager menubar"
echo ""
