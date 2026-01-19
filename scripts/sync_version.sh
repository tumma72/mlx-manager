#!/bin/bash
# Sync version from VERSION file to all project files
# Usage: ./scripts/sync_version.sh [version]
# If no version provided, reads from VERSION file

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Get version from argument or VERSION file
if [ -n "$1" ]; then
    VERSION="$1"
    echo "$VERSION" > "$PROJECT_ROOT/VERSION"
else
    VERSION=$(cat "$PROJECT_ROOT/VERSION" | tr -d '[:space:]')
fi

echo "Syncing version: $VERSION"
echo ""

# 1. Update backend pyproject.toml
PYPROJECT="$PROJECT_ROOT/backend/pyproject.toml"
if [ -f "$PYPROJECT" ]; then
    # Use Python script with tomlkit if available, fallback to regex
    if python3 "$SCRIPT_DIR/update_pyproject_version.py" "$PYPROJECT" "$VERSION" 2>/dev/null; then
        echo "✓ Updated backend/pyproject.toml"
    else
        # Fallback: direct sed replacement (works for simple version = "x.y.z" lines)
        if sed -i '' "s/^version = \"[^\"]*\"/version = \"$VERSION\"/" "$PYPROJECT" 2>/dev/null; then
            echo "✓ Updated backend/pyproject.toml (sed fallback)"
        else
            echo "⚠ Could not update backend/pyproject.toml"
        fi
    fi
fi

# 2. Update backend __init__.py (runtime version access)
INIT_FILE="$PROJECT_ROOT/backend/mlx_manager/__init__.py"
if [ -f "$INIT_FILE" ]; then
    if grep -q "__version__" "$INIT_FILE"; then
        sed -i '' "s/__version__ = \"[^\"]*\"/__version__ = \"$VERSION\"/" "$INIT_FILE"
    else
        echo "__version__ = \"$VERSION\"" >> "$INIT_FILE"
    fi
    echo "✓ Updated backend/mlx_manager/__init__.py"
fi

# 3. Update frontend package.json
PACKAGE_JSON="$PROJECT_ROOT/frontend/package.json"
if [ -f "$PACKAGE_JSON" ]; then
    node -e "
        const fs = require('fs');
        const pkg = JSON.parse(fs.readFileSync('$PACKAGE_JSON', 'utf8'));
        pkg.version = '$VERSION';
        fs.writeFileSync('$PACKAGE_JSON', JSON.stringify(pkg, null, 2) + '\n');
    "
    echo "✓ Updated frontend/package.json"
fi

# 4. Update Homebrew formula version
FORMULA="$PROJECT_ROOT/Formula/mlx-manager.rb"
if [ -f "$FORMULA" ]; then
    sed -i '' "s/version \"[^\"]*\"/version \"$VERSION\"/" "$FORMULA"
    echo "✓ Updated Formula/mlx-manager.rb"
fi

echo ""
echo "═══════════════════════════════════════════"
echo " Version synced to $VERSION"
echo "═══════════════════════════════════════════"
echo ""
echo "Files updated:"
echo "  • VERSION"
echo "  • backend/pyproject.toml"
echo "  • backend/mlx_manager/__init__.py"
echo "  • frontend/package.json"
echo "  • Formula/mlx-manager.rb"
echo ""
echo "Note: After PyPI release, update Formula sha256"
