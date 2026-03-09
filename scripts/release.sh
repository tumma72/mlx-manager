#!/bin/bash
# Release script for mlx-manager
# Usage: ./scripts/release.sh <version>
# Example: ./scripts/release.sh 1.2.1
#
# Steps performed:
#   1. Validate version and clean working tree
#   2. Sync version to all project files
#   3. Scaffold changelog entry (opens $EDITOR)
#   4. Commit version bump + changelog
#   5. Create git tag
#   6. Push commits + tag (triggers PyPI publish via CI)
#   7. Wait for PyPI package availability
#   8. Download GitHub tarball and compute SHA256
#   9. Update Homebrew formula with new URL + SHA
#  10. Commit and push formula update

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
FORMULA="$PROJECT_ROOT/Formula/mlx-manager.rb"
CHANGELOG="$PROJECT_ROOT/CHANGELOG.md"
GITHUB_REPO="tumma72/mlx-manager"

# ─── Helpers ──────────────────────────────────────────────────────────────────

info()  { echo "==> $1"; }
ok()    { echo "  [ok] $1"; }
fail()  { echo "  [error] $1" >&2; exit 1; }

# ─── Validate arguments ──────────────────────────────────────────────────────

if [ $# -ne 1 ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 1.2.1"
    exit 1
fi

VERSION="$1"

# Validate semver format
if ! echo "$VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$'; then
    fail "Invalid version format: $VERSION (expected X.Y.Z)"
fi

# ─── Pre-flight checks ───────────────────────────────────────────────────────

info "Pre-flight checks"

cd "$PROJECT_ROOT"

# Clean working tree
if [ -n "$(git status --porcelain)" ]; then
    fail "Working tree is not clean. Commit or stash changes first."
fi

# On main branch
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" != "main" ]; then
    fail "Not on main branch (on: $BRANCH). Switch to main first."
fi

# Tag doesn't already exist
if git tag -l "v$VERSION" | grep -q "v$VERSION"; then
    fail "Tag v$VERSION already exists."
fi

ok "Clean tree on main, tag v$VERSION available"

# ─── Quality gates ──────────────────────────────────────────────────────────

info "Running quality gates"

cd "$PROJECT_ROOT/backend"

# Activate venv if present
if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

ruff check . || fail "ruff check failed — fix lint errors before releasing"
ruff format --check . || fail "ruff format failed — run 'ruff format .' before releasing"
mypy mlx_manager || fail "mypy failed — fix type errors before releasing"
pytest -x -q || fail "Tests failed — fix failing tests before releasing"

cd "$PROJECT_ROOT"

ok "All quality gates passed"

# ─── Step 1: Sync version ────────────────────────────────────────────────────

info "Syncing version to $VERSION"
"$SCRIPT_DIR/sync_version.sh" "$VERSION"

# ─── Step 2: Update changelog ────────────────────────────────────────────────

info "Updating changelog"

# Get today's date
TODAY=$(date +%Y-%m-%d)

# Check if this version already has a changelog entry
if grep -q "## \[$VERSION\]" "$CHANGELOG"; then
    echo "  Changelog entry for $VERSION already exists, skipping."
else
    # Scaffold a new entry at the top
    TMPFILE=$(mktemp)
    {
        head -6 "$CHANGELOG"
        echo ""
        echo "## [$VERSION] - $TODAY"
        echo ""
        echo "### Fixed"
        echo ""
        echo "- TODO: describe changes"
        echo ""
    } > "$TMPFILE"
    tail -n +7 "$CHANGELOG" >> "$TMPFILE"
    mv "$TMPFILE" "$CHANGELOG"

    # Open editor for manual changelog editing
    EDITOR="${EDITOR:-${VISUAL:-vi}}"
    echo "  Opening changelog in $EDITOR..."
    echo "  Edit the [$VERSION] entry, then save and close."
    echo ""
    $EDITOR "$CHANGELOG"

    # Verify TODO was removed
    if grep -q "TODO: describe changes" "$CHANGELOG"; then
        fail "Changelog still contains TODO placeholder. Aborting."
    fi
fi

ok "Changelog updated"

# ─── Step 3: Commit and tag ──────────────────────────────────────────────────

info "Committing version bump"

git add -A
git commit -m "chore: release v$VERSION"

info "Creating tag v$VERSION"
git tag "v$VERSION"

ok "Tagged v$VERSION"

# ─── Step 4: Push ────────────────────────────────────────────────────────────

info "Pushing commits and tag to origin"
git push origin main --tags

ok "Pushed — PyPI publish workflow triggered"

# ─── Step 5: Wait for PyPI ───────────────────────────────────────────────────

info "Waiting for mlx-manager==$VERSION on PyPI..."

MAX_WAIT=300  # 5 minutes
ELAPSED=0
INTERVAL=15

while [ $ELAPSED -lt $MAX_WAIT ]; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "https://pypi.org/pypi/mlx-manager/$VERSION/json")
    if [ "$STATUS" = "200" ]; then
        ok "Package available on PyPI"
        break
    fi
    echo "  Not yet available (${ELAPSED}s elapsed), retrying in ${INTERVAL}s..."
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo ""
    echo "  WARNING: PyPI package not available after ${MAX_WAIT}s."
    echo "  The formula update will proceed anyway — Homebrew installs from PyPI"
    echo "  in post_install, so it should work once the package appears."
    echo ""
fi

# ─── Step 6: Update Homebrew formula ─────────────────────────────────────────

info "Downloading GitHub tarball for SHA256"

TARBALL_URL="https://github.com/$GITHUB_REPO/archive/refs/tags/v$VERSION.tar.gz"
TMPTAR=$(mktemp)
curl -sL "$TARBALL_URL" -o "$TMPTAR"
NEW_SHA=$(shasum -a 256 "$TMPTAR" | awk '{print $1}')
rm -f "$TMPTAR"

ok "SHA256: $NEW_SHA"

info "Updating Homebrew formula"

# Update URL
sed -i '' "s|archive/refs/tags/v[0-9]*\.[0-9]*\.[0-9]*\.tar\.gz|archive/refs/tags/v${VERSION}.tar.gz|" "$FORMULA"
# Update SHA
sed -i '' "s/sha256 \"[a-f0-9]*\"/sha256 \"${NEW_SHA}\"/" "$FORMULA"
# Update version (sync_version.sh already did this, but be safe)
sed -i '' "s/version \"[^\"]*\"/version \"${VERSION}\"/" "$FORMULA"

ok "Formula updated"

# ─── Step 7: Commit and push formula ─────────────────────────────────────────

info "Committing formula update"

git add "$FORMULA"
git commit -m "chore(homebrew): update formula sha256 for v$VERSION"
git push origin main

ok "Formula pushed"

# ─── Done ─────────────────────────────────────────────────────────────────────

echo ""
echo "========================================"
echo "  Release v$VERSION complete!"
echo "========================================"
echo ""
echo "  PyPI:     https://pypi.org/project/mlx-manager/$VERSION/"
echo "  GitHub:   https://github.com/$GITHUB_REPO/releases/tag/v$VERSION"
echo "  Homebrew: brew upgrade mlx-manager"
echo ""
