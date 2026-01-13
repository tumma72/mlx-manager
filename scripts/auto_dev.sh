#!/bin/bash
# DEPRECATED: This script has been superseded by dev.sh
#
# The new dev.sh automatically detects network status.
#
# Use instead:
#   ./scripts/dev.sh           # Auto-detects network
#   ./scripts/dev.sh --offline # Force offline mode
#
# Or via Make:
#   make dev         # Auto-detects network
#   make dev-offline # Force offline mode
#
# This script will be removed in a future release.

echo "WARNING: auto_dev.sh is deprecated"
echo "The new dev.sh automatically detects network status."
echo "Please use: ./scripts/dev.sh"
echo ""
echo "Starting in 3 seconds..."
sleep 3

# Forward to new implementation
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/dev.sh"
