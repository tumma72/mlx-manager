#!/bin/bash
# DEPRECATED: This script has been superseded by dev.sh --offline
#
# Use instead:
#   ./scripts/dev.sh --offline
#
# Or via Make:
#   make dev-offline
#
# This script will be removed in a future release.

echo "WARNING: offline_dev.sh is deprecated"
echo "Please use: ./scripts/dev.sh --offline"
echo ""
echo "Starting in 3 seconds..."
sleep 3

# Forward to new implementation
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/dev.sh" --offline
