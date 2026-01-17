#!/bin/bash
# Coverage Summary Script
# Extracts and displays coverage percentages from backend and frontend test runs

set -e

BACKEND_COV="N/A"
FRONTEND_LINES="N/A"
FRONTEND_BRANCHES="N/A"

# Get backend coverage (stored after pytest run)
if [ -f backend/.coverage ]; then
    BACKEND_COV=$(cd backend && .venv/bin/coverage report --format=total 2>/dev/null || echo "N/A")
fi

# Get frontend coverage from last run (vitest stores it in coverage/coverage-summary.json)
if [ -f frontend/coverage/coverage-summary.json ]; then
    # Extract lines coverage percentage
    FRONTEND_LINES=$(python3 -c "import json; d=json.load(open('frontend/coverage/coverage-summary.json')); print(d.get('total',{}).get('lines',{}).get('pct', 'N/A'))" 2>/dev/null || echo "N/A")
    FRONTEND_BRANCHES=$(python3 -c "import json; d=json.load(open('frontend/coverage/coverage-summary.json')); print(d.get('total',{}).get('branches',{}).get('pct', 'N/A'))" 2>/dev/null || echo "N/A")
fi

echo ""
echo "=============================================="
echo "         Combined Coverage Summary            "
echo "=============================================="
echo "Backend (Python):      ${BACKEND_COV}%"
echo "Frontend Lines:        ${FRONTEND_LINES}%"
echo "Frontend Branches:     ${FRONTEND_BRANCHES}%"

# Calculate average if both are available
if [[ "$BACKEND_COV" != "N/A" && "$FRONTEND_LINES" != "N/A" ]]; then
    AVG=$(echo "scale=2; ($BACKEND_COV + $FRONTEND_LINES) / 2" | bc)
    echo "----------------------------------------------"
    echo "Average (Lines):       ${AVG}%"
fi

echo "=============================================="
echo ""
