#!/bin/bash
# Convenience script to generate animations from simulation logs
#
# Usage:
#   ./scripts/animate.sh [log_dir] [options]
#
# Examples:
#   ./scripts/animate.sh simulation_logs/ --html --video
#   ./scripts/animate.sh simulation_logs/ --run-id my_run_run1 --html

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

python -m prediction_market_sim.visualization.animate "$@"
