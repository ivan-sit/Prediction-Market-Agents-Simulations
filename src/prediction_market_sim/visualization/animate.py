"""CLI tool for generating HTML animations from simulation logs.

Usage:
    python -m prediction_market_sim.visualization.animate <log_dir> [options]

Examples:
    # Generate HTML animation
    python -m prediction_market_sim.visualization.animate simulation_logs/ --html

    # Generate HTML for a specific run
    python -m prediction_market_sim.visualization.animate simulation_logs/ --run-id my_run_run1 --html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from .animation_exporter import AnimationExporter, load_flow_data
from .html_renderer import render_html_animation


def find_latest_run(log_dir: Path) -> Optional[str]:
    """Find the most recent run ID in a log directory.

    Args:
        log_dir: Directory containing simulation logs

    Returns:
        Run ID string or None if no runs found
    """
    flow_files = list(log_dir.glob("*_flow.json"))
    if not flow_files:
        # Try to find from market files
        market_files = list(log_dir.glob("*_market.json"))
        if not market_files:
            return None
        latest = max(market_files, key=lambda p: p.stat().st_mtime)
        return latest.stem.replace("_market", "")

    latest = max(flow_files, key=lambda p: p.stat().st_mtime)
    return latest.stem.replace("_flow", "")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate animations from prediction market simulation logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s simulation_logs/ --html
  %(prog)s simulation_logs/ --run-id my_simulation_run1 --html
        """,
    )

    parser.add_argument(
        "log_dir",
        type=Path,
        nargs="?",
        default=Path("artifacts"),
        help="Directory containing simulation logs (default: artifacts/)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="prediction_sim_run1",
        help="Specific run ID to animate (default: prediction_sim_run1)",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        default=True,
        help="Generate interactive HTML animation",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output file path (auto-generated if not specified)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for generated files (defaults to current directory)",
    )

    args = parser.parse_args()

    # Validate log directory
    if not args.log_dir.exists():
        print(f"Error: Log directory not found: {args.log_dir}")
        sys.exit(1)

    # Use specified run ID (defaults to prediction_sim_run1)
    run_id = args.run_id
    print(f"Using run: {run_id}")

    # Set output directory
    output_dir = args.output_dir or Path(".")

    # Load animation data
    print(f"Loading simulation data from {args.log_dir}...")
    try:
        exporter = AnimationExporter.from_simulation_logs(args.log_dir, run_id)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    print(f"Loaded {exporter.total_timesteps} timesteps")
    print(f"  - {len(exporter.source_nodes)} source nodes")
    print(f"  - {len(exporter.agent_ids)} agents")
    print(f"  - {len(exporter.market_prices)} price points")

    # Generate HTML
    html_path = args.output if args.output and args.output.suffix == ".html" else output_dir / f"{run_id}_animation.html"
    print(f"\nGenerating HTML animation...")
    try:
        result_path = render_html_animation(exporter, html_path)
        print(f"  Saved to: {result_path}")
    except Exception as e:
        print(f"  Error generating HTML: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
