"""CLI tool for generating animations from simulation logs.

Usage:
    python -m prediction_market_sim.visualization.animate <log_dir> [options]

Examples:
    # Generate both HTML and video
    python -m prediction_market_sim.visualization.animate simulation_logs/ --html --video

    # Generate only HTML for a specific run
    python -m prediction_market_sim.visualization.animate simulation_logs/ --run-id my_run_run1 --html

    # Generate video with custom settings
    python -m prediction_market_sim.visualization.animate simulation_logs/ --video --fps 4 --width 1280 --height 720
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from .animation_exporter import AnimationExporter, load_flow_data
from .html_renderer import render_html_animation
from .video_renderer import render_video_animation


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
  %(prog)s simulation_logs/ --html --video
  %(prog)s simulation_logs/ --run-id my_simulation_run1 --html
  %(prog)s simulation_logs/ --video --fps 4 --output my_animation.mp4
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
        help="Generate interactive HTML animation",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Generate MP4 video animation",
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

    # Video options
    video_group = parser.add_argument_group("Video Options")
    video_group.add_argument(
        "--fps",
        type=int,
        default=2,
        help="Frames per second for video (default: 2)",
    )
    video_group.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Video width in pixels (default: 1920)",
    )
    video_group.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Video height in pixels (default: 1080)",
    )

    args = parser.parse_args()

    # Validate log directory
    if not args.log_dir.exists():
        print(f"Error: Log directory not found: {args.log_dir}")
        sys.exit(1)

    # Use specified run ID (defaults to prediction_sim_run1)
    run_id = args.run_id
    print(f"Using run: {run_id}")

    # Default to both outputs if none specified
    if not args.html and not args.video:
        args.html = True
        args.video = True

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
    if args.html:
        html_path = args.output if args.output and args.output.suffix == ".html" else output_dir / f"{run_id}_animation.html"
        print(f"\nGenerating HTML animation...")
        try:
            result_path = render_html_animation(exporter, html_path)
            print(f"  Saved to: {result_path}")
        except Exception as e:
            print(f"  Error generating HTML: {e}")

    # Generate video
    if args.video:
        video_path = args.output if args.output and args.output.suffix == ".mp4" else output_dir / f"{run_id}_animation.mp4"
        print(f"\nGenerating video animation...")
        print(f"  Resolution: {args.width}x{args.height}")
        print(f"  FPS: {args.fps}")
        try:
            result_path = render_video_animation(
                exporter,
                video_path,
                width=args.width,
                height=args.height,
                fps=args.fps,
            )
            print(f"  Saved to: {result_path}")
        except ImportError as e:
            print(f"  Error: {e}")
            print("  Install required dependencies: pip install networkx matplotlib")
        except RuntimeError as e:
            print(f"  Error: {e}")
        except Exception as e:
            print(f"  Error generating video: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
