"""Visualization module for prediction market simulations.

This module provides tools for creating animated visualizations of
information flow and market dynamics in prediction market simulations.

Usage:
    # Generate animations from CLI
    python -m prediction_market_sim.visualization.animate simulation_logs/ --html

    # Or use programmatically
    from prediction_market_sim.visualization import (
        AnimationExporter,
        render_html_animation,
    )

    exporter = AnimationExporter.from_simulation_logs(log_dir, run_id)
    render_html_animation(exporter, "animation.html")
"""

from .animation_exporter import AnimationExporter, load_flow_data
from .html_renderer import render_html_animation, render_html_from_logs

__all__ = [
    "AnimationExporter",
    "load_flow_data",
    "render_html_animation",
    "render_html_from_logs",
]
