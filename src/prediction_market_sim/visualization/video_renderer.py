"""Video animation renderer using NetworkX and Matplotlib.

Generates MP4/GIF animations of information flow in prediction market simulations.
Uses NetworkX for graph layout and Matplotlib for rendering frames.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from .animation_exporter import AnimationExporter


# Custom colormap for beliefs (red -> yellow -> green)
BELIEF_COLORS = LinearSegmentedColormap.from_list(
    "belief", ["#e74c3c", "#f39c12", "#2ecc71"]
)

# Node colors by type
SOURCE_COLORS = {
    "twitter": "#1DA1F2",
    "news_feed": "#E74C3C",
    "expert_analysis": "#2ECC71",
    "reddit": "#FF4500",
    "discord": "#7289DA",
}
DEFAULT_SOURCE_COLOR = "#9B59B6"
MARKET_COLOR = "#F39C12"
AGENT_COLOR = "#3498DB"


class VideoRenderer:
    """Renders animation frames and compiles to video."""

    def __init__(
        self,
        exporter: AnimationExporter,
        width: int = 1920,
        height: int = 1080,
        dpi: int = 100,
        fps: int = 2,
    ):
        """Initialize video renderer.

        Args:
            exporter: AnimationExporter with loaded data
            width: Video width in pixels
            height: Video height in pixels
            dpi: Resolution for rendering
            fps: Frames per second in output video
        """
        if not HAS_NETWORKX:
            raise ImportError("NetworkX is required for video rendering. Install with: pip install networkx")

        self.exporter = exporter
        self.width = width
        self.height = height
        self.dpi = dpi
        self.fps = fps

        # Build NetworkX graph
        self.graph = self._build_graph()
        self.pos = self._compute_layout()

    def _build_graph(self) -> "nx.DiGraph":
        """Build NetworkX graph from animation data."""
        G = nx.DiGraph()

        # Add nodes
        for node in self.exporter.nodes:
            G.add_node(
                node["id"],
                node_type=node.get("type", "unknown"),
                label=node.get("label", node["id"]),
            )

        # Add edges
        for edge in self.exporter.edges:
            G.add_edge(
                edge["source"],
                edge["target"],
                edge_type=edge.get("type", "unknown"),
            )

        return G

    def _compute_layout(self) -> Dict[str, Tuple[float, float]]:
        """Compute hierarchical layout for the graph."""
        # Separate nodes by type for layered layout
        sources = [n for n in self.graph.nodes if self.graph.nodes[n].get("node_type") == "source"]
        agents = [n for n in self.graph.nodes if self.graph.nodes[n].get("node_type") == "agent"]
        market = [n for n in self.graph.nodes if self.graph.nodes[n].get("node_type") == "market"]

        pos = {}

        # Position sources at top
        for i, node in enumerate(sources):
            x = (i + 1) / (len(sources) + 1)
            pos[node] = (x, 0.85)

        # Position agents in middle
        for i, node in enumerate(agents):
            x = (i + 1) / (len(agents) + 1)
            pos[node] = (x, 0.5)

        # Position market at bottom center
        for node in market:
            pos[node] = (0.5, 0.15)

        return pos

    def _get_node_color(
        self,
        node_id: str,
        beliefs: Dict[str, float],
        market_price: float
    ) -> str:
        """Get color for a node based on current state."""
        node_type = self.graph.nodes[node_id].get("node_type")

        if node_type == "source":
            return SOURCE_COLORS.get(node_id, DEFAULT_SOURCE_COLOR)
        elif node_type == "market":
            # Color market by current price
            rgba = BELIEF_COLORS(market_price)
            return f"#{int(rgba[0]*255):02x}{int(rgba[1]*255):02x}{int(rgba[2]*255):02x}"
        elif node_type == "agent":
            if node_id in beliefs:
                rgba = BELIEF_COLORS(beliefs[node_id])
                return f"#{int(rgba[0]*255):02x}{int(rgba[1]*255):02x}{int(rgba[2]*255):02x}"
            return AGENT_COLOR

        return "#888888"

    def _get_node_size(self, node_id: str) -> int:
        """Get size for a node based on type."""
        node_type = self.graph.nodes[node_id].get("node_type")

        if node_type == "source":
            return 800
        elif node_type == "market":
            return 1200
        elif node_type == "agent":
            return 600

        return 400

    def render_frame(
        self,
        timestep: int,
        output_path: Optional[Path] = None,
        show: bool = False,
    ) -> Optional[Path]:
        """Render a single animation frame.

        Args:
            timestep: Timestep to render
            output_path: Path to save frame (optional)
            show: Whether to display the frame

        Returns:
            Path to saved frame if output_path provided
        """
        frame = self.exporter.get_frame(timestep)
        if frame is None:
            return None

        beliefs = frame.get("agent_beliefs", {})
        market_price = frame.get("market_price", 0.5)
        events = frame.get("events", [])

        # Create figure
        fig_width = self.width / self.dpi
        fig_height = self.height / self.dpi
        fig, axes = plt.subplots(
            2, 1,
            figsize=(fig_width, fig_height),
            gridspec_kw={"height_ratios": [4, 1]},
            facecolor="#1a1a2e"
        )

        ax_graph = axes[0]
        ax_timeline = axes[1]

        # === GRAPH PANEL ===
        ax_graph.set_facecolor("#1a1a2e")
        ax_graph.set_xlim(-0.1, 1.1)
        ax_graph.set_ylim(-0.05, 1.05)
        ax_graph.axis("off")

        # Draw edges
        for u, v, data in self.graph.edges(data=True):
            x1, y1 = self.pos[u]
            x2, y2 = self.pos[v]
            edge_type = data.get("edge_type", "unknown")

            style = "--" if edge_type == "subscription" else "-"
            color = "#4a5568" if edge_type == "subscription" else "#F39C12"
            alpha = 0.3 if edge_type == "subscription" else 0.6

            ax_graph.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    alpha=alpha,
                    linestyle=style,
                    connectionstyle="arc3,rad=0.1",
                ),
            )

        # Draw nodes
        for node_id in self.graph.nodes:
            x, y = self.pos[node_id]
            color = self._get_node_color(node_id, beliefs, market_price)
            size = self._get_node_size(node_id)

            # Draw node circle
            circle = plt.Circle(
                (x, y),
                radius=size / 20000,
                facecolor=color,
                edgecolor="white",
                linewidth=2,
                zorder=10,
            )
            ax_graph.add_patch(circle)

            # Draw label
            label = self.graph.nodes[node_id].get("label", node_id)
            ax_graph.text(
                x, y - size / 15000 - 0.05,
                label,
                ha="center",
                va="top",
                fontsize=9,
                color="white",
                zorder=11,
            )

            # Show belief value for agents
            node_type = self.graph.nodes[node_id].get("node_type")
            if node_type == "agent" and node_id in beliefs:
                ax_graph.text(
                    x, y,
                    f"{beliefs[node_id]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                    zorder=12,
                )

        # Draw event indicators
        self._draw_events(ax_graph, events, beliefs, market_price)

        # Title and stats
        ax_graph.text(
            0.5, 0.98,
            f"Information Flow Animation - {self.exporter.run_id}",
            ha="center",
            va="top",
            fontsize=14,
            color="white",
            fontweight="bold",
            transform=ax_graph.transAxes,
        )

        ax_graph.text(
            0.02, 0.98,
            f"Timestep: {timestep}",
            ha="left",
            va="top",
            fontsize=12,
            color="#e94560",
            transform=ax_graph.transAxes,
        )

        ax_graph.text(
            0.98, 0.98,
            f"Price: {market_price:.3f}",
            ha="right",
            va="top",
            fontsize=12,
            color="#F39C12",
            transform=ax_graph.transAxes,
        )

        # Legend
        self._draw_legend(ax_graph)

        # === TIMELINE PANEL ===
        ax_timeline.set_facecolor("#16213e")
        prices = self.exporter.market_prices
        if prices:
            timesteps = list(range(len(prices)))
            ax_timeline.plot(timesteps, prices, color="#e94560", linewidth=2)
            ax_timeline.axvline(x=timestep, color="white", linewidth=2, alpha=0.8)
            ax_timeline.fill_between(
                timesteps[:timestep+1],
                prices[:timestep+1],
                alpha=0.3,
                color="#e94560"
            )
            ax_timeline.set_xlim(0, len(prices) - 1)
            ax_timeline.set_ylim(0, 1)
            ax_timeline.set_xlabel("Timestep", color="white", fontsize=10)
            ax_timeline.set_ylabel("Price", color="white", fontsize=10)
            ax_timeline.tick_params(colors="white", labelsize=8)
            for spine in ax_timeline.spines.values():
                spine.set_color("#4a5568")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=self.dpi, facecolor="#1a1a2e")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return output_path

    def _draw_events(
        self,
        ax: plt.Axes,
        events: List[Dict[str, Any]],
        beliefs: Dict[str, float],
        market_price: float
    ) -> None:
        """Draw event indicators on the graph."""
        for event in events:
            event_type = event.get("type")

            if event_type == "source_emit":
                # Highlight source node
                source_id = event.get("source_id")
                if source_id in self.pos:
                    x, y = self.pos[source_id]
                    ring = plt.Circle(
                        (x, y),
                        radius=0.06,
                        facecolor="none",
                        edgecolor="#e94560",
                        linewidth=3,
                        alpha=0.7,
                        zorder=5,
                    )
                    ax.add_patch(ring)

            elif event_type == "trade":
                # Draw trade arrow
                agent_id = event.get("agent_id")
                if agent_id in self.pos and "market" in self.pos:
                    x1, y1 = self.pos[agent_id]
                    x2, y2 = self.pos["market"]
                    color = "#2ECC71" if event.get("side") == "BUY" else "#E74C3C"

                    ax.annotate(
                        "",
                        xy=(x2, y2),
                        xytext=(x1, y1),
                        arrowprops=dict(
                            arrowstyle="->",
                            color=color,
                            linewidth=3,
                            alpha=0.8,
                        ),
                        zorder=8,
                    )

    def _draw_legend(self, ax: plt.Axes) -> None:
        """Draw legend on the graph."""
        legend_elements = [
            mpatches.Patch(facecolor="#1DA1F2", label="Twitter"),
            mpatches.Patch(facecolor="#E74C3C", label="News Feed"),
            mpatches.Patch(facecolor="#2ECC71", label="Expert"),
            mpatches.Patch(facecolor="#F39C12", label="Market"),
        ]

        ax.legend(
            handles=legend_elements,
            loc="lower left",
            fontsize=8,
            facecolor="#16213e",
            edgecolor="#4a5568",
            labelcolor="white",
        )

    def render_all_frames(self, output_dir: Path) -> List[Path]:
        """Render all animation frames.

        Args:
            output_dir: Directory to save frames

        Returns:
            List of paths to rendered frames
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = []

        for ts in range(self.exporter.total_timesteps):
            frame_path = output_dir / f"frame_{ts:04d}.png"
            self.render_frame(ts, frame_path)
            frame_paths.append(frame_path)
            print(f"Rendered frame {ts + 1}/{self.exporter.total_timesteps}")

        return frame_paths

    def compile_video(
        self,
        frame_dir: Path,
        output_path: Path,
        cleanup_frames: bool = True,
    ) -> Path:
        """Compile frames into video using FFmpeg.

        Args:
            frame_dir: Directory containing frame images
            output_path: Output video path
            cleanup_frames: Whether to delete frames after compilation

        Returns:
            Path to generated video
        """
        # FFmpeg command
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-framerate", str(self.fps),
            "-i", str(frame_dir / "frame_%04d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "23",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr.decode()}")
            raise
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found. Please install FFmpeg to generate videos.")

        # Cleanup frames
        if cleanup_frames:
            for frame_file in frame_dir.glob("frame_*.png"):
                frame_file.unlink()

        return output_path

    def render_video(
        self,
        output_path: Optional[Path] = None,
        cleanup_frames: bool = True,
    ) -> Path:
        """Render complete video animation.

        Args:
            output_path: Output video path (defaults to {run_id}_animation.mp4)
            cleanup_frames: Whether to delete frames after compilation

        Returns:
            Path to generated video
        """
        if output_path is None:
            output_path = Path(f"{self.exporter.run_id}_animation.mp4")

        with tempfile.TemporaryDirectory() as tmpdir:
            frame_dir = Path(tmpdir)
            self.render_all_frames(frame_dir)
            return self.compile_video(frame_dir, output_path, cleanup_frames=False)


def render_video_animation(
    exporter: AnimationExporter,
    output_path: Optional[Path] = None,
    width: int = 1920,
    height: int = 1080,
    fps: int = 2,
) -> Path:
    """Convenience function to render video animation.

    Args:
        exporter: AnimationExporter with loaded data
        output_path: Output video path
        width: Video width in pixels
        height: Video height in pixels
        fps: Frames per second

    Returns:
        Path to generated video
    """
    renderer = VideoRenderer(exporter, width=width, height=height, fps=fps)
    return renderer.render_video(output_path)


def render_video_from_logs(
    log_dir: Path,
    run_id: str,
    output_path: Optional[Path] = None,
    **kwargs
) -> Path:
    """Render video animation directly from log directory.

    Args:
        log_dir: Directory containing simulation logs
        run_id: Run identifier
        output_path: Output video path
        **kwargs: Additional arguments for VideoRenderer

    Returns:
        Path to generated video
    """
    exporter = AnimationExporter.from_simulation_logs(log_dir, run_id)
    return render_video_animation(exporter, output_path, **kwargs)
