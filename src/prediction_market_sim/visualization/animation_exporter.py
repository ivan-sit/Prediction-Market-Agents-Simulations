"""Animation data exporter for simulation visualization.

Converts simulation flow logs into animation-ready data structures
for both D3.js (HTML) and video rendering.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AnimationExporter:
    """Exports simulation data for animation rendering.

    Converts flow logs and simulation data into formats suitable for
    D3.js interactive visualization and video frame generation.
    """

    # Network topology
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)

    # Animation frames (one per timestep)
    frames: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    run_id: str = ""
    total_timesteps: int = 0
    source_nodes: List[str] = field(default_factory=list)
    agent_ids: List[str] = field(default_factory=list)

    # Market data for timeline
    market_prices: List[float] = field(default_factory=list)
    market_volumes: List[float] = field(default_factory=list)

    @classmethod
    def from_flow_log(cls, flow_log_path: Path) -> "AnimationExporter":
        """Create exporter from a flow log JSON file.

        Args:
            flow_log_path: Path to *_flow.json file

        Returns:
            Configured AnimationExporter instance
        """
        with open(flow_log_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        exporter = cls()
        exporter.run_id = data.get("run_id", "unknown")

        # Load network topology
        network = data.get("network", {})
        exporter.nodes = network.get("nodes", [])
        exporter.edges = network.get("edges", [])

        # Extract source nodes and agent IDs
        for node in exporter.nodes:
            if node.get("type") == "source":
                exporter.source_nodes.append(node["id"])
            elif node.get("type") == "agent":
                exporter.agent_ids.append(node["id"])

        # Process events into frames
        events = data.get("events", [])
        exporter._build_frames_from_events(events)

        return exporter

    @classmethod
    def from_simulation_logs(
        cls,
        log_dir: Path,
        run_id: str
    ) -> "AnimationExporter":
        """Create exporter from simulation log directory.

        Combines flow logs with market and belief data for comprehensive animation.

        Args:
            log_dir: Directory containing simulation logs
            run_id: Run identifier (prefix for log files)

        Returns:
            Configured AnimationExporter instance
        """
        exporter = cls()
        exporter.run_id = run_id

        # Load flow log if available
        flow_path = log_dir / f"{run_id}_flow.json"
        has_flow_data = False
        if flow_path.exists():
            with open(flow_path, 'r', encoding='utf-8') as f:
                flow_data = json.load(f)

            network = flow_data.get("network", {})
            exporter.nodes = network.get("nodes", [])
            exporter.edges = network.get("edges", [])

            for node in exporter.nodes:
                if node.get("type") == "source":
                    exporter.source_nodes.append(node["id"])
                elif node.get("type") == "agent":
                    exporter.agent_ids.append(node["id"])

            events = flow_data.get("events", [])
            if events:
                exporter._build_frames_from_events(events)
                has_flow_data = True

        # Load market data
        market_path = log_dir / f"{run_id}_market.json"
        market_by_ts: Dict[int, Dict[str, Any]] = {}
        if market_path.exists():
            with open(market_path, 'r', encoding='utf-8') as f:
                market_data = json.load(f)

            for record in market_data:
                ts = record.get("timestep", 0)
                exporter.market_prices.append(record.get("price", 0.5))
                exporter.market_volumes.append(record.get("volume", 0))
                market_by_ts[ts] = record

        # Load belief data
        belief_path = log_dir / f"{run_id}_beliefs.json"
        beliefs_by_ts: Dict[int, Dict[str, float]] = {}
        belief_records_by_ts: Dict[int, List[Dict[str, Any]]] = {}
        if belief_path.exists():
            with open(belief_path, 'r', encoding='utf-8') as f:
                belief_data = json.load(f)

            # Group beliefs by timestep and collect agent IDs
            for record in belief_data:
                ts = record.get("timestep", 0)
                agent_id = record["agent_id"]
                if ts not in beliefs_by_ts:
                    beliefs_by_ts[ts] = {}
                    belief_records_by_ts[ts] = []
                beliefs_by_ts[ts][agent_id] = record["belief"]
                belief_records_by_ts[ts].append(record)

                # Collect agent IDs if not from flow data
                if agent_id not in exporter.agent_ids:
                    exporter.agent_ids.append(agent_id)

        # Load source data for events
        sources_path = log_dir / f"{run_id}_sources.json"
        sources_by_ts: Dict[int, List[Dict[str, Any]]] = {}
        if sources_path.exists():
            with open(sources_path, 'r', encoding='utf-8') as f:
                sources_data = json.load(f)

            for record in sources_data:
                ts = record.get("timestep", 0)
                if ts not in sources_by_ts:
                    sources_by_ts[ts] = []
                sources_by_ts[ts].append(record)

        # Build frames from market/belief data if no flow events
        if not has_flow_data and exporter.market_prices:
            exporter._build_frames_with_synthetic_events(
                beliefs_by_ts,
                belief_records_by_ts,
                sources_by_ts,
                market_by_ts
            )

        # If we have flow frames, update them with market/belief data
        elif has_flow_data:
            for i, frame in enumerate(exporter.frames):
                if i < len(exporter.market_prices):
                    frame["market_price"] = exporter.market_prices[i]
                    frame["market_volume"] = exporter.market_volumes[i]
                ts = frame.get("timestep", 0)
                if ts in beliefs_by_ts:
                    frame["agent_beliefs"] = beliefs_by_ts[ts]

        # Build network from sources data if no flow data
        if not exporter.nodes:
            exporter._build_network_from_sources(log_dir, run_id)

        exporter.total_timesteps = len(exporter.frames)
        return exporter

    def _build_frames_with_synthetic_events(
        self,
        beliefs_by_ts: Dict[int, Dict[str, float]],
        belief_records_by_ts: Dict[int, List[Dict[str, Any]]],
        sources_by_ts: Dict[int, List[Dict[str, Any]]],
        market_by_ts: Dict[int, Dict[str, Any]]
    ) -> None:
        """Build animation frames with synthetic events from log data.

        Creates source_emit, belief_update, and trade events for animation.

        Args:
            beliefs_by_ts: Beliefs grouped by timestep
            belief_records_by_ts: Full belief records by timestep
            sources_by_ts: Source messages grouped by timestep
            market_by_ts: Market data by timestep
        """
        # Track previous beliefs to detect changes
        prev_beliefs: Dict[str, float] = {}

        for i, price in enumerate(self.market_prices):
            events: List[Dict[str, Any]] = []

            # 1. Create source_emit events from source messages
            for source_record in sources_by_ts.get(i, []):
                # Parse source_nodes from the record
                source_nodes = source_record.get("source_nodes", [])
                if isinstance(source_nodes, str):
                    try:
                        source_nodes = json.loads(source_nodes)
                    except json.JSONDecodeError:
                        source_nodes = []

                # Create an event for each source node
                for source_id in source_nodes:
                    events.append({
                        "type": "source_emit",
                        "timestep": i,
                        "source_id": source_id,
                        "tagline": source_record.get("tagline", ""),
                        "recipients": list(self.agent_ids),  # All agents receive
                        "num_recipients": len(self.agent_ids),
                    })

            # 2. Create belief_update events from belief changes
            current_beliefs = beliefs_by_ts.get(i, {})
            for agent_id, belief in current_beliefs.items():
                prev_belief = prev_beliefs.get(agent_id, 0.5)
                # Only create event if belief changed
                if abs(belief - prev_belief) > 0.001 or i == 0:
                    events.append({
                        "type": "belief_update",
                        "timestep": i,
                        "agent_id": agent_id,
                        "belief_before": prev_belief,
                        "belief_after": belief,
                        "market_price": price,
                    })
                prev_beliefs[agent_id] = belief

            # 3. Create trade events from market data
            market_record = market_by_ts.get(i, {})
            num_trades = market_record.get("num_trades", 0)
            prev_trades = market_by_ts.get(i - 1, {}).get("num_trades", 0) if i > 0 else 0
            new_trades = num_trades - prev_trades

            if new_trades > 0:
                # Distribute trades among agents based on belief-price gap
                for agent_id, belief in current_beliefs.items():
                    gap = belief - price
                    if abs(gap) > 0.05:  # Only trade if significant gap
                        side = "BUY" if gap > 0 else "SELL"
                        # Estimate shares from volume
                        volume = market_record.get("tick_volume", 0)
                        shares_per_agent = volume / max(len(current_beliefs), 1)
                        events.append({
                            "type": "trade",
                            "timestep": i,
                            "agent_id": agent_id,
                            "side": side,
                            "shares": shares_per_agent,
                            "price": price,
                        })

            frame = {
                "timestep": i,
                "market_price": price,
                "market_volume": self.market_volumes[i] if i < len(self.market_volumes) else 0,
                "agent_beliefs": current_beliefs,
                "events": events,
                "event_counts": self._count_event_types(events),
            }
            self.frames.append(frame)

    def _build_frames_from_market_data(
        self,
        beliefs_by_ts: Dict[int, Dict[str, float]]
    ) -> None:
        """Build animation frames from market and belief data (legacy).

        Args:
            beliefs_by_ts: Beliefs grouped by timestep
        """
        for i, price in enumerate(self.market_prices):
            frame = {
                "timestep": i,
                "market_price": price,
                "market_volume": self.market_volumes[i] if i < len(self.market_volumes) else 0,
                "agent_beliefs": beliefs_by_ts.get(i, {}),
                "events": [],
                "event_counts": {},
            }
            self.frames.append(frame)

    def _build_network_from_sources(self, log_dir: Path, run_id: str) -> None:
        """Build network topology from sources data.

        Args:
            log_dir: Directory containing log files
            run_id: Run identifier
        """
        # Load agent subscriptions if available
        agent_subscriptions: Dict[str, List[str]] = {}
        subscriptions_path = log_dir / f"{run_id}_subscriptions.json"
        if subscriptions_path.exists():
            with open(subscriptions_path, 'r', encoding='utf-8') as f:
                subscriptions_data = json.load(f)
            for record in subscriptions_data:
                agent_subscriptions[record['agent_id']] = record['subscriptions']

        # Add source nodes from sources file
        sources_path = log_dir / f"{run_id}_sources.json"
        if sources_path.exists():
            with open(sources_path, 'r', encoding='utf-8') as f:
                sources_data = json.load(f)

            # Collect unique sources - check both source_id and source_nodes fields
            source_ids = set()
            for record in sources_data:
                # Try source_nodes array first (from NBA data format)
                source_nodes = record.get("source_nodes", [])
                # Handle JSON-encoded lists from logging
                if isinstance(source_nodes, str):
                    try:
                        source_nodes = json.loads(source_nodes)
                    except json.JSONDecodeError:
                        source_nodes = []
                if source_nodes:
                    for src in source_nodes:
                        source_ids.add(src)
                else:
                    # Fall back to source_id field
                    source_id = record.get("source_id", "unknown")
                    if source_id and source_id != "unknown":
                        source_ids.add(source_id)

            # If no valid sources found, create a generic one
            if not source_ids:
                source_ids.add("news_feed")

            for source_id in source_ids:
                if source_id not in self.source_nodes:
                    self.source_nodes.append(source_id)
                # Create readable labels
                label = source_id.replace("_", " ").title()
                self.nodes.append({
                    "id": source_id,
                    "type": "source",
                    "label": label,
                })

        # Add agent nodes with their specific subscriptions
        for agent_id in self.agent_ids:
            # Create shorter, readable label (remove _trader suffix, title case)
            label = agent_id.replace("_trader", "").replace("_", " ").title()

            # Use agent-specific subscriptions if available, otherwise all sources
            agent_subs = agent_subscriptions.get(agent_id, self.source_nodes)

            self.nodes.append({
                "id": agent_id,
                "type": "agent",
                "label": label,
                "subscriptions": agent_subs,
            })
            # Add edges only from subscribed sources to this agent
            for source_id in agent_subs:
                # Only add edge if source exists in our network
                if source_id in self.source_nodes:
                    self.edges.append({
                        "source": source_id,
                        "target": agent_id,
                        "type": "subscription",
                    })

        # Add market node
        self.nodes.append({
            "id": "market",
            "type": "market",
            "label": "Market",
        })

        # Add agent -> market edges
        for agent_id in self.agent_ids:
            self.edges.append({
                "source": agent_id,
                "target": "market",
                "type": "trading",
            })

    def _build_frames_from_events(self, events: List[Dict[str, Any]]) -> None:
        """Build animation frames from flow events.

        Args:
            events: List of flow event dictionaries
        """
        # Group events by timestep
        events_by_ts: Dict[int, List[Dict[str, Any]]] = {}
        max_ts = 0

        for event in events:
            ts = event.get("timestep", 0)
            if ts not in events_by_ts:
                events_by_ts[ts] = []
            events_by_ts[ts].append(event)
            max_ts = max(max_ts, ts)

        # Build frames with cumulative state
        current_beliefs: Dict[str, float] = {}
        current_price = 0.5

        for ts in range(max_ts + 1):
            ts_events = events_by_ts.get(ts, [])

            # Update state from events
            for event in ts_events:
                event_type = event.get("type")
                if event_type == "belief_update":
                    current_beliefs[event["agent_id"]] = event["belief_after"]
                    current_price = event.get("market_price", current_price)
                elif event_type == "trade":
                    current_price = event.get("price", current_price)

            self.frames.append({
                "timestep": ts,
                "market_price": current_price,
                "agent_beliefs": dict(current_beliefs),
                "events": ts_events,
                "event_counts": self._count_event_types(ts_events),
            })

    @staticmethod
    def _count_event_types(events: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count events by type for frame summary."""
        counts: Dict[str, int] = {}
        for event in events:
            event_type = event.get("type", "unknown")
            counts[event_type] = counts.get(event_type, 0) + 1
        return counts

    def export_for_d3(self) -> Dict[str, Any]:
        """Export data for D3.js visualization.

        Returns:
            Dictionary with all data needed for D3.js rendering
        """
        return {
            "run_id": self.run_id,
            "network": {
                "nodes": self._enhance_nodes_for_d3(),
                "edges": self._enhance_edges_for_d3(),
            },
            "frames": self.frames,
            "timeline": {
                "prices": self.market_prices,
                "volumes": self.market_volumes,
            },
            "metadata": {
                "total_timesteps": self.total_timesteps,
                "num_sources": len(self.source_nodes),
                "num_agents": len(self.agent_ids),
                "source_nodes": self.source_nodes,
                "agent_ids": self.agent_ids,
            },
        }

    def _enhance_nodes_for_d3(self) -> List[Dict[str, Any]]:
        """Enhance nodes with D3-specific properties."""
        enhanced = []

        # Source node colors
        source_colors = {
            "twitter": "#1DA1F2",
            "news_feed": "#E74C3C",
            "expert_analysis": "#2ECC71",
            "reddit": "#FF4500",
            "discord": "#7289DA",
        }

        for node in self.nodes:
            enhanced_node = dict(node)

            if node.get("type") == "source":
                # Position sources at top
                enhanced_node["layer"] = 0
                enhanced_node["color"] = source_colors.get(node["id"], "#9B59B6")
                enhanced_node["radius"] = 25
            elif node.get("type") == "agent":
                # Position agents in middle
                enhanced_node["layer"] = 1
                enhanced_node["color"] = "#3498DB"
                enhanced_node["radius"] = 20
            elif node.get("type") == "market":
                # Position market at bottom
                enhanced_node["layer"] = 2
                enhanced_node["color"] = "#F39C12"
                enhanced_node["radius"] = 35

            enhanced.append(enhanced_node)

        return enhanced

    def _enhance_edges_for_d3(self) -> List[Dict[str, Any]]:
        """Enhance edges with D3-specific properties."""
        enhanced = []

        for edge in self.edges:
            enhanced_edge = dict(edge)

            if edge.get("type") == "subscription":
                enhanced_edge["style"] = "dashed"
                enhanced_edge["color"] = "#BDC3C7"
                enhanced_edge["width"] = 1
            elif edge.get("type") == "trading":
                enhanced_edge["style"] = "solid"
                enhanced_edge["color"] = "#F39C12"
                enhanced_edge["width"] = 2
            elif edge.get("type") == "crosspost":
                enhanced_edge["style"] = "dotted"
                enhanced_edge["color"] = "#9B59B6"
                enhanced_edge["width"] = 1.5

            enhanced.append(enhanced_edge)

        return enhanced

    def export_for_video(self) -> Dict[str, Any]:
        """Export data for video frame generation.

        Returns:
            Dictionary with data optimized for frame-by-frame rendering
        """
        return {
            "run_id": self.run_id,
            "nodes": self.nodes,
            "edges": self.edges,
            "frames": self.frames,
            "prices": self.market_prices,
            "source_nodes": self.source_nodes,
            "agent_ids": self.agent_ids,
            "total_timesteps": self.total_timesteps,
        }

    def save_d3_data(self, output_path: Path) -> Path:
        """Save D3.js data to JSON file.

        Args:
            output_path: Output file path

        Returns:
            Path to saved file
        """
        data = self.export_for_d3()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        return output_path

    def get_frame(self, timestep: int) -> Optional[Dict[str, Any]]:
        """Get animation frame for a specific timestep.

        Args:
            timestep: Timestep to get frame for

        Returns:
            Frame data or None if timestep not found
        """
        if 0 <= timestep < len(self.frames):
            return self.frames[timestep]
        return None

    def get_events_between(
        self,
        start_ts: int,
        end_ts: int,
        event_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all events between two timesteps.

        Args:
            start_ts: Start timestep (inclusive)
            end_ts: End timestep (inclusive)
            event_type: Optional filter by event type

        Returns:
            List of matching events
        """
        events = []
        for ts in range(start_ts, min(end_ts + 1, len(self.frames))):
            frame = self.frames[ts]
            for event in frame.get("events", []):
                if event_type is None or event.get("type") == event_type:
                    events.append(event)
        return events


def load_flow_data(path: Path) -> AnimationExporter:
    """Convenience function to load flow data from file.

    Args:
        path: Path to flow log JSON or log directory

    Returns:
        Configured AnimationExporter instance
    """
    if path.is_dir():
        # Find the most recent flow log in directory
        flow_files = list(path.glob("*_flow.json"))
        if not flow_files:
            raise FileNotFoundError(f"No flow log files found in {path}")
        # Use the most recently modified
        flow_file = max(flow_files, key=lambda p: p.stat().st_mtime)
        return AnimationExporter.from_flow_log(flow_file)
    else:
        return AnimationExporter.from_flow_log(path)
