"""Data logging and tracking for simulation runs.

This module provides comprehensive logging of simulation state including:
- Market state (price, spread, volume, net_flow)
- Agent beliefs and metadata
- Source messages
- Trade history
- Information flow events (for animation/analysis)
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import csv


class FlowEventType(Enum):
    """Types of information flow events for animation."""
    SOURCE_EMIT = "source_emit"        # Source node emits an event
    AGENT_RECEIVE = "agent_receive"    # Agent receives a message
    AGENT_CROSSPOST = "agent_crosspost"  # Agent cross-posts to a channel
    BELIEF_UPDATE = "belief_update"    # Agent updates their belief
    TRADE = "trade"                    # Agent executes a trade
    ROUTE = "route"                    # Portal routes message to agents


@dataclass
class FlowEvent:
    """A single information flow event for animation visualization."""
    timestep: int
    event_type: FlowEventType
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestep": self.timestep,
            "type": self.event_type.value,
            **self.data
        }


@dataclass
class InformationFlowLogger:
    """Captures detailed information flow for animation visualization.

    Tracks the complete journey of information through the simulation:
    - Source emissions with recipient lists
    - Agent message ingestion with belief changes
    - Cross-post events with content transformation
    - Trade executions linked to triggering information
    """

    log_dir: Path = field(default_factory=lambda: Path("simulation_logs"))
    run_id: str = "run_001"

    def __post_init__(self):
        """Initialize flow event storage."""
        self.flow_events: List[FlowEvent] = []
        self.agent_beliefs: Dict[str, float] = {}  # Track current beliefs
        self.agent_subscriptions: Dict[str, List[str]] = {}  # agent_id -> [source_ids]
        self.agent_display_names: Dict[str, str] = {}  # agent_id -> display name
        self.source_nodes: List[str] = []  # Track known source nodes
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def register_agent_subscriptions(
        self,
        agent_id: str,
        subscriptions: List[str],
        display_name: Optional[str] = None
    ) -> None:
        """Register an agent's source subscriptions for network visualization.

        Args:
            agent_id: Agent identifier
            subscriptions: List of source node IDs the agent subscribes to
            display_name: Human-readable name for the agent (e.g., persona name)
        """
        self.agent_subscriptions[agent_id] = subscriptions
        if display_name:
            self.agent_display_names[agent_id] = display_name
        # Track unique source nodes
        for source in subscriptions:
            if source not in self.source_nodes:
                self.source_nodes.append(source)

    def register_source_node(self, source_id: str) -> None:
        """Register a source node in the network.

        Args:
            source_id: Source node identifier
        """
        if source_id not in self.source_nodes:
            self.source_nodes.append(source_id)

    def log_source_emission(
        self,
        timestep: int,
        source_id: str,
        event_id: str,
        content: str,
        recipients: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a source node emitting an event.

        Args:
            timestep: Current simulation timestep
            source_id: Source node that emitted the event
            event_id: Unique event identifier
            content: Event content/description
            recipients: List of agent IDs that will receive this event
            metadata: Additional event metadata (reliability, sentiment, etc.)
        """
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]

        event = FlowEvent(
            timestep=timestep,
            event_type=FlowEventType.SOURCE_EMIT,
            data={
                "source_id": source_id,
                "event_id": event_id,
                "content_hash": content_hash,
                "content_preview": content[:100] if content else "",
                "recipients": recipients,
                "num_recipients": len(recipients),
                **(metadata or {})
            }
        )
        self.flow_events.append(event)
        self.register_source_node(source_id)

    def log_routing(
        self,
        timestep: int,
        messages: Sequence[Mapping[str, Any]],
        routing_result: Mapping[str, List[dict]]
    ) -> None:
        """Log the portal's routing of messages to agents.

        Args:
            timestep: Current simulation timestep
            messages: List of source messages
            routing_result: Mapping of agent_id -> list of messages routed
        """
        for message in messages:
            source_id = message.get("source_id", message.get("source_nodes", ["unknown"])[0] if isinstance(message.get("source_nodes"), list) else "unknown")
            event_id = message.get("event_id", f"msg_{timestep}")
            content = message.get("description", message.get("tagline", str(message)))

            # Find which agents received this message
            recipients = []
            for agent_id, agent_messages in routing_result.items():
                for agent_msg in agent_messages:
                    if agent_msg.get("event_id") == event_id or agent_msg.get("description") == content:
                        recipients.append(agent_id)
                        break

            self.log_source_emission(
                timestep=timestep,
                source_id=source_id,
                event_id=event_id,
                content=content,
                recipients=recipients,
                metadata={
                    "reliability": message.get("reliability"),
                    "sentiment": message.get("sentiment"),
                    "tagline": message.get("tagline"),
                }
            )

    def log_agent_receive(
        self,
        timestep: int,
        agent_id: str,
        source_id: str,
        event_id: str,
        belief_before: Optional[float] = None,
        belief_after: Optional[float] = None
    ) -> None:
        """Log an agent receiving a message.

        Args:
            timestep: Current simulation timestep
            agent_id: Agent that received the message
            source_id: Source the message came from
            event_id: Event identifier
            belief_before: Agent's belief before processing (if known)
            belief_after: Agent's belief after processing (if known)
        """
        event = FlowEvent(
            timestep=timestep,
            event_type=FlowEventType.AGENT_RECEIVE,
            data={
                "agent_id": agent_id,
                "source_id": source_id,
                "event_id": event_id,
                "belief_before": belief_before or self.agent_beliefs.get(agent_id, 0.5),
                "belief_after": belief_after,
            }
        )
        self.flow_events.append(event)

    def log_belief_update(
        self,
        timestep: int,
        agent_id: str,
        belief_before: float,
        belief_after: float,
        market_price: float,
        triggered_by: Optional[str] = None
    ) -> None:
        """Log an agent updating their belief.

        Args:
            timestep: Current simulation timestep
            agent_id: Agent updating belief
            belief_before: Belief before update
            belief_after: Belief after update
            market_price: Current market price
            triggered_by: Event ID that triggered this update (if known)
        """
        belief_delta = belief_after - belief_before

        event = FlowEvent(
            timestep=timestep,
            event_type=FlowEventType.BELIEF_UPDATE,
            data={
                "agent_id": agent_id,
                "belief_before": belief_before,
                "belief_after": belief_after,
                "belief_delta": belief_delta,
                "market_price": market_price,
                "belief_price_gap": abs(belief_after - market_price),
                "triggered_by": triggered_by,
            }
        )
        self.flow_events.append(event)
        self.agent_beliefs[agent_id] = belief_after

    def log_crosspost(
        self,
        timestep: int,
        agent_id: str,
        target_channel: str,
        original_event_id: Optional[str],
        content: str,
        transformation: str = "forwarded"  # "forwarded", "summarized", "amplified", "contradicted"
    ) -> None:
        """Log an agent cross-posting to a channel.

        Args:
            timestep: Current simulation timestep
            agent_id: Agent doing the cross-post
            target_channel: Channel/source node being posted to
            original_event_id: Original event that inspired this post
            content: Content of the cross-post
            transformation: How the content was transformed
        """
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]

        event = FlowEvent(
            timestep=timestep,
            event_type=FlowEventType.AGENT_CROSSPOST,
            data={
                "agent_id": agent_id,
                "target_channel": target_channel,
                "original_event_id": original_event_id,
                "content_hash": content_hash,
                "content_preview": content[:100] if content else "",
                "transformation": transformation,
            }
        )
        self.flow_events.append(event)

    def log_trade(
        self,
        timestep: int,
        agent_id: str,
        side: str,
        shares: float,
        price: float,
        triggered_by: Optional[str] = None,
        confidence: Optional[float] = None
    ) -> None:
        """Log a trade execution.

        Args:
            timestep: Current simulation timestep
            agent_id: Agent executing the trade
            side: "BUY" or "SELL"
            shares: Number of shares traded
            price: Execution price
            triggered_by: Event ID that triggered this trade
            confidence: Agent's confidence level
        """
        event = FlowEvent(
            timestep=timestep,
            event_type=FlowEventType.TRADE,
            data={
                "agent_id": agent_id,
                "side": side.upper(),
                "shares": shares,
                "price": price,
                "value": shares * price,
                "triggered_by": triggered_by,
                "confidence": confidence,
            }
        )
        self.flow_events.append(event)

    def get_network_topology(self) -> Dict[str, Any]:
        """Get the network topology for visualization.

        Returns:
            Dictionary with nodes and edges for network visualization
        """
        nodes = []
        edges = []

        # Add source nodes
        for source_id in self.source_nodes:
            nodes.append({
                "id": source_id,
                "type": "source",
                "label": source_id,
            })

        # Add agent nodes
        for agent_id, subscriptions in self.agent_subscriptions.items():
            # Use display name if available, otherwise agent_id
            label = self.agent_display_names.get(agent_id, agent_id)
            nodes.append({
                "id": agent_id,
                "type": "agent",
                "label": label,
                "subscriptions": subscriptions,
            })

            # Add subscription edges (source -> agent)
            for source_id in subscriptions:
                edges.append({
                    "source": source_id,
                    "target": agent_id,
                    "type": "subscription",
                })

        # Add market node
        nodes.append({
            "id": "market",
            "type": "market",
            "label": "Market",
        })

        # Add agent -> market edges
        for agent_id in self.agent_subscriptions.keys():
            edges.append({
                "source": agent_id,
                "target": "market",
                "type": "trading",
            })

        return {
            "nodes": nodes,
            "edges": edges,
        }

    def get_events_for_timestep(self, timestep: int) -> List[Dict[str, Any]]:
        """Get all flow events for a specific timestep.

        Args:
            timestep: Timestep to get events for

        Returns:
            List of event dictionaries for the timestep
        """
        return [
            e.to_dict() for e in self.flow_events
            if e.timestep == timestep
        ]

    def save_to_json(self) -> Path:
        """Save flow events to JSON file.

        Returns:
            Path to the saved file
        """
        output_path = self.log_dir / f"{self.run_id}_flow.json"

        data = {
            "run_id": self.run_id,
            "network": self.get_network_topology(),
            "events": [e.to_dict() for e in self.flow_events],
            "summary": {
                "total_events": len(self.flow_events),
                "num_sources": len(self.source_nodes),
                "num_agents": len(self.agent_subscriptions),
                "events_by_type": self._count_events_by_type(),
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

        return output_path

    def _count_events_by_type(self) -> Dict[str, int]:
        """Count events by type for summary statistics."""
        counts: Dict[str, int] = {}
        for event in self.flow_events:
            event_type = event.event_type.value
            counts[event_type] = counts.get(event_type, 0) + 1
        return counts

    def export_for_animation(self) -> Dict[str, Any]:
        """Export all data needed for animation rendering.

        Returns:
            Complete animation data structure
        """
        # Group events by timestep
        events_by_timestep: Dict[int, List[Dict[str, Any]]] = {}
        for event in self.flow_events:
            ts = event.timestep
            if ts not in events_by_timestep:
                events_by_timestep[ts] = []
            events_by_timestep[ts].append(event.to_dict())

        # Build frames (one per timestep)
        max_ts = max(events_by_timestep.keys()) if events_by_timestep else 0
        frames = []

        current_beliefs: Dict[str, float] = {}
        current_price = 0.5

        for ts in range(max_ts + 1):
            ts_events = events_by_timestep.get(ts, [])

            # Update state from events
            for event in ts_events:
                if event["type"] == "belief_update":
                    current_beliefs[event["agent_id"]] = event["belief_after"]
                    current_price = event.get("market_price", current_price)
                elif event["type"] == "trade":
                    current_price = event.get("price", current_price)

            frames.append({
                "timestep": ts,
                "market_price": current_price,
                "agent_beliefs": dict(current_beliefs),
                "events": ts_events,
            })

        return {
            "run_id": self.run_id,
            "network": self.get_network_topology(),
            "frames": frames,
            "metadata": {
                "total_timesteps": len(frames),
                "total_events": len(self.flow_events),
                "source_nodes": self.source_nodes,
                "agent_ids": list(self.agent_subscriptions.keys()),
            }
        }


def create_flow_logger(run_id: str, log_dir: Optional[Path] = None) -> InformationFlowLogger:
    """Create an information flow logger.

    Args:
        run_id: Unique identifier for this run
        log_dir: Directory for log files (defaults to ./simulation_logs)

    Returns:
        Configured InformationFlowLogger instance
    """
    if log_dir is None:
        log_dir = Path("simulation_logs")

    return InformationFlowLogger(log_dir=log_dir, run_id=run_id)


@dataclass
class SimulationLogger:
    """Logs all simulation data to structured formats.
    
    Tracks:
    - market_df: Market state per timestep (price, spread, volume, net_flow)
    - belief_df: Agent beliefs per timestep
    - agent_meta_df: Agent metadata and positions
    - text_df: Source messages/signals
    - trade_df: Individual trades
    """
    
    log_dir: Path = Path("simulation_logs")
    run_id: str = "run_001"
    
    def __post_init__(self):
        """Initialize logging structures."""
        self.market_records: List[Dict[str, Any]] = []
        self.belief_records: List[Dict[str, Any]] = []
        self.agent_meta_records: List[Dict[str, Any]] = []
        self.text_records: List[Dict[str, Any]] = []
        self.trade_records: List[Dict[str, Any]] = []
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def log_market_state(
        self,
        timestep: int,
        price: float,
        market_snapshot: Mapping[str, object]
    ) -> None:
        """Log market state for a timestep.
        
        Args:
            timestep: Current timestep
            price: Current market price
            market_snapshot: Full market state snapshot
        """
        record = {
            "timestep": timestep,
            "price": price,
            "spread": market_snapshot.get("spread"),
            "net_flow": market_snapshot.get("net_flow", 0.0),
            "volume": market_snapshot.get("tick_volume", 0.0),
            "total_volume": market_snapshot.get("total_volume", 0.0),
            "num_trades": market_snapshot.get("num_trades", 0),
            "best_bid": market_snapshot.get("best_bid"),
            "best_ask": market_snapshot.get("best_ask"),
            "mid_price": market_snapshot.get("mid_price"),
        }
        
        # Add any additional fields from snapshot
        for key, value in market_snapshot.items():
            if key not in record and not key.startswith("_"):
                # Skip complex nested structures
                if isinstance(value, (int, float, str, bool)) or value is None:
                    record[key] = value
                    
        self.market_records.append(record)
        
    def log_beliefs(
        self,
        timestep: int,
        agent_beliefs: Mapping[str, float],
        market_price: float
    ) -> None:
        """Log agent beliefs for a timestep.
        
        Args:
            timestep: Current timestep
            agent_beliefs: Dictionary mapping agent_id to belief
            market_price: Current market price for comparison
        """
        for agent_id, belief in agent_beliefs.items():
            record = {
                "timestep": timestep,
                "agent_id": agent_id,
                "belief": belief,
                "market_price": market_price,
                "belief_price_gap": abs(belief - market_price)
            }
            self.belief_records.append(record)
            
    def log_agent_metadata(
        self,
        timestep: int,
        agent_id: str,
        metadata: Mapping[str, object]
    ) -> None:
        """Log agent metadata (positions, cash, etc).
        
        Args:
            timestep: Current timestep
            agent_id: Agent identifier
            metadata: Agent-specific metadata
        """
        record = {
            "timestep": timestep,
            "agent_id": agent_id,
        }
        
        # Add metadata fields
        for key, value in metadata.items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                record[key] = value
                
        self.agent_meta_records.append(record)
        
    def log_source_message(
        self,
        timestep: int,
        message: Mapping[str, object]
    ) -> None:
        """Log a source message/signal.

        Args:
            timestep: Current timestep
            message: Message data
        """
        # Handle both source_id (single) and source_nodes (array) formats
        source_nodes = message.get("source_nodes", [])
        if source_nodes:
            source_id = source_nodes[0] if len(source_nodes) == 1 else ",".join(source_nodes)
        else:
            source_id = message.get("source_id", "unknown")

        record = {
            "timestep": timestep,
            "source_id": source_id,
        }

        # Add message fields (including source_nodes if present)
        for key, value in message.items():
            if key not in record:
                if isinstance(value, (int, float, str, bool)) or value is None:
                    record[key] = value
                elif isinstance(value, list):
                    # Serialize lists as JSON
                    record[key] = json.dumps(value)
                elif isinstance(value, dict):
                    # Serialize nested dicts as JSON
                    record[key] = json.dumps(value)

        self.text_records.append(record)
        
    def log_trade(
        self,
        timestep: int,
        trade_data: Mapping[str, object]
    ) -> None:
        """Log an individual trade.
        
        Args:
            timestep: Current timestep
            trade_data: Trade information
        """
        record = {
            "timestep": timestep,
        }
        
        # Add trade fields
        for key, value in trade_data.items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                record[key] = value
                
        self.trade_records.append(record)
        
    def save_to_csv(self) -> Dict[str, Path]:
        """Save all logged data to CSV files.
        
        Returns:
            Dictionary mapping data type to file path
        """
        saved_files = {}
        
        # Save market data
        if self.market_records:
            market_path = self.log_dir / f"{self.run_id}_market.csv"
            self._save_csv(market_path, self.market_records)
            saved_files["market"] = market_path
            
        # Save belief data
        if self.belief_records:
            belief_path = self.log_dir / f"{self.run_id}_beliefs.csv"
            self._save_csv(belief_path, self.belief_records)
            saved_files["beliefs"] = belief_path
            
        # Save agent metadata
        if self.agent_meta_records:
            agent_path = self.log_dir / f"{self.run_id}_agent_meta.csv"
            self._save_csv(agent_path, self.agent_meta_records)
            saved_files["agent_meta"] = agent_path
            
        # Save text/source data
        if self.text_records:
            text_path = self.log_dir / f"{self.run_id}_sources.csv"
            self._save_csv(text_path, self.text_records)
            saved_files["sources"] = text_path
            
        # Save trade data
        if self.trade_records:
            trade_path = self.log_dir / f"{self.run_id}_trades.csv"
            self._save_csv(trade_path, self.trade_records)
            saved_files["trades"] = trade_path
            
        return saved_files
        
    def save_to_json(self) -> Dict[str, Path]:
        """Save all logged data to JSON files.
        
        Returns:
            Dictionary mapping data type to file path
        """
        saved_files = {}
        
        # Save market data
        if self.market_records:
            market_path = self.log_dir / f"{self.run_id}_market.json"
            self._save_json(market_path, self.market_records)
            saved_files["market"] = market_path
            
        # Save belief data
        if self.belief_records:
            belief_path = self.log_dir / f"{self.run_id}_beliefs.json"
            self._save_json(belief_path, self.belief_records)
            saved_files["beliefs"] = belief_path
            
        # Save agent metadata
        if self.agent_meta_records:
            agent_path = self.log_dir / f"{self.run_id}_agent_meta.json"
            self._save_json(agent_path, self.agent_meta_records)
            saved_files["agent_meta"] = agent_path
            
        # Save text/source data
        if self.text_records:
            text_path = self.log_dir / f"{self.run_id}_sources.json"
            self._save_json(text_path, self.text_records)
            saved_files["sources"] = text_path
            
        # Save trade data
        if self.trade_records:
            trade_path = self.log_dir / f"{self.run_id}_trades.json"
            self._save_json(trade_path, self.trade_records)
            saved_files["trades"] = trade_path
            
        return saved_files
        
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the simulation run.
        
        Returns:
            Dictionary of summary statistics
        """
        stats = {
            "run_id": self.run_id,
            "num_timesteps": len(self.market_records),
            "num_beliefs_logged": len(self.belief_records),
            "num_agents": len(set(r["agent_id"] for r in self.belief_records)) if self.belief_records else 0,
            "num_source_messages": len(self.text_records),
            "num_trades": len(self.trade_records),
        }
        
        # Market statistics
        if self.market_records:
            prices = [r["price"] for r in self.market_records if r.get("price") is not None]
            if prices:
                stats["initial_price"] = prices[0]
                stats["final_price"] = prices[-1]
                stats["mean_price"] = sum(prices) / len(prices)
                stats["min_price"] = min(prices)
                stats["max_price"] = max(prices)
                stats["price_change"] = prices[-1] - prices[0]
                
            volumes = [r.get("volume", 0) for r in self.market_records]
            stats["total_volume"] = sum(volumes)
            stats["mean_volume_per_tick"] = sum(volumes) / len(volumes) if volumes else 0
            
        # Belief statistics
        if self.belief_records:
            gaps = [r["belief_price_gap"] for r in self.belief_records]
            stats["mean_belief_price_gap"] = sum(gaps) / len(gaps) if gaps else 0
            stats["max_belief_price_gap"] = max(gaps) if gaps else 0
            
        return stats
        
    @staticmethod
    def _sanitize_text(value: Any) -> Any:
        """Sanitize text values to remove problematic Unicode characters.

        Args:
            value: Value to sanitize

        Returns:
            Sanitized value (strings cleaned, other types unchanged)
        """
        if isinstance(value, str):
            # Remove Unicode replacement character and other problematic chars
            # Replace \ufffd (replacement char) and encode/decode to handle others
            sanitized = value.replace('\ufffd', '?')
            # Also handle any other non-encodable characters by replacing them
            try:
                sanitized.encode('utf-8')
            except UnicodeEncodeError:
                sanitized = sanitized.encode('utf-8', errors='replace').decode('utf-8')
            return sanitized
        return value

    @staticmethod
    def _sanitize_record(record: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize all string values in a record.

        Args:
            record: Record dictionary

        Returns:
            Sanitized record with cleaned string values
        """
        return {k: SimulationLogger._sanitize_text(v) for k, v in record.items()}

    @staticmethod
    def _save_csv(path: Path, records: List[Dict[str, Any]]) -> None:
        """Save records to CSV file.

        Args:
            path: Output file path
            records: List of record dictionaries
        """
        if not records:
            return

        # Get all unique keys across all records
        fieldnames = set()
        for record in records:
            fieldnames.update(record.keys())
        fieldnames = sorted(fieldnames)

        # Sanitize records to handle problematic Unicode characters
        sanitized_records = [SimulationLogger._sanitize_record(r) for r in records]

        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sanitized_records)

    @staticmethod
    def _save_json(path: Path, records: List[Dict[str, Any]]) -> None:
        """Save records to JSON file.

        Args:
            path: Output file path
            records: List of record dictionaries
        """
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, default=str)


def create_logger(run_id: str, log_dir: Optional[Path] = None) -> SimulationLogger:
    """Create a simulation logger.
    
    Args:
        run_id: Unique identifier for this run
        log_dir: Directory for log files (defaults to ./simulation_logs)
        
    Returns:
        Configured SimulationLogger instance
    """
    if log_dir is None:
        log_dir = Path("simulation_logs")
        
    return SimulationLogger(log_dir=log_dir, run_id=run_id)

