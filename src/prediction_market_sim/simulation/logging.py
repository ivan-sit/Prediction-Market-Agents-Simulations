"""Data logging and tracking for simulation runs.

This module provides comprehensive logging of simulation state including:
- Market state (price, spread, volume, net_flow)
- Agent beliefs and metadata
- Source messages
- Trade history
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import csv


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
        record = {
            "timestep": timestep,
            "source_id": message.get("source_id", "unknown"),
        }
        
        # Add message fields
        for key, value in message.items():
            if key not in record:
                if isinstance(value, (int, float, str, bool)) or value is None:
                    record[key] = value
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

