"""Market adapters for prediction market simulation.

This module provides the LMSR (Logarithmic Market Scoring Rule) adapter,
which is the recommended market mechanism for prediction markets.

LMSR provides:
- Instant liquidity (always ready to trade)
- Full trade visibility and tracking
- Automatic price discovery
- Position tracking
- Perfect for agent-based simulation
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping, Sequence

from ..simulation.interfaces import MarketAdapter, MarketOrder


class LMSRMarketAdapter(MarketAdapter):
    """LMSR market maker adapter for prediction markets.
    
    LMSR (Logarithmic Market Scoring Rule) is the standard automated market
    maker for prediction markets, used by platforms like Augur and Gnosis.
    
    Features:
    - Automated market making (always has liquidity)
    - Instant trade execution
    - Full trade visibility
    - Price tracking and net flow monitoring
    - Agent position tracking with PnL calculation
    
    Args:
        liquidity_param: Controls how much price moves per trade (higher = more liquid)
        track_positions: Whether to track agent positions and PnL
    """
    
    def __init__(
        self,
        *,
        liquidity_param: float = 100.0,
        track_positions: bool = True
    ):
        """Initialize the LMSR adapter.
        
        Args:
            liquidity_param: LMSR liquidity parameter (higher = more liquidity)
            track_positions: Whether to track agent positions
        """
        from .lmsr import LMSRMarket, LMSROrderConverter
        
        self._market = LMSRMarket(liquidity_param=liquidity_param)
        self._converter = LMSROrderConverter(self._market)
        self._track_positions = track_positions
        
        # State tracking
        self._net_flow_tick = 0.0
        self._tick_volume = 0.0
        
        # Agent position tracking
        self._positions: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"YES": 0.0, "NO": 0.0, "cash": 0.0}
        )
        
    def submit_orders(self, orders: Sequence[MarketOrder], timestep: int) -> None:
        """Submit orders to the LMSR market.
        
        Args:
            orders: Sequence of market orders
            timestep: Current simulation timestep
        """
        self._net_flow_tick = 0.0
        self._tick_volume = 0.0
        
        for order in orders:
            # Convert order to LMSR trade
            trade = self._converter.submit_limit_order(
                agent_id=order.agent_id,
                side=order.side,
                size=order.size,
                limit_price=order.limit_price,
                timestamp=timestep
            )
            
            if trade is not None:
                # Track net flow (buy = positive, sell = negative)
                if order.side.lower() == "buy":
                    self._net_flow_tick += trade.shares
                else:
                    self._net_flow_tick -= trade.shares
                    
                self._tick_volume += abs(trade.shares)
                
                # Update agent positions
                if self._track_positions:
                    agent_pos = self._positions[trade.agent_id]
                    agent_pos[trade.outcome] += trade.shares
                    agent_pos["cash"] -= trade.cost
                    
    def current_price(self) -> float:
        """Get the current market price for YES outcome."""
        return self._market.get_price("YES")
    
    def snapshot(self) -> Mapping[str, object]:
        """Get comprehensive market state snapshot."""
        market_snapshot = self._market.snapshot()
        
        snapshot_data = {
            "price": market_snapshot["yes_price"],
            "yes_price": market_snapshot["yes_price"],
            "no_price": market_snapshot["no_price"],
            "yes_shares": market_snapshot["yes_shares"],
            "no_shares": market_snapshot["no_shares"],
            "spread": 0.0,  # LMSR has no spread
            "net_flow": self._net_flow_tick,
            "tick_volume": self._tick_volume,
            "total_volume": market_snapshot["total_volume"],
            "num_trades": market_snapshot["num_trades"],
            "liquidity_param": market_snapshot["liquidity_param"],
        }
        
        # Add position tracking if enabled
        if self._track_positions:
            snapshot_data["positions"] = dict(self._positions)
            
        return snapshot_data
    
    def get_agent_position(self, agent_id: str) -> Dict[str, float]:
        """Get an agent's current position.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Dictionary with position information
        """
        pos = self._positions[agent_id]
        
        # Calculate PnL based on current prices
        yes_price = self._market.get_price("YES")
        no_price = self._market.get_price("NO")
        
        pnl = (
            pos["YES"] * yes_price +
            pos["NO"] * no_price +
            pos["cash"]
        )
        
        return {
            "yes_shares": pos["YES"],
            "no_shares": pos["NO"],
            "cash": pos["cash"],
            "pnl": pnl
        }
