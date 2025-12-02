"""Market adapters for prediction market simulation.

This module provides two market mechanisms:

1. LMSR (Logarithmic Market Scoring Rule):
   - Automated market maker
   - Always has liquidity
   - Good for low-liquidity simulations
   
2. Order Book (using PyOrderBook):
   - Realistic market like Kalshi/Polymarket
   - Liquidity depends on traders
   - Orders may not execute
   - Bid-ask spread exists
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping, Sequence

from ..simulation.interfaces import MarketAdapter, MarketOrder
from .lmsr import LMSRTrade


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
        self._trade_log: List[object] = []
        
        # Agent position tracking
        self._positions: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"YES": 0.0, "NO": 0.0, "cash": 0.0}
        )
        
    def reset_tick_stats(self) -> None:
        """Reset per-tick statistics. Call at start of each timestep."""
        self._net_flow_tick = 0.0
        self._tick_volume = 0.0

    def submit_orders(self, orders: Sequence[MarketOrder], timestep: int) -> None:
        """Submit orders to the LMSR market.

        Args:
            orders: Sequence of market orders
            timestep: Current simulation timestep
        """
        # Note: Don't reset tick stats here - they're reset at timestep start

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
                self._trade_log.append(trade)
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

    def get_price(self, outcome: str) -> float:
        """Get the current market price for a specific outcome.

        Args:
            outcome: 'YES' or 'NO'

        Returns:
            Current price for that outcome
        """
        return self._market.get_price(outcome)

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

    def get_trades(self) -> List[object]:
        """Return all executed trades recorded by this adapter."""
        return list(self._trade_log)

    def buy_up_to_price(self, agent_id: str, outcome: str, max_cost: float, timestamp: int) -> object:
        """Delegate buy_up_to_price to the underlying LMSR market."""
        trade = self._market.buy_up_to_price(
            agent_id=agent_id,
            outcome=outcome,
            max_cost=max_cost,
            timestamp=timestamp
        )
        if trade:
            self._trade_log.append(trade)
            # Track net flow
            if outcome.upper() == "YES": # Buying YES
                self._net_flow_tick += trade.shares
            else: # Buying NO (selling YES)
                self._net_flow_tick -= trade.shares
            
            self._tick_volume += abs(trade.shares)

            if self._track_positions:
                agent_pos = self._positions[trade.agent_id]
                agent_pos[trade.outcome] += trade.shares
                agent_pos["cash"] -= trade.cost
        
        return trade


class OrderBookMarketAdapter(MarketAdapter):
    """
    Realistic order book adapter using PyOrderBook.
    
    Simulates Kalshi/Polymarket-style prediction markets where:
    - Traders submit limit orders (bid/ask)
    - Orders are matched via price-time priority
    - Liquidity depends on trader participation
    - Bid-ask spread exists
    - Orders may not execute if no counterparty exists
    
    This is MORE REALISTIC than LMSR but requires:
    - More agents to provide liquidity
    - Agents handle failed order execution
    - More complex market dynamics
    
    Args:
        market_id: Unique market identifier
        tick_size: Minimum price increment (default 0.01 = 1 cent)
        initial_liquidity: Seed book with initial orders
        track_positions: Track agent positions and PnL
    """
    
    def __init__(
        self,
        *,
        market_id: str = "default_market",
        tick_size: float = 0.01,
        initial_liquidity: bool = True,
        track_positions: bool = True
    ):
        from .orderbook import OrderBookMarket
        
        self._market = OrderBookMarket(
            market_id=market_id,
            outcomes=["YES", "NO"],
            tick_size=tick_size,
            initial_liquidity=initial_liquidity
        )
        
        self._track_positions = track_positions
        self._positions: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"YES": 0.0, "NO": 0.0, "cash": 0.0}
        )
        
        # Track order submission attempts
        self._submitted_orders = []
        self._failed_orders = []
        self._last_results: List[Dict[str, object]] = []
        self._last_market_state: Dict[str, object] = self._market.get_market_state()
    
    def submit_orders(self, orders: Sequence[MarketOrder], timestep: int) -> None:
        """Conform to the MarketAdapter interface expected by the engine."""
        if not orders:
            self._last_results = []
            return
        self._last_results = self.process_orders(orders, timestep)
    
    def current_price(self) -> float:
        if not self._last_market_state:
            self._last_market_state = self._market.get_market_state()
        return float(self._last_market_state.get("mid_price", 0.5))

    def get_price(self, outcome: str) -> float:
        """Get the current market price for a specific outcome.

        Args:
            outcome: 'YES' or 'NO'

        Returns:
            Current mid price for that outcome
        """
        mid_price = self.current_price()
        if outcome.upper() == "YES":
            return mid_price
        else:
            return 1.0 - mid_price

    def snapshot(self) -> Mapping[str, object]:
        return self.get_market_state()
    
    def process_orders(
        self,
        orders: Sequence[MarketOrder],
        timestep: int
    ) -> List[Dict]:
        """
        Process orders from agents.
        
        NOTE: Unlike LMSR, not all orders will execute!
        Orders only execute if there's a counterparty.
        
        Args:
            orders: List of market orders from agents
            timestep: Current simulation timestep
            
        Returns:
            List of execution results (may be empty if no matches)
        """
        results: List[Dict[str, object]] = []
        
        for order in orders:
            self._submitted_orders.append(order)
            quantity = float(order.size)
            if quantity <= 0:
                continue
            
            metadata = order.metadata or {}
            meta_kind = metadata.get("order_type")
            if isinstance(meta_kind, str) and meta_kind:
                order_kind = meta_kind.lower()
            else:
                order_kind = (order.order_type or "limit").lower()
            meta_outcome = metadata.get("outcome")
            if isinstance(meta_outcome, str) and meta_outcome:
                outcome = meta_outcome.upper()
            else:
                outcome = (order.outcome or "YES").upper()
            side = order.side.upper()
            
            if order_kind == "market":
                trades = self._market.submit_market_order(
                    agent_id=order.agent_id,
                    outcome=outcome,
                    side=side,
                    quantity=quantity,
                    timestamp=timestep
                )
                
                if trades:
                    for trade in trades:
                        result = {
                            "agent_id": order.agent_id,
                            "outcome": trade.outcome,
                            "side": side,
                            "quantity": trade.quantity,
                            "price": trade.price,
                            "order_type": order_kind,
                            "executed": True,
                            "trade_id": trade.trade_id
                        }
                        results.append(result)
                        
                        if self._track_positions:
                            self._update_position(
                                agent_id=order.agent_id,
                                outcome=trade.outcome,
                                side=side,
                                quantity=trade.quantity,
                                price=trade.price
                            )
                else:
                    self._failed_orders.append(order)
                    results.append({
                        "agent_id": order.agent_id,
                        "outcome": outcome,
                        "side": side,
                        "quantity": quantity,
                        "price": None,
                        "order_type": order_kind,
                        "executed": False,
                        "reason": "No counterparty available"
                    })
            else:
                limit_price = order.limit_price if order.limit_price is not None else 0.5
                ob_order = self._market.submit_limit_order(
                    agent_id=order.agent_id,
                    outcome=outcome,
                    side=side,
                    price=limit_price,
                    quantity=quantity,
                    timestamp=timestep
                )
                
                if ob_order:
                    result = {
                        "agent_id": order.agent_id,
                        "outcome": outcome,
                        "side": side,
                        "quantity": quantity,
                        "price": limit_price,
                        "order_type": order_kind,
                        "executed": ob_order.filled_quantity > 0,
                        "filled_quantity": ob_order.filled_quantity,
                        "order_id": ob_order.order_id
                    }
                    results.append(result)
                    
                    if self._track_positions and ob_order.filled_quantity > 0:
                        self._update_position(
                            agent_id=order.agent_id,
                            outcome=outcome,
                            side=side,
                            quantity=ob_order.filled_quantity,
                            price=limit_price
                        )
                else:
                    self._failed_orders.append(order)
                    results.append({
                        "agent_id": order.agent_id,
                        "outcome": outcome,
                        "side": side,
                        "quantity": quantity,
                        "price": limit_price,
                        "order_type": order_kind,
                        "executed": False,
                        "reason": "Order rejected"
                    })
        
        self._last_market_state = self._market.get_market_state()
        return results
    
    def _update_position(
        self,
        agent_id: str,
        outcome: str,
        side: str,
        quantity: float,
        price: float
    ):
        """Update agent position after trade execution."""
        pos = self._positions[agent_id]
        norm_side = side.upper()
        norm_outcome = outcome.upper()
        
        if norm_side == "BUY":
            pos[norm_outcome] += quantity
            pos["cash"] -= quantity * price
        else:  # SELL
            pos[norm_outcome] -= quantity
            pos["cash"] += quantity * price
    
    def get_market_state(self) -> Dict:
        """
        Get current market state.
        
        Returns order book state including:
        - Best bid/ask
        - Spread
        - Market depth
        - Last traded price
        - Volume
        """
        base_state = self._market.get_market_state()
        self._last_market_state = base_state
        state = dict(base_state)
        
        # Add adapter-level stats
        state["submitted_orders"] = len(self._submitted_orders)
        state["failed_orders"] = len(self._failed_orders)
        state["execution_rate"] = (
            1.0 - len(self._failed_orders) / len(self._submitted_orders)
            if self._submitted_orders else 1.0
        )
        
        return state
    
    def get_agent_position(self, agent_id: str) -> Dict:
        """Get agent's current position and PnL."""
        if not self._track_positions:
            return {}
        
        pos = self._positions[agent_id]
        state = self.get_market_state()
        
        # Calculate PnL using mid price
        mid_price = state.get("mid_price", 0.5)
        yes_price = mid_price
        no_price = 1.0 - mid_price
        
        pnl = (
            pos["YES"] * yes_price +
            pos["NO"] * no_price +
            pos["cash"]
        )
        
        return {
            "yes_shares": pos["YES"],
            "no_shares": pos["NO"],
            "cash": pos["cash"],
            "pnl": pnl,
            "unrealized_pnl": pnl - pos["cash"]
        }
    
    def get_order_book(self, outcome: str = "YES", depth: int = 10) -> Dict:
        """Get current order book for an outcome."""
        return self._market.get_order_book(outcome.upper(), depth)
    
    def get_trades(self) -> List:
        """Get all executed trades."""
        return self._market.get_trades()
    
    def get_failed_orders(self) -> List[MarketOrder]:
        """Get list of orders that failed to execute."""
        return self._failed_orders.copy()

    def buy_up_to_price(self, agent_id: str, outcome: str, max_cost: float, timestamp: int) -> object:
        """Buy shares up to a maximum cost using market orders.

        For order books, this submits a market order and tries to fill
        as much as possible within the budget. Unlike LMSR, execution
        depends on available liquidity.

        Args:
            agent_id: Agent making the purchase
            outcome: "YES" or "NO"
            max_cost: Maximum amount willing to spend
            timestamp: Current simulation timestamp

        Returns:
            Trade result dict or None if no execution
        """
        if max_cost <= 0:
            return None

        outcome = outcome.upper()

        # Get current best ask to estimate quantity
        state = self._market.get_market_state()
        best_ask = state.get("best_ask", 0.5)

        if best_ask is None or best_ask <= 0:
            best_ask = 0.5  # Default if no asks

        # Estimate quantity we can afford at best ask
        estimated_qty = max_cost / best_ask

        # Submit market order - it will fill what it can
        trades = self._market.submit_market_order(
            agent_id=agent_id,
            outcome=outcome,
            side="BUY",
            quantity=estimated_qty,
            timestamp=timestamp
        )

        if trades:
            # Return first trade (simplified - could aggregate)
            trade = trades[0]

            if self._track_positions:
                self._update_position(
                    agent_id=agent_id,
                    outcome=trade.outcome,
                    side="BUY",
                    quantity=trade.quantity,
                    price=trade.price
                )

            # Return an LMSRTrade dataclass for compatibility with asdict()
            return LMSRTrade(
                trade_id=str(trade.trade_id) if hasattr(trade, 'trade_id') else f"ob_{timestamp}",
                timestamp=timestamp,
                agent_id=agent_id,
                outcome=trade.outcome,
                shares=trade.quantity,
                cost=trade.quantity * trade.price,
                price=trade.price
            )

        return None
