"""Market adapters: LMSR (always liquid) and OrderBook (realistic)."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping, Sequence

from ..simulation.interfaces import MarketAdapter, MarketOrder
from .lmsr import LMSRTrade


class LMSRMarketAdapter(MarketAdapter):
    """LMSR adapter. Always liquid, instant execution."""

    def __init__(
        self,
        *,
        liquidity_param: float = 100.0,
        track_positions: bool = True
    ):
        from .lmsr import LMSRMarket, LMSROrderConverter
        
        self._market = LMSRMarket(liquidity_param=liquidity_param)
        self._converter = LMSROrderConverter(self._market)
        self._track_positions = track_positions
        
        self._net_flow_tick = 0.0
        self._tick_volume = 0.0
        self._trade_log: List[object] = []
        self._positions: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"YES": 0.0, "NO": 0.0, "cash": 0.0}
        )
        
    def reset_tick_stats(self) -> None:
        self._net_flow_tick = 0.0
        self._tick_volume = 0.0

    def submit_orders(self, orders: Sequence[MarketOrder], timestep: int) -> None:
        for order in orders:
            trade = self._converter.submit_limit_order(
                agent_id=order.agent_id,
                side=order.side,
                size=order.size,
                limit_price=order.limit_price,
                timestamp=timestep
            )

            if trade is not None:
                self._trade_log.append(trade)
                if order.side.lower() == "buy":
                    self._net_flow_tick += trade.shares
                else:
                    self._net_flow_tick -= trade.shares
                self._tick_volume += abs(trade.shares)

                if self._track_positions:
                    agent_pos = self._positions[trade.agent_id]
                    agent_pos[trade.outcome] += trade.shares
                    agent_pos["cash"] -= trade.cost
                    
    def current_price(self) -> float:
        return self._market.get_price("YES")

    def get_price(self, outcome: str) -> float:
        return self._market.get_price(outcome)

    def snapshot(self) -> Mapping[str, object]:
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

        if self._track_positions:
            snapshot_data["positions"] = dict(self._positions)
            
        return snapshot_data
    
    def get_agent_position(self, agent_id: str) -> Dict[str, float]:
        pos = self._positions[agent_id]
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
        return list(self._trade_log)

    def buy_up_to_price(self, agent_id: str, outcome: str, max_cost: float, timestamp: int) -> object:
        trade = self._market.buy_up_to_price(
            agent_id=agent_id,
            outcome=outcome,
            max_cost=max_cost,
            timestamp=timestamp
        )
        if trade:
            self._trade_log.append(trade)
            if outcome.upper() == "YES":
                self._net_flow_tick += trade.shares
            else:
                self._net_flow_tick -= trade.shares
            self._tick_volume += abs(trade.shares)

            if self._track_positions:
                agent_pos = self._positions[trade.agent_id]
                agent_pos[trade.outcome] += trade.shares
                agent_pos["cash"] -= trade.cost
        
        return trade


class OrderBookMarketAdapter(MarketAdapter):
    """OrderBook adapter. Realistic matching, orders may fail."""

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
        self._submitted_orders = []
        self._failed_orders = []
        self._last_results: List[Dict[str, object]] = []
        self._last_market_state: Dict[str, object] = self._market.get_market_state()

    def submit_orders(self, orders: Sequence[MarketOrder], timestep: int) -> None:
        if not orders:
            self._last_results = []
            return
        self._last_results = self.process_orders(orders, timestep)
    
    def current_price(self) -> float:
        if not self._last_market_state:
            self._last_market_state = self._market.get_market_state()
        return float(self._last_market_state.get("mid_price", 0.5))

    def get_price(self, outcome: str) -> float:
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
        pos = self._positions[agent_id]
        norm_side = side.upper()
        norm_outcome = outcome.upper()
        
        if norm_side == "BUY":
            pos[norm_outcome] += quantity
            pos["cash"] -= quantity * price
        else:
            pos[norm_outcome] -= quantity
            pos["cash"] += quantity * price
    
    def get_market_state(self) -> Dict:
        base_state = self._market.get_market_state()
        self._last_market_state = base_state
        state = dict(base_state)
        
        state["submitted_orders"] = len(self._submitted_orders)
        state["failed_orders"] = len(self._failed_orders)
        state["execution_rate"] = (
            1.0 - len(self._failed_orders) / len(self._submitted_orders)
            if self._submitted_orders else 1.0
        )
        
        return state

    def get_agent_position(self, agent_id: str) -> Dict:
        if not self._track_positions:
            return {}
        
        pos = self._positions[agent_id]
        state = self.get_market_state()
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
        return self._market.get_order_book(outcome.upper(), depth)

    def get_trades(self) -> List:
        return self._market.get_trades()

    def get_failed_orders(self) -> List[MarketOrder]:
        return self._failed_orders.copy()

    def buy_up_to_price(self, agent_id: str, outcome: str, max_cost: float, timestamp: int) -> object:
        """Buy max shares within budget. Returns None if no liquidity."""
        if max_cost <= 0:
            return None

        outcome = outcome.upper()
        state = self._market.get_market_state()
        best_ask = state.get("best_ask", 0.5)
        if best_ask is None or best_ask <= 0:
            best_ask = 0.5
        estimated_qty = max_cost / best_ask

        trades = self._market.submit_market_order(
            agent_id=agent_id,
            outcome=outcome,
            side="BUY",
            quantity=estimated_qty,
            timestamp=timestamp
        )

        if trades:
            trade = trades[0]
            if self._track_positions:
                self._update_position(
                    agent_id=agent_id,
                    outcome=trade.outcome,
                    side="BUY",
                    quantity=trade.quantity,
                    price=trade.price
                )
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
