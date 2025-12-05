"""Order Book implementation backed by pyorderbook library."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Literal
from uuid import UUID

try: 
    from pyorderbook.book import Book as PyOrderBook
    from pyorderbook.book import TradeBlotter as PyTradeBlotter
    from pyorderbook.order import Order as PyBookOrder
    from pyorderbook.order import Side as PySide
except ImportError:
    PyOrderBook = None
    PyBookOrder = None
    PySide = None
    PyTradeBlotter = None
    HAS_PYORDERBOOK = False
else:
    HAS_PYORDERBOOK = True


@dataclass
class Order:
    order_id: str
    agent_id: str
    side: Literal["BUY", "SELL"]
    outcome: str  # "YES" or "NO"
    price: float
    quantity: float
    timestamp: int
    filled_quantity: float = 0.0

    @property
    def remaining_quantity(self) -> float:
        return max(0.0, self.quantity - self.filled_quantity)

    @property
    def is_filled(self) -> bool:
        return self.remaining_quantity <= 0.0 + 1e-9


@dataclass
class Trade:
    trade_id: str
    timestamp: int
    buyer_id: str
    seller_id: str
    outcome: str
    price: float
    quantity: float
    buy_order_id: str
    sell_order_id: str


class OrderBookMarket:
    """Realistic order book using pyorderbook. Price-time priority matching."""

    def __init__(
        self,
        market_id: str,
        outcomes: List[str],
        tick_size: float = 0.01,
        initial_liquidity: bool = False,
        quantity_scale: int = 1000,
    ):
        if not HAS_PYORDERBOOK:
            raise ImportError("pyorderbook is required for OrderBookMarket.")

        self.market_id = market_id
        self.outcomes = [outcome.upper() for outcome in outcomes]
        self.tick_size = tick_size
        self.quantity_scale = max(1, quantity_scale)

        self._book = PyOrderBook()
        self._orders: Dict[str, Order] = {}
        self._trades: List[Trade] = []
        self._order_counter = 0
        self._trade_counter = 0
        self._total_volume = 0.0

        # Bid/ask mappings between client orders and pyorderbook orders
        self._client_to_py: Dict[str, PyBookOrder] = {}
        self._py_to_client: Dict[UUID, str] = {}

        if initial_liquidity:
            self._seed_initial_liquidity()

    def submit_limit_order(
        self,
        agent_id: str,
        outcome: str,
        side: Literal["BUY", "SELL"],
        price: float,
        quantity: float,
        timestamp: int,
    ) -> Optional[Order]:
        outcome = outcome.upper()
        side = side.upper()

        if outcome not in self.outcomes:
            raise ValueError(f"Unknown outcome '{outcome}'")
        if not (0 < price < 1):
            return None
        if quantity <= 0:
            return None

        price = round(price / self.tick_size) * self.tick_size
        scaled_quantity = self._scale_quantity(quantity)
        order_id = f"O{self._order_counter}"
        self._order_counter += 1

        order = Order(
            order_id=order_id,
            agent_id=agent_id,
            side=side,
            outcome=outcome,
            price=price,
            quantity=quantity,
            timestamp=timestamp,
        )
        self._orders[order_id] = order

        py_order = self._build_py_order(order, scaled_quantity)
        trades = self._match_py_order(order, py_order, timestamp)

        # Clean up finished limit orders
        if order.is_filled:
            self._cleanup_py_order(order_id)

        return order

    def submit_market_order(
        self,
        agent_id: str,
        outcome: str,
        side: Literal["BUY", "SELL"],
        quantity: float,
        timestamp: int,
    ) -> List[Trade]:
        outcome = outcome.upper()
        side = side.upper()

        if quantity <= 0:
            return []

        price = 0.99 if side == "BUY" else 0.01
        order_id = f"M{self._order_counter}"
        self._order_counter += 1

        order = Order(
            order_id=order_id,
            agent_id=agent_id,
            side=side,
            outcome=outcome,
            price=price,
            quantity=quantity,
            timestamp=timestamp,
        )
        self._orders[order_id] = order

        scaled_quantity = self._scale_quantity(quantity)
        py_order = self._build_py_order(order, scaled_quantity)
        trades = self._match_py_order(order, py_order, timestamp)

        if order.is_filled:
            self._cleanup_py_order(order_id)

        return trades

    def cancel_order(self, order_id: str) -> bool:
        if order_id not in self._orders:
            return False

        order = self._orders[order_id]
        if order.is_filled:
            return False

        py_order = self._client_to_py.get(order_id)
        if py_order is None:
            return False

        try:
            self._book.cancel(py_order)
        except Exception:
            return False

        self._cleanup_py_order(order_id)
        del self._orders[order_id]
        return True

    def get_market_state(self) -> Dict:
        reference_outcome = self.outcomes[0] if self.outcomes else "YES"
        best_bid = self._get_best_price(reference_outcome, PySide.BID)
        best_ask = self._get_best_price(reference_outcome, PySide.ASK)

        spread = None
        mid_price = 0.5
        if best_bid is not None and best_ask is not None:
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2
        elif best_bid is not None:
            mid_price = best_bid
        elif best_ask is not None:
            mid_price = best_ask

        open_orders = [o for o in self._orders.values() if not o.is_filled]
        bid_depth = sum(o.remaining_quantity for o in open_orders if o.side == "BUY")
        ask_depth = sum(o.remaining_quantity for o in open_orders if o.side == "SELL")

        return {
            "market_id": self.market_id,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "mid_price": mid_price,
            "total_volume": self._total_volume,
            "num_trades": len(self._trades),
            "num_open_orders": len(open_orders),
            "book_depth": {
                "bid_depth": bid_depth,
                "ask_depth": ask_depth,
            },
        }

    def get_order_book(self, outcome: str, depth: int = 10) -> Dict:
        outcome = outcome.upper()
        bids = self._build_book_entries(outcome, PySide.BID, depth)
        asks = self._build_book_entries(outcome, PySide.ASK, depth)
        return {
            "outcome": outcome,
            "bids": bids,
            "asks": asks,
        }

    def get_trades(self) -> List[Trade]:
        return self._trades.copy()

    def get_last_price(self, outcome: str) -> Optional[float]:
        outcome = outcome.upper()
        for trade in reversed(self._trades):
            if trade.outcome == outcome:
                return trade.price
        return None

    def _seed_initial_liquidity(self) -> None:
        initial_orders = [
            ("YES", "BUY", 0.45, 100),
            ("YES", "BUY", 0.48, 150),
            ("YES", "BUY", 0.49, 200),
            ("YES", "SELL", 0.51, 200),
            ("YES", "SELL", 0.52, 150),
            ("YES", "SELL", 0.55, 100),
        ]

        for outcome, side, price, qty in initial_orders:
            self.submit_limit_order(
                agent_id="market_maker",
                outcome=outcome,
                side=side,
                price=price,
                quantity=qty,
                timestamp=0,
            )

    def _scale_quantity(self, quantity: float) -> int:
        scaled = int(round(quantity * self.quantity_scale))
        return max(1, scaled)

    def _build_py_order(self, order: Order, scaled_quantity: int) -> PyBookOrder:
        py_side = self._to_py_side(order.side)
        py_order = PyBookOrder(py_side, order.outcome, order.price, scaled_quantity)
        py_order.agent_id = order.agent_id
        py_order.client_order_id = order.order_id
        self._register_py_order(order.order_id, py_order)
        return py_order

    def _register_py_order(self, client_order_id: str, py_order: PyBookOrder) -> None:
        self._client_to_py[client_order_id] = py_order
        self._py_to_client[py_order.id] = client_order_id

    def _cleanup_py_order(self, client_order_id: str) -> None:
        py_order = self._client_to_py.pop(client_order_id, None)
        if py_order is not None:
            self._py_to_client.pop(py_order.id, None)

    def _match_py_order(
        self,
        order: Order,
        py_order: PyBookOrder,
        timestamp: int,
    ) -> List[Trade]:
        blotter: PyTradeBlotter = self._book.match(py_order)
        trades = self._convert_trades(blotter.trades, timestamp)

        filled_qty = sum(
            trade.quantity
            for trade in trades
            if (order.side == "BUY" and trade.buy_order_id == order.order_id)
            or (order.side == "SELL" and trade.sell_order_id == order.order_id)
        )
        if filled_qty:
            self._update_order_fill(order.order_id, filled_qty)

        return trades

    def _convert_trades(self, pb_trades: List, timestamp: int) -> List[Trade]:
        trades: List[Trade] = []
        for pb_trade in pb_trades:
            incoming_client = self._py_to_client.get(pb_trade.incoming_order_id)
            standing_client = self._py_to_client.get(pb_trade.standing_order_id)
            if incoming_client is None or standing_client is None:
                continue

            incoming_order = self._orders.get(incoming_client)
            standing_order = self._orders.get(standing_client)
            if incoming_order is None or standing_order is None:
                continue

            quantity = pb_trade.fill_quantity / self.quantity_scale
            price = float(pb_trade.fill_price)

            if incoming_order.side == "BUY":
                buyer_order = incoming_order
                seller_order = standing_order
            else:
                buyer_order = standing_order
                seller_order = incoming_order

            trade = Trade(
                trade_id=f"T{self._trade_counter}",
                timestamp=timestamp,
                buyer_id=buyer_order.agent_id,
                seller_id=seller_order.agent_id,
                outcome=incoming_order.outcome,
                price=price,
                quantity=quantity,
                buy_order_id=buyer_order.order_id,
                sell_order_id=seller_order.order_id,
            )
            self._trade_counter += 1
            self._total_volume += quantity
            self._trades.append(trade)
            trades.append(trade)

            self._update_order_fill(buyer_order.order_id, quantity)
            self._update_order_fill(seller_order.order_id, quantity)

        return trades

    def _update_order_fill(self, client_order_id: str, fill_qty: float) -> None:
        order = self._orders.get(client_order_id)
        if order is None or fill_qty <= 0:
            return
        order.filled_quantity = min(order.quantity, order.filled_quantity + fill_qty)
        if order.is_filled:
            self._cleanup_py_order(client_order_id)

    def _to_py_side(self, side: str) -> PySide:
        return PySide.BID if side == "BUY" else PySide.ASK

    def _get_best_price(self, outcome: str, side: PySide) -> Optional[float]:
        symbol_levels = self._book.levels.get(outcome)
        if not symbol_levels:
            return None
        heap = symbol_levels.get(side)
        if not heap:
            return None
        best_level = heap[0]
        if not best_level.orders:
            return None
        return float(best_level.price)

    def _build_book_entries(
        self,
        outcome: str,
        side: PySide,
        depth: int,
    ) -> List[Dict[str, object]]:
        entries: List[Dict[str, object]] = []
        level_map = self._book.level_map.get(outcome, {}).get(side, {})
        if not level_map:
            return entries

        sorted_prices = sorted(level_map.keys(), reverse=(side == PySide.BID))
        for price in sorted_prices:
            level = level_map[price]
            for py_order_id, py_order in level.orders.items():
                client_id = self._py_to_client.get(py_order_id)
                if client_id is None:
                    continue
                client_order = self._orders.get(client_id)
                if client_order is None:
                    continue
                remaining = py_order.quantity / self.quantity_scale
                entries.append(
                    {
                        "price": float(price),
                        "quantity": remaining,
                        "order_id": client_id,
                        "agent_id": client_order.agent_id,
                    }
                )
                if len(entries) >= depth:
                    return entries
        return entries
