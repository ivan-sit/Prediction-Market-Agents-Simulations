from __future__ import annotations

from typing import Mapping, Sequence

from .interfaces import MarketAdapter, MarketOrder

try: 
    from orderbook import OrderBook 
except ImportError:
    OrderBook = None


class ExternalOrderBookAdapter(MarketAdapter):
    def __init__(self, *, initial_price: float = 0.5, **orderbook_kwargs):
        if OrderBook is None:
            raise ImportError(
                "The `orderbook` package is required for ExternalOrderBookAdapter. "
                "Install via `pip install orderbook`."
            )
        self._book = OrderBook(**orderbook_kwargs)
        self._last_price = initial_price

    def submit_orders(self, orders: Sequence[MarketOrder], timestep: int) -> None:
        for order in orders:
            side = order.side.lower()
            if side == "buy":
                trade = self._book.add_buy(
                    size=order.size,
                    price=order.limit_price,
                    meta={"agent_id": order.agent_id, **(order.metadata or {})},
                )
            elif side == "sell":
                trade = self._book.add_sell(
                    size=order.size,
                    price=order.limit_price,
                    meta={"agent_id": order.agent_id, **(order.metadata or {})},
                )
            else:  # pragma: no cover - defensive
                raise ValueError(f"Unknown order side {order.side}")
            if trade is not None:
                self._last_price = trade.price  # type: ignore[attr-defined]

    def current_price(self) -> float:
        return float(self._last_price)

    def snapshot(self) -> Mapping[str, object]:
        book_state = self._book.to_dict()
        book_state["last_price"] = self._last_price
        return book_state
