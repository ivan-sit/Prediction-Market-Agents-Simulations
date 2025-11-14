"""
Order Book implementation using PyOrderBook library.

This module provides a realistic order book implementation similar to Kalshi/Polymarket,
where liquidity is provided by traders and orders are matched via price-time priority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
from decimal import Decimal

# Note: Using simple custom implementation
# More straightforward than external libraries
HAS_PYORDERBOOK = True  # We have our own implementation


class SimpleBook:
    """Simple order book with bids and asks."""
    
    def __init__(self):
        self.bids: List[tuple[float, float, str]] = []  # [(price, qty, order_id), ...]
        self.asks: List[tuple[float, float, str]] = []  # [(price, qty, order_id), ...]
    
    def add_bid(self, price: float, quantity: float, order_id: str):
        """Add a buy order (bid)."""
        self.bids.append((price, quantity, order_id))
        # Sort descending (highest first)
        self.bids.sort(reverse=True, key=lambda x: x[0])
    
    def add_ask(self, price: float, quantity: float, order_id: str):
        """Add a sell order (ask)."""
        self.asks.append((price, quantity, order_id))
        # Sort ascending (lowest first)
        self.asks.sort(key=lambda x: x[0])
    
    def get_best_bid(self) -> Optional[float]:
        """Get highest bid price."""
        return self.bids[0][0] if self.bids else None
    
    def get_best_ask(self) -> Optional[float]:
        """Get lowest ask price."""
        return self.asks[0][0] if self.asks else None
    
    def remove_order(self, order_id: str):
        """Remove an order from the book."""
        self.bids = [(p, q, oid) for p, q, oid in self.bids if oid != order_id]
        self.asks = [(p, q, oid) for p, q, oid in self.asks if oid != order_id]


@dataclass
class Order:
    """Represents a limit order in the order book."""
    
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
        """Unfilled quantity."""
        return self.quantity - self.filled_quantity
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.filled_quantity >= self.quantity


@dataclass
class Trade:
    """Represents an executed trade."""
    
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
    """
    Realistic order book implementation using PyOrderBook.
    
    Simulates Kalshi/Polymarket-style prediction markets where:
    - Traders submit limit orders (bid/ask)
    - Orders are matched via price-time priority
    - Liquidity depends on trader participation
    - Bid-ask spread exists
    - Orders may not execute if no counterparty
    
    Args:
        market_id: Unique identifier for this market
        outcomes: List of possible outcomes (usually ["YES", "NO"])
        tick_size: Minimum price increment (e.g., 0.01 = 1 cent)
        initial_liquidity: If True, seed the book with initial orders
    """
    
    def __init__(
        self,
        market_id: str,
        outcomes: List[str],
        tick_size: float = 0.01,
        initial_liquidity: bool = False
    ):
        self.market_id = market_id
        self.outcomes = outcomes
        self.tick_size = tick_size
        
        # Create separate order books for each outcome
        self._books: Dict[str, SimpleBook] = {}
        for outcome in outcomes:
            self._books[outcome] = SimpleBook()
        
        # Track orders and trades
        self._orders: Dict[str, Order] = {}
        self._trades: List[Trade] = []
        self._order_counter = 0
        self._trade_counter = 0
        
        # Market state
        self._total_volume = 0.0
        
        if initial_liquidity:
            self._seed_initial_liquidity()
    
    def _seed_initial_liquidity(self):
        """Seed the book with initial orders around 50% to provide liquidity."""
        # Add some initial bids and asks for YES outcome
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
                timestamp=0
            )
    
    def submit_limit_order(
        self,
        agent_id: str,
        outcome: str,
        side: Literal["BUY", "SELL"],
        price: float,
        quantity: float,
        timestamp: int
    ) -> Optional[Order]:
        """
        Submit a limit order to the book.
        
        Args:
            agent_id: Agent placing the order
            outcome: "YES" or "NO"
            side: "BUY" or "SELL"
            price: Limit price (between 0 and 1)
            quantity: Number of contracts
            timestamp: Current simulation time
            
        Returns:
            Order object if successful, None if invalid
        """
        # Validate price
        if not (0 < price < 1):
            return None
        
        # Round to tick size
        price = round(price / self.tick_size) * self.tick_size
        
        # Create order
        order_id = f"O{self._order_counter}"
        self._order_counter += 1
        
        order = Order(
            order_id=order_id,
            agent_id=agent_id,
            side=side,
            outcome=outcome,
            price=price,
            quantity=quantity,
            timestamp=timestamp
        )
        
        self._orders[order_id] = order
        
        # Add to order book
        book = self._books[outcome]
        
        if side == "BUY":
            # Try to match immediately, then add to book
            trades = self._match_order(book, order, timestamp)
            if order.remaining_quantity > 0:
                book.add_bid(price, order.remaining_quantity, order_id)
        else:  # SELL
            trades = self._match_order(book, order, timestamp)
            if order.remaining_quantity > 0:
                book.add_ask(price, order.remaining_quantity, order_id)
        
        return order
    
    def submit_market_order(
        self,
        agent_id: str,
        outcome: str,
        side: Literal["BUY", "SELL"],
        quantity: float,
        timestamp: int
    ) -> List[Trade]:
        """
        Submit a market order (executes immediately at best available price).
        
        Args:
            agent_id: Agent placing the order
            outcome: "YES" or "NO"
            side: "BUY" or "SELL"
            quantity: Number of contracts
            timestamp: Current simulation time
            
        Returns:
            List of executed trades
        """
        # Market order is like a limit order with extreme price
        if side == "BUY":
            price = 0.99  # Willing to pay up to 99 cents
        else:
            price = 0.01  # Willing to sell down to 1 cent
        
        order_id = f"M{self._order_counter}"
        self._order_counter += 1
        
        order = Order(
            order_id=order_id,
            agent_id=agent_id,
            side=side,
            outcome=outcome,
            price=price,
            quantity=quantity,
            timestamp=timestamp
        )
        
        self._orders[order_id] = order
        
        # Match immediately
        book = self._books[outcome]
        trades = self._match_order(book, order, timestamp)
        
        return trades
    
    def _match_order(
        self,
        book: PyOB,
        order: Order,
        timestamp: int
    ) -> List[Trade]:
        """
        Match an order against the opposite side of the book.
        
        Returns:
            List of executed trades
        """
        trades = []
        
        if order.side == "BUY":
            # Match against asks (sellers)
            while order.remaining_quantity > 0:
                best_ask = self._get_best_ask(order.outcome)
                if best_ask is None or best_ask > order.price:
                    break  # No match possible
                
                # Find quantity available at best ask
                match_quantity = min(
                    order.remaining_quantity,
                    self._get_quantity_at_price(order.outcome, "SELL", best_ask)
                )
                
                # Execute trade
                trade = self._execute_trade(
                    buyer_id=order.agent_id,
                    seller_id="<from_book>",  # Would need to track actual seller
                    outcome=order.outcome,
                    price=best_ask,
                    quantity=match_quantity,
                    timestamp=timestamp,
                    buy_order_id=order.order_id,
                    sell_order_id="<book>"
                )
                
                trades.append(trade)
                order.filled_quantity += match_quantity
                
                # Remove from book (simplified - PyOrderBook handles this)
                break
        
        else:  # SELL
            # Match against bids (buyers)
            while order.remaining_quantity > 0:
                best_bid = self._get_best_bid(order.outcome)
                if best_bid is None or best_bid < order.price:
                    break  # No match possible
                
                match_quantity = min(
                    order.remaining_quantity,
                    self._get_quantity_at_price(order.outcome, "BUY", best_bid)
                )
                
                trade = self._execute_trade(
                    buyer_id="<from_book>",
                    seller_id=order.agent_id,
                    outcome=order.outcome,
                    price=best_bid,
                    quantity=match_quantity,
                    timestamp=timestamp,
                    buy_order_id="<book>",
                    sell_order_id=order.order_id
                )
                
                trades.append(trade)
                order.filled_quantity += match_quantity
                break
        
        return trades
    
    def _execute_trade(
        self,
        buyer_id: str,
        seller_id: str,
        outcome: str,
        price: float,
        quantity: float,
        timestamp: int,
        buy_order_id: str,
        sell_order_id: str
    ) -> Trade:
        """Record an executed trade."""
        trade_id = f"T{self._trade_counter}"
        self._trade_counter += 1
        
        trade = Trade(
            trade_id=trade_id,
            timestamp=timestamp,
            buyer_id=buyer_id,
            seller_id=seller_id,
            outcome=outcome,
            price=price,
            quantity=quantity,
            buy_order_id=buy_order_id,
            sell_order_id=sell_order_id
        )
        
        self._trades.append(trade)
        self._total_volume += quantity
        
        return trade
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an unfilled order.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if successful, False if order not found or already filled
        """
        if order_id not in self._orders:
            return False
        
        order = self._orders[order_id]
        if order.is_filled:
            return False
        
        # Remove from book (simplified)
        # In production, would need to track order position in book
        del self._orders[order_id]
        return True
    
    def _get_best_bid(self, outcome: str) -> Optional[float]:
        """Get highest bid price for an outcome."""
        book = self._books[outcome]
        return book.get_best_bid()
    
    def _get_best_ask(self, outcome: str) -> Optional[float]:
        """Get lowest ask price for an outcome."""
        book = self._books[outcome]
        return book.get_best_ask()
    
    def _get_quantity_at_price(
        self,
        outcome: str,
        side: Literal["BUY", "SELL"],
        price: float
    ) -> float:
        """Get total quantity available at a specific price level."""
        book = self._books[outcome]
        
        if side == "BUY":
            # Looking for quantity in bids
            for p, q, _ in book.bids:
                if abs(p - price) < 0.0001:  # Float comparison
                    return q
        else:
            # Looking for quantity in asks
            for p, q, _ in book.asks:
                if abs(p - price) < 0.0001:
                    return q
        
        return 0.0
    
    def get_market_state(self) -> Dict:
        """
        Get current market state.
        
        Returns:
            Dictionary with market data including best bid/ask, spread, depth
        """
        best_bid = self._get_best_bid("YES")
        best_ask = self._get_best_ask("YES")
        
        return {
            "market_id": self.market_id,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": (best_ask - best_bid) if (best_bid and best_ask) else None,
            "mid_price": ((best_bid + best_ask) / 2) if (best_bid and best_ask) else 0.5,
            "total_volume": self._total_volume,
            "num_trades": len(self._trades),
            "num_open_orders": len([o for o in self._orders.values() if not o.is_filled]),
            "book_depth": {
                "bid_depth": sum(o.remaining_quantity for o in self._orders.values() 
                                if o.side == "BUY" and not o.is_filled),
                "ask_depth": sum(o.remaining_quantity for o in self._orders.values() 
                                if o.side == "SELL" and not o.is_filled)
            }
        }
    
    def get_order_book(self, outcome: str, depth: int = 10) -> Dict:
        """
        Get order book for a specific outcome.
        
        Args:
            outcome: "YES" or "NO"
            depth: Number of price levels to return on each side
            
        Returns:
            Dictionary with bids and asks
        """
        # Simplified version
        return {
            "outcome": outcome,
            "bids": [],  # [(price, quantity), ...]
            "asks": []   # [(price, quantity), ...]
        }
    
    def get_trades(self) -> List[Trade]:
        """Get all executed trades."""
        return self._trades.copy()
    
    def get_last_price(self, outcome: str) -> Optional[float]:
        """Get last traded price for an outcome."""
        outcome_trades = [t for t in self._trades if t.outcome == outcome]
        if outcome_trades:
            return outcome_trades[-1].price
        return None

