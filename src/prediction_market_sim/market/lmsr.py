"""LMSR (Logarithmic Market Scoring Rule) market maker for prediction markets.

LMSR is an automated market maker algorithm designed for prediction markets.
It provides instant liquidity and adjusts prices based on the outstanding shares.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class LMSRTrade:
    """Represents a trade in an LMSR market."""
    
    trade_id: str
    timestamp: int
    agent_id: str
    outcome: str  # "YES" or "NO"
    shares: float
    cost: float
    price: float  # Marginal price at time of trade


@dataclass
class LMSRMarket:
    """LMSR market maker for binary prediction markets.
    
    The LMSR (Logarithmic Market Scoring Rule) is a popular automated market maker
    for prediction markets. It ensures that the market is always liquid and prices
    reflect the aggregated beliefs of participants.
    
    The cost function is: C(q) = b * log(exp(q_yes/b) + exp(q_no/b))
    where b is the liquidity parameter and q is the vector of outstanding shares.
    
    Args:
        liquidity_param: Controls market depth (higher = more liquidity, less price movement)
        initial_shares: Initial outstanding shares for each outcome
    """
    
    liquidity_param: float = 100.0
    initial_shares: Dict[str, float] = field(default_factory=lambda: {"YES": 0.0, "NO": 0.0})
    
    def __post_init__(self):
        """Initialize market state."""
        self._outstanding_shares = self.initial_shares.copy()
        self._trades: List[LMSRTrade] = []
        self._trade_counter = 0
        self._total_volume = 0.0
        
    def get_price(self, outcome: str) -> float:
        """Calculate the current marginal price for an outcome.
        
        The price is the derivative of the cost function with respect to the
        quantity of shares for that outcome.
        
        Args:
            outcome: "YES" or "NO"
            
        Returns:
            Current price for the outcome (between 0 and 1)
        """
        q_yes = self._outstanding_shares["YES"]
        q_no = self._outstanding_shares["NO"]
        b = self.liquidity_param
        
        if outcome.upper() == "YES":
            exp_yes = math.exp(q_yes / b)
            exp_no = math.exp(q_no / b)
            price = exp_yes / (exp_yes + exp_no)
        else:  # NO
            exp_yes = math.exp(q_yes / b)
            exp_no = math.exp(q_no / b)
            price = exp_no / (exp_yes + exp_no)
            
        return price
    
    def get_cost(self, shares: Dict[str, float]) -> float:
        """Calculate the cost function for a given quantity of shares.
        
        Args:
            shares: Dictionary mapping outcomes to share quantities
            
        Returns:
            Cost value
        """
        q_yes = shares.get("YES", 0.0)
        q_no = shares.get("NO", 0.0)
        b = self.liquidity_param
        
        return b * math.log(math.exp(q_yes / b) + math.exp(q_no / b))
    
    def calculate_cost_for_shares(self, outcome: str, num_shares: float) -> float:
        """Calculate the cost to purchase a given number of shares.
        
        Args:
            outcome: "YES" or "NO"
            num_shares: Number of shares to purchase (positive) or sell (negative)
            
        Returns:
            Cost of the transaction (positive = payment, negative = payout)
        """
        # Current cost
        current_cost = self.get_cost(self._outstanding_shares)
        
        # New outstanding shares after trade
        new_shares = self._outstanding_shares.copy()
        new_shares[outcome.upper()] += num_shares
        
        # New cost
        new_cost = self.get_cost(new_shares)
        
        # Cost is the difference
        return new_cost - current_cost
    
    def buy_shares(
        self,
        *,
        agent_id: str,
        outcome: str,
        num_shares: float,
        timestamp: int
    ) -> LMSRTrade:
        """Buy shares of an outcome.
        
        Args:
            agent_id: Agent making the purchase
            outcome: "YES" or "NO"
            num_shares: Number of shares to buy
            timestamp: Current simulation timestamp
            
        Returns:
            Trade record
        """
        outcome = outcome.upper()
        
        # Calculate cost
        cost = self.calculate_cost_for_shares(outcome, num_shares)
        
        # Get marginal price before trade
        price = self.get_price(outcome)
        
        # Update outstanding shares
        self._outstanding_shares[outcome] += num_shares
        self._total_volume += abs(num_shares)
        
        # Record trade
        trade = LMSRTrade(
            trade_id=f"T{self._trade_counter}",
            timestamp=timestamp,
            agent_id=agent_id,
            outcome=outcome,
            shares=num_shares,
            cost=cost,
            price=price
        )
        
        self._trades.append(trade)
        self._trade_counter += 1
        
        return trade
    
    def buy_up_to_price(
        self,
        *,
        agent_id: str,
        outcome: str,
        max_cost: float,
        timestamp: int
    ) -> Optional[LMSRTrade]:
        """Buy as many shares as possible within a budget constraint.
        
        Args:
            agent_id: Agent making the purchase
            outcome: "YES" or "NO"
            max_cost: Maximum amount willing to spend
            timestamp: Current simulation timestamp
            
        Returns:
            Trade record or None if max_cost is too small
        """
        if max_cost <= 0:
            return None
            
        outcome = outcome.upper()
        
        # Binary search to find the maximum number of shares we can buy
        low, high = 0.0, max_cost * 10  # Upper bound heuristic
        epsilon = 0.001  # Precision
        
        best_shares = 0.0
        while high - low > epsilon:
            mid = (low + high) / 2
            cost = self.calculate_cost_for_shares(outcome, mid)
            
            if cost <= max_cost:
                best_shares = mid
                low = mid
            else:
                high = mid
                
        if best_shares < epsilon:
            return None
            
        return self.buy_shares(
            agent_id=agent_id,
            outcome=outcome,
            num_shares=best_shares,
            timestamp=timestamp
        )
    
    def get_probability(self, outcome: str) -> float:
        """Get the current market probability for an outcome.
        
        In LMSR, the price equals the implied probability.
        
        Args:
            outcome: "YES" or "NO"
            
        Returns:
            Probability (between 0 and 1)
        """
        return self.get_price(outcome)
    
    def get_outstanding_shares(self) -> Dict[str, float]:
        """Get the current outstanding shares for all outcomes."""
        return self._outstanding_shares.copy()
    
    def get_trades(self) -> List[LMSRTrade]:
        """Get all trades executed in this market."""
        return self._trades.copy()
    
    def snapshot(self) -> Dict[str, object]:
        """Get a complete snapshot of the market state."""
        yes_price = self.get_price("YES")
        no_price = self.get_price("NO")
        
        return {
            "yes_price": yes_price,
            "no_price": no_price,
            "yes_shares": self._outstanding_shares["YES"],
            "no_shares": self._outstanding_shares["NO"],
            "total_volume": self._total_volume,
            "num_trades": len(self._trades),
            "liquidity_param": self.liquidity_param,
            # For compatibility with orderbook interface
            "last_price": yes_price,
            "spread": 0.0,  # LMSR has no spread
        }


class LMSROrderConverter:
    """Converts limit orders to LMSR market trades.
    
    This adapter allows agents to submit limit orders (as they would to an orderbook)
    and automatically converts them to LMSR market trades.
    """
    
    def __init__(self, market: LMSRMarket):
        """Initialize the converter.
        
        Args:
            market: The LMSR market to trade against
        """
        self.market = market
        
    def submit_limit_order(
        self,
        *,
        agent_id: str,
        side: str,
        size: float,
        limit_price: float,
        timestamp: int
    ) -> Optional[LMSRTrade]:
        """Submit a limit order and convert it to an LMSR trade.
        
        Args:
            agent_id: Agent placing the order
            side: "buy" or "sell"
            size: Order size (interpreted as budget for buy, shares for sell)
            limit_price: Limit price (order only executes if current price is favorable)
            timestamp: Current timestamp
            
        Returns:
            Trade record or None if order couldn't be filled
        """
        side = side.lower()
        
        if side == "buy":
            # Buy YES shares if current price is at or below limit
            current_price = self.market.get_price("YES")
            
            if current_price <= limit_price:
                # Use size as budget
                return self.market.buy_up_to_price(
                    agent_id=agent_id,
                    outcome="YES",
                    max_cost=size,
                    timestamp=timestamp
                )
        else:  # sell
            # Buying NO shares is equivalent to selling YES shares
            current_price = self.market.get_price("YES")
            
            if current_price >= limit_price:
                # Use size as budget for NO shares
                return self.market.buy_up_to_price(
                    agent_id=agent_id,
                    outcome="NO",
                    max_cost=size,
                    timestamp=timestamp
                )
                
        return None

