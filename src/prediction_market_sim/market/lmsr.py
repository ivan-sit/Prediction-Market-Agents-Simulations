"""LMSR market maker for prediction markets."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class LMSRTrade:
    trade_id: str
    timestamp: int
    agent_id: str
    outcome: str 
    shares: float
    cost: float
    price: float 


@dataclass
class LMSRMarket:
    """LMSR market maker. Cost function: C(q) = b * log(exp(q_yes/b) + exp(q_no/b))"""

    liquidity_param: float = 100.0
    initial_shares: Dict[str, float] = field(default_factory=lambda: {"YES": 0.0, "NO": 0.0})
    
    def __post_init__(self):
        self._outstanding_shares = self.initial_shares.copy()
        self._trades: List[LMSRTrade] = []
        self._trade_counter = 0
        self._total_volume = 0.0
        
    def get_price(self, outcome: str) -> float:
        """Get marginal price for outcome (0-1)."""
        q_yes = self._outstanding_shares["YES"]
        q_no = self._outstanding_shares["NO"]
        b = self.liquidity_param

        if outcome.upper() == "YES":
            diff = (q_no - q_yes) / b
            if diff > 100:
                price = 0.0
            elif diff < -100:
                price = 1.0
            else:
                price = 1.0 / (1.0 + math.exp(diff))
        else:
            diff = (q_yes - q_no) / b
            if diff > 100:
                price = 0.0
            elif diff < -100:
                price = 1.0
            else:
                price = 1.0 / (1.0 + math.exp(diff))

        return price
    
    def get_cost(self, shares: Dict[str, float]) -> float:
        """Calculate cost function for given shares."""
        q_yes = shares.get("YES", 0.0)
        q_no = shares.get("NO", 0.0)
        b = self.liquidity_param

        a = q_yes / b
        b_val = q_no / b
        max_val = max(a, b_val)
        diff = abs(a - b_val)

        if diff > 100:
            return b * max_val
        else:
            return b * (max_val + math.log(1.0 + math.exp(-diff)))
    
    def calculate_cost_for_shares(self, outcome: str, num_shares: float) -> float:
        """Calculate cost to buy/sell shares (positive=pay, negative=receive)."""
        current_cost = self.get_cost(self._outstanding_shares)
        new_shares = self._outstanding_shares.copy()
        new_shares[outcome.upper()] += num_shares
        new_cost = self.get_cost(new_shares)
        return new_cost - current_cost
    
    def buy_shares(
        self,
        *,
        agent_id: str,
        outcome: str,
        num_shares: float,
        timestamp: int
    ) -> LMSRTrade:
        """Buy shares of an outcome."""
        outcome = outcome.upper()
        cost = self.calculate_cost_for_shares(outcome, num_shares)
        price = self.get_price(outcome)
        self._outstanding_shares[outcome] += num_shares
        self._total_volume += abs(num_shares)

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
        """Buy max shares within budget. Returns None if budget too small."""
        if max_cost <= 0:
            return None

        outcome = outcome.upper()
        low, high = 0.0, max_cost * 10
        epsilon = 0.001
        
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
        """In LMSR, price equals implied probability."""
        return self.get_price(outcome)

    def get_outstanding_shares(self) -> Dict[str, float]:
        return self._outstanding_shares.copy()

    def get_trades(self) -> List[LMSRTrade]:
        return self._trades.copy()

    def snapshot(self) -> Dict[str, object]:
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
            "last_price": yes_price,
            "spread": 0.0,
        }


class LMSROrderConverter:
    """Converts limit orders to LMSR trades."""

    def __init__(self, market: LMSRMarket):
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
        """Submit limit order, convert to LMSR trade. Returns None if not filled."""
        side = side.lower()

        if side == "buy":
            current_price = self.market.get_price("YES")
            if current_price <= limit_price:
                return self.market.buy_up_to_price(
                    agent_id=agent_id,
                    outcome="YES",
                    max_cost=size,
                    timestamp=timestamp
                )
        else:
            current_price = self.market.get_price("YES")
            if current_price >= limit_price:
                return self.market.buy_up_to_price(
                    agent_id=agent_id,
                    outcome="NO",
                    max_cost=size,
                    timestamp=timestamp
                )
                
        return None

