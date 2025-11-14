"""Market implementation modules for prediction market simulation.

This package provides TWO market implementations:

1. LMSR (Logarithmic Market Scoring Rule):
   - Automated market maker
   - Always has liquidity
   - Good for low-liquidity / simple scenarios
   
2. Order Book (using PyOrderBook):
   - Realistic market like Kalshi/Polymarket
   - Liquidity depends on traders
   - Orders may not execute without counterparty
   - Bid-ask spread exists
   
Choose based on your simulation needs via .env configuration.
"""

# Market adapters
from .adapters import LMSRMarketAdapter, OrderBookMarketAdapter

# LMSR components
from .lmsr import LMSRMarket, LMSRTrade, LMSROrderConverter

# Order Book components
from .orderbook import OrderBookMarket, Order, Trade

__all__ = [
    # LMSR (simple, always liquid)
    "LMSRMarketAdapter",
    "LMSRMarket",
    "LMSRTrade",
    "LMSROrderConverter",
    # Order Book (realistic, like Kalshi)
    "OrderBookMarketAdapter",
    "OrderBookMarket",
    "Order",
    "Trade",
]
