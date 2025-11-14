"""Market implementation modules for prediction market simulation.

This package provides LMSR (Logarithmic Market Scoring Rule) implementation,
the industry-standard automated market maker for prediction markets.

LMSR is used by major prediction market platforms like Augur, Gnosis, and others.
It provides instant liquidity, full trade visibility, and automatic price discovery.
"""

# Main market adapter (RECOMMENDED)
from .adapters import LMSRMarketAdapter

# LMSR core components (for advanced usage)
from .lmsr import LMSRMarket, LMSRTrade, LMSROrderConverter

__all__ = [
    # Main adapter (use this in your simulations)
    "LMSRMarketAdapter",
    # Core components (for advanced customization)
    "LMSRMarket",
    "LMSRTrade",
    "LMSROrderConverter",
]
