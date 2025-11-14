"""Configuration management for prediction market simulation.

Reads configuration from config.env file or environment variables.
"""

import os
from pathlib import Path
from typing import Optional, Literal


class SimulationConfig:
    """Configuration manager for simulation settings."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Path to .env file (default: config.env in project root)
        """
        self._load_env(config_file)
    
    def _load_env(self, config_file: Optional[str]):
        """Load environment variables from file."""
        if config_file is None:
            # Look for config.env in project root
            project_root = Path(__file__).parent.parent.parent.parent
            config_file = project_root / "config.env"
        
        config_path = Path(config_file)
        if config_path.exists():
            with open(config_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        # Don't override existing env vars
                        if key not in os.environ:
                            os.environ[key] = value
    
    @property
    def market_type(self) -> Literal["lmsr", "orderbook"]:
        """Get market type from config."""
        return os.getenv("MARKET_TYPE", "orderbook").lower()
    
    @property
    def lmsr_liquidity_param(self) -> float:
        """Get LMSR liquidity parameter."""
        return float(os.getenv("LMSR_LIQUIDITY_PARAM", "100.0"))
    
    @property
    def orderbook_tick_size(self) -> float:
        """Get order book tick size."""
        return float(os.getenv("ORDERBOOK_TICK_SIZE", "0.01"))
    
    @property
    def orderbook_initial_liquidity(self) -> bool:
        """Get whether to seed order book with initial liquidity."""
        value = os.getenv("ORDERBOOK_INITIAL_LIQUIDITY", "true").lower()
        return value in ("true", "1", "yes", "on")
    
    @property
    def max_ticks(self) -> int:
        """Get maximum simulation ticks."""
        return int(os.getenv("MAX_TICKS", "100"))
    
    @property
    def enable_logging(self) -> bool:
        """Get whether logging is enabled."""
        value = os.getenv("ENABLE_LOGGING", "true").lower()
        return value in ("true", "1", "yes", "on")
    
    @property
    def save_logs_csv(self) -> bool:
        """Get whether to save logs as CSV."""
        value = os.getenv("SAVE_LOGS_CSV", "true").lower()
        return value in ("true", "1", "yes", "on")
    
    @property
    def save_logs_json(self) -> bool:
        """Get whether to save logs as JSON."""
        value = os.getenv("SAVE_LOGS_JSON", "true").lower()
        return value in ("true", "1", "yes", "on")
    
    @property
    def log_dir(self) -> Path:
        """Get log directory path."""
        return Path(os.getenv("LOG_DIR", "simulation_logs"))
    
    @property
    def output_dir(self) -> Path:
        """Get output directory path."""
        return Path(os.getenv("OUTPUT_DIR", "output"))


def create_market_from_config(
    config: Optional[SimulationConfig] = None,
    market_id: str = "default_market"
):
    """
    Create a market adapter based on configuration.
    
    Args:
        config: Configuration object (default: loads from config.env)
        market_id: Unique market identifier
        
    Returns:
        Market adapter instance (LMSR or OrderBook)
        
    Example:
        >>> config = SimulationConfig()
        >>> market = create_market_from_config(config)
        >>> # Uses market type from config.env
    """
    if config is None:
        config = SimulationConfig()
    
    if config.market_type == "lmsr":
        from ..market import LMSRMarketAdapter
        
        print(f"🔵 Creating LMSR market (liquidity={config.lmsr_liquidity_param})")
        return LMSRMarketAdapter(
            liquidity_param=config.lmsr_liquidity_param
        )
    
    elif config.market_type == "orderbook":
        from ..market import OrderBookMarketAdapter
        
        print(f"📗 Creating Order Book market (tick_size={config.orderbook_tick_size}, "
              f"initial_liquidity={config.orderbook_initial_liquidity})")
        return OrderBookMarketAdapter(
            market_id=market_id,
            tick_size=config.orderbook_tick_size,
            initial_liquidity=config.orderbook_initial_liquidity
        )
    
    else:
        raise ValueError(
            f"Unknown market type: {config.market_type}. "
            f"Must be 'lmsr' or 'orderbook'"
        )

