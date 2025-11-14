# Prediction-Market-Agents-Simulations

A comprehensive simulation framework for multi-agent prediction markets with support for **TWO market mechanisms**:

## 🎯 Choose Your Market Type

### 📗 **Order Book** (Realistic - Like Kalshi/Polymarket)
- Price-time priority matching
- Bid/ask spreads and market depth  
- Orders may NOT execute without counterparty
- Realistic liquidity constraints
- **Use for:** Production-like simulations

### 🔵 **LMSR** (Simple - Automated Market Maker)
- Guaranteed liquidity (always trades)
- No spread, instant execution
- Formula-based pricing
- **Use for:** Testing, low-liquidity scenarios

## ✨ Features

- ✅ **Dynamic Market Selection** via config file
- ✅ **Comprehensive Logging** of market state, beliefs, and flows
- ✅ **Multiple Agent Types** (Bayesian, Momentum, Noise traders)
- ✅ **Flexible Architecture** for easy extension
- ✅ **Real-time Position Tracking** and PnL calculation
- ✅ **Visualization Tools** for analysis

## Market Implementations

### 1. Order Book (PyOrderBook)
**Realistic market like Kalshi/Polymarket:**
- Limit orders and market orders
- Order matching via price-time priority
- Bid/ask spread exists
- Liquidity depends on traders
- Orders can fail if no counterparty

### 2. LMSR (Logarithmic Market Scoring Rule)
**Simple automated market maker:**
- Always has liquidity
- Instant execution guaranteed
- Zero spread
- Formula-based pricing
- Good for testing

### Data Logging

The simulator logs all state to structured CSV/JSON files:

- `market_df`: Price, spread, volume, net flow over time
- `belief_df`: Agent beliefs and convergence metrics
- `sources_df`: Information signals fed to agents
- `agent_meta_df`: Agent-specific metadata

### Visualization

Built-in visualization tools to analyze:
- Price evolution and dynamics
- Agent belief convergence
- Trading volume and flow
- Information signal impact

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repo
git clone <repo-url>
cd Prediction-Market-Agents-Simulations

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Market Type

```bash
# Copy config template
cp config.env.example config.env

# Edit config.env to choose market type
# MARKET_TYPE=orderbook  (realistic, like Kalshi)
# MARKET_TYPE=lmsr       (simple, always liquid)
```

### 3. Run Example

```bash
# Run dynamic market selection demo
python examples/demo_market_selection.py
```

The framework supports **multiple orderbook implementations**:
- 🚀 **limit-order-book** (C++/Python, high-performance) - RECOMMENDED
- 🐍 **pyorderbook** (Pure Python, easy to debug)
- 📦 **Custom implementation** (Built-in fallback, no dependencies)

### Run a Demo

```bash
# Basic demo (stub components)
PYTHONPATH=src python examples/demo_simulation.py

# Full-featured demo (OrderBook + LMSR)
PYTHONPATH=src python examples/demo_full_simulation.py

# Demo with external orderbook library (RECOMMENDED for production)
PYTHONPATH=src python examples/demo_external_orderbook.py

# Visualize results
pip install pandas matplotlib seaborn  # if not already installed
python examples/visualize_results.py orderbook_demo_run1
```

## Documentation

- 📘 **[Quick Start Guide](docs/QUICKSTART.md)** - Get up and running quickly
- 📚 **[Library Choices](docs/LIBRARY_CHOICES.md)** - Orderbook library options and recommendations
- 📐 **[Simulator Architecture](docs/SIMULATOR_SKELETON.md)** - System design and interfaces
- 🏦 **[Market Module](docs/MARKET_MODULE.md)** - OrderBook and LMSR documentation

## Architecture

```
Simulator (Orchestrator)
    ↓
MessageStream → PortalNetwork → Agents → MarketAdapter
                                  ↓
                            Market State
                                  ↓
                             Evaluators
```

### Core Components

1. **SimulationEngine**: High-level orchestrator coordinating all modules
2. **MessageStream**: Generates information signals each timestep
3. **PortalNetwork**: Routes messages to agent inboxes
4. **Agents**: Update beliefs and generate orders
5. **MarketAdapter**: Processes orders and updates prices
6. **Evaluators**: Track metrics and measure performance

## Example Usage

### Recommended: Smart Adapter (Auto-detects best library)

```python
from pathlib import Path
from prediction_market_sim import SimulationEngine, SimulationRuntimeConfig
from prediction_market_sim.market import create_orderbook_adapter

# Build simulation with smart adapter
engine = SimulationEngine(
    stream_factory=lambda: MyDataStream(),
    portal_factory=lambda: MyPortalNetwork(),
    agent_factories=[lambda: MyAgent(agent_id="agent_1")],
    # Smart adapter automatically uses best available library
    market_factory=lambda: create_orderbook_adapter(
        initial_price=0.5,
        tick_size=0.01,
        prefer="auto"  # Uses: limit-order-book > pyorderbook > custom
    ),
    evaluator_factories=[],
    runtime_config=SimulationRuntimeConfig(
        max_timesteps=100,
        run_name="my_sim",
        enable_logging=True
    )
)

# Run simulation
result = engine.run_once(run_id=1, seed=42)

# Access results
print(f"Final price: {result.prices[-1]:.4f}")
print(f"Logs saved to: {result.log_files}")
```

### Alternative: Specific Library

```python
from prediction_market_sim.market import LimitOrderBookAdapter

# Explicitly use external library (if installed)
market = LimitOrderBookAdapter(initial_price=0.5, tick_size=0.01)
```

## Project Structure

```
Prediction-Market-Agents-Simulations/
├── src/
│   └── prediction_market_sim/
│       ├── agents/          # Agent implementations
│       ├── data_sources/    # Data streams and portals
│       ├── evaluation/      # Metric calculators
│       ├── market/          # OrderBook and LMSR
│       │   ├── orderbook.py
│       │   ├── lmsr.py
│       │   └── adapters.py
│       ├── simulation/      # Core orchestrator
│       │   ├── engine.py
│       │   ├── interfaces.py
│       │   ├── logging.py
│       │   └── market_adapters.py
│       └── utils/
├── examples/
│   ├── demo_simulation.py         # Basic demo
│   ├── demo_full_simulation.py    # Full-featured demo
│   └── visualize_results.py       # Visualization script
├── docs/
│   ├── QUICKSTART.md              # Getting started
│   ├── SIMULATOR_SKELETON.md      # Architecture details
│   └── MARKET_MODULE.md           # Market documentation
├── requirements.txt
└── README.md
```

## Key Features

### OrderBook Market

```python
from prediction_market_sim.market import OrderBookMarketAdapter

market = OrderBookMarketAdapter(
    initial_price=0.50,
    tick_size=0.01,
    track_positions=True
)
```

Features:
- Price-time priority matching
- Bid-ask spreads
- Market depth tracking
- Position and PnL tracking

### LMSR Market Maker

```python
from prediction_market_sim.market import LMSRMarketAdapter

market = LMSRMarketAdapter(
    liquidity_param=100.0,
    track_positions=True
)
```

Features:
- Instant liquidity
- No spread
- Automated price discovery
- Bounded market maker loss

## Agent Examples

The demo includes several agent types:

- **BayesianAgent**: Updates beliefs using weighted signals
- **MomentumAgent**: Follows price trends
- **NoiseTrader**: Adds random market noise

See `examples/demo_full_simulation.py` for implementations.

## Data Analysis

After running simulations, analyze the data:

```python
import pandas as pd

# Load data
market_df = pd.read_csv("simulation_logs/my_sim_run1_market.csv")
beliefs_df = pd.read_csv("simulation_logs/my_sim_run1_beliefs.csv")

# Analyze
print(f"Price change: {market_df['price'].iloc[-1] - market_df['price'].iloc[0]}")
print(f"Total volume: {market_df['total_volume'].iloc[-1]}")
```

Or use the built-in visualization:

```bash
python examples/visualize_results.py my_sim_run1
```

## Integration with External Libraries

The framework can integrate with external orderbook implementations. See the `OrderbookSim` C++ library example mentioned in the documentation.

## Contributing

To add new components:

1. Implement the relevant protocol from `interfaces.py`
2. Add tests
3. Update documentation
4. Provide example usage

## References

- **LMSR**: Hanson, R. (2003). "Combinatorial Information Market Design"
- **OrderBooks**: Harris, L. (2003). "Trading and Exchanges"

## License

See LICENSE file for details.
