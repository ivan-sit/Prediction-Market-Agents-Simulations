#!/usr/bin/env python3
"""
High-Level Driver Script for Prediction Market Simulation with Synthetic Data

This script properly integrates all modules:
- data: EventDatabase for loading events from JSON
- data_sources: SourceNode network for information routing
- agents: LLM-powered agents that react to events and place trades
- market: LMSR or OrderBook for price discovery
- simulation: SimulationEngine orchestration

Configuration is loaded from config.env with command-line overrides.

Usage:
    # Use defaults from config.env
    python examples/run_with_synthetic_data.py

    # Override specific settings
    python examples/run_with_synthetic_data.py --events data/other.json --agents 5

    # Override all settings
    python examples/run_with_synthetic_data.py --events data/my_events.json --market lmsr --agents 3
"""

import sys
import argparse
from pathlib import Path
import os

import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_env_config(env_path: Path = None) -> dict:
    """
    Load configuration from config.env file.

    Returns dict with defaults that can be overridden by command-line args.
    """
    # Default config values
    config = {
        'events_file': 'data/sample_election_events.json',
        'market_type': 'lmsr',
        'num_agents': 3,
        'max_timesteps': 100,
        'liquidity_param': 100.0,
        'run_name': 'prediction_sim',
        'random_seed': 42,
        'log_dir': 'simulation_logs',
    }

    # Find config.env - check multiple locations
    if env_path is None:
        possible_paths = [
            Path(__file__).parent.parent / "config.env",
            Path.cwd() / "config.env",
        ]
        for p in possible_paths:
            if p.exists():
                env_path = p
                break

    if env_path is None or not env_path.exists():
        print("[INFO] No config.env found, using defaults")
        return config

    print(f"[CONFIG] Loading config from: {env_path}")

    # Parse config.env
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                # Map env vars to config keys
                mapping = {
                    'EVENTS_FILE': ('events_file', str),
                    'MARKET_TYPE': ('market_type', str),
                    'NUM_AGENTS': ('num_agents', int),
                    'MAX_TICKS': ('max_timesteps', int),
                    'LMSR_LIQUIDITY_PARAM': ('liquidity_param', float),
                    'RUN_NAME': ('run_name', str),
                    'RANDOM_SEED': ('random_seed', int),
                    'LOG_DIR': ('log_dir', str),
                }

                if key in mapping:
                    config_key, converter = mapping[key]
                    try:
                        config[config_key] = converter(value)
                    except ValueError:
                        pass  # Keep default if conversion fails

    return config

from prediction_market_sim.simulation.engine import (
    SimulationEngine,
    SimulationRuntimeConfig
)
from prediction_market_sim.data_sources import create_event_stream, create_portal_network
from prediction_market_sim.agents import create_prediction_agent
from prediction_market_sim.market import LMSRMarketAdapter, OrderBookMarketAdapter
from prediction_market_sim.utils.config import SimulationConfig


def create_standard_portals():
    """
    Create the standard set of information portals.

    These portal IDs should be used in your synthetic datasets.
    """
    return create_portal_network([
        {'node_id': 'twitter', 'reliability': 0.6},
        {'node_id': 'news_feed', 'reliability': 0.8},
        {'node_id': 'expert_analysis', 'reliability': 0.9},
        {'node_id': 'reddit', 'reliability': 0.65},
        {'node_id': 'discord', 'reliability': 0.65},
    ])


def create_agents_with_subscriptions(portal_network, num_agents=3):
    """
    Create agents with different personalities and portal subscriptions.

    Returns:
        List of agent factory functions
    """
    agent_configs = [
        {
            'agent_id': 'conservative_trader',
            'personality': 'Conservative trader who values high-quality sources and expert analysis. Risk-averse and careful.',
            'subscriptions': ['news_feed', 'expert_analysis'],
            'initial_cash': 10000.0
        },
        {
            'agent_id': 'aggressive_trader',
            'personality': 'Aggressive momentum trader who follows social media sentiment and takes bold positions.',
            'subscriptions': ['twitter', 'reddit', 'news_feed'],
            'initial_cash': 10000.0
        },
        {
            'agent_id': 'contrarian_trader',
            'personality': 'Contrarian value investor who bets against public opinion and looks for mispricing.',
            'subscriptions': ['twitter', 'expert_analysis'],
            'initial_cash': 10000.0
        },
        {
            'agent_id': 'well_informed_trader',
            'personality': 'Well-informed trader who monitors all sources and synthesizes information carefully.',
            'subscriptions': ['twitter', 'news_feed', 'expert_analysis', 'reddit'],
            'initial_cash': 10000.0
        },
        {
            'agent_id': 'social_trader',
            'personality': 'Social trader who primarily follows community sentiment and discussion.',
            'subscriptions': ['twitter', 'reddit', 'discord'],
            'initial_cash': 10000.0
        },
    ]

    # Limit to requested number of agents
    agent_configs = agent_configs[:num_agents]

    # Subscribe agents to portals
    for config in agent_configs:
        portal_network.subscribe_agent(config['agent_id'], config['subscriptions'])

    # Create agent factories
    agent_factories = []
    for config in agent_configs:
        def make_agent(cfg=config):
            return create_prediction_agent(
                agent_id=cfg['agent_id'],
                personality=cfg['personality'],
                initial_cash=cfg['initial_cash']
            )
        agent_factories.append(make_agent)

    return agent_factories


def build_simulation_engine(
    events_path: str,
    market_type: str = 'lmsr',
    num_agents: int = 3,
    timesteps: int = 50,
    liquidity_param: float = 100.0,
    run_name: str = 'synthetic_data_sim'
):
    """
    Build the complete simulation engine with all modules integrated.

    Args:
        events_path: Path to JSON file with event data
        market_type: 'lmsr' or 'orderbook'
        num_agents: Number of trading agents (1-5)
        timesteps: Maximum simulation timesteps
        liquidity_param: Market liquidity parameter
        run_name: Name for this simulation run

    Returns:
        Configured SimulationEngine ready to run
    """

    # 1. Create portal network with standard portals
    print("[SETUP] Setting up information portal network...")
    portal_network = create_standard_portals()

    # 2. Create agents and subscribe them to portals
    print(f"[AGENTS] Creating {num_agents} agents with different strategies...")
    agent_factories = create_agents_with_subscriptions(portal_network, num_agents)

    # 3. Create market
    print(f"[MARKET] Setting up {market_type.upper()} market...")
    if market_type.lower() == 'lmsr':
        market_factory = lambda: LMSRMarketAdapter(
            liquidity_param=liquidity_param,
            track_positions=True
        )
    else:
        market_factory = lambda: OrderBookMarketAdapter(
            market_id="prediction_market",
            initial_price=0.5,
            tick_size=0.01,
            track_positions=True
        )

    # 4. Create event stream from JSON file
    print(f"[DATA] Loading events from: {events_path}")
    stream_factory = lambda: create_event_stream(str(events_path))

    # 5. Configure simulation runtime
    runtime_config = SimulationRuntimeConfig(
        max_timesteps=timesteps,
        log_dir=Path("artifacts"),
        run_name=run_name,
        log_every=1,
        stop_when_stream_finishes=True,  # Stop when all events consumed
        enable_logging=True,
        save_logs_as_csv=True,
        save_logs_as_json=True
    )

    # 6. Build the engine
    print("[BUILD] Building simulation engine...")
    engine = SimulationEngine(
        stream_factory=stream_factory,
        portal_factory=lambda: portal_network,
        agent_factories=agent_factories,
        market_factory=market_factory,
        evaluator_factories=[],  # Add custom evaluators if needed
        runtime_config=runtime_config
    )

    print("[OK] Engine ready!\n")
    return engine


def main():
    # Load defaults from config.env
    env_config = load_env_config()

    parser = argparse.ArgumentParser(
        description="Run prediction market simulation with synthetic event data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use defaults from config.env
  python examples/run_with_synthetic_data.py

  # Override events file only
  python examples/run_with_synthetic_data.py --events data/other_events.json

  # Override multiple settings
  python examples/run_with_synthetic_data.py --market orderbook --agents 5 --seed 123
        """
    )
    parser.add_argument(
        '--events',
        type=str,
        default=env_config['events_file'],
        help=f"Path to JSON file with event data (default from config.env: {env_config['events_file']})"
    )
    parser.add_argument(
        '--market',
        type=str,
        choices=['lmsr', 'orderbook'],
        default=env_config['market_type'],
        help=f"Market type: lmsr or orderbook (default from config.env: {env_config['market_type']})"
    )
    parser.add_argument(
        '--agents',
        type=int,
        default=env_config['num_agents'],
        help=f"Number of agents 1-5 (default from config.env: {env_config['num_agents']})"
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=env_config['max_timesteps'],
        help=f"Maximum timesteps (default from config.env: {env_config['max_timesteps']})"
    )
    parser.add_argument(
        '--liquidity',
        type=float,
        default=env_config['liquidity_param'],
        help=f"Market liquidity parameter for LMSR (default from config.env: {env_config['liquidity_param']})"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=env_config['random_seed'],
        help=f"Random seed for reproducibility, -1 for random (default from config.env: {env_config['random_seed']})"
    )
    parser.add_argument(
        '--name',
        type=str,
        default=env_config['run_name'],
        help=f"Run name for output files (default from config.env: {env_config['run_name']})"
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help="Path to custom config.env file (optional)"
    )

    args = parser.parse_args()

    # Handle random seed
    if args.seed == -1:
        import random
        args.seed = random.randint(0, 999999)
        print(f"[RANDOM] Using random seed: {args.seed}")

    # Validate inputs
    if args.agents < 1 or args.agents > 5:
        print("[WARNING] Number of agents should be between 1 and 5")
        args.agents = max(1, min(5, args.agents))

    # Check if events file exists
    events_path = Path(args.events)
    if not events_path.exists():
        print(f"[ERROR] Events file not found: {events_path}")
        print(f"\nCreate a JSON file with this structure:")
        print("""
{
  "events": [
    {
      "event_id": "evt_001",
      "initial_time": 0,
      "source_nodes": ["twitter", "news_feed"],
      "tagline": "Market event headline",
      "description": "Detailed description of what happened..."
    }
  ]
}
        """)
        sys.exit(1)

    # Use provided run name (already has default from config.env)
    run_name = args.name

    # Print configuration
    print("\n" + "="*60)
    print("PREDICTION MARKET SIMULATION WITH SYNTHETIC DATA")
    print("="*60)
    print(f"Events file:    {events_path}")
    print(f"Market type:    {args.market.upper()}")
    print(f"Agents:         {args.agents}")
    print(f"Max timesteps:  {args.timesteps}")
    print(f"Liquidity:      {args.liquidity}")
    print(f"Random seed:    {args.seed}")
    print(f"Run name:       {run_name}")
    print("="*60 + "\n")

    # Build and run simulation
    try:
        engine = build_simulation_engine(
            events_path=str(events_path),
            market_type=args.market,
            num_agents=args.agents,
            timesteps=args.timesteps,
            liquidity_param=args.liquidity,
            run_name=run_name
        )

        print("[RUN] Starting simulation...\n")
        print("-" * 60)

        result = engine.run_once(run_id=1, seed=args.seed)

        print("\n" + "-" * 60)
        print("[DONE] Simulation complete!\n")

        # Display results
        print("RESULTS SUMMARY")
        print("="*60)
        if result.prices:
            print(f"Total timesteps:  {len(result.prices)}")
            print(f"Starting price:   {result.prices[0]:.4f}")
            print(f"Final price:      {result.prices[-1]:.4f}")
            print(f"Price change:     {result.prices[-1] - result.prices[0]:+.4f}")
            print(f"Price range:      [{min(result.prices):.4f}, {max(result.prices):.4f}]")
        else:
            print("No price data (no events or trades)")

        print("\nLog files saved:")
        for log_type, path in result.log_files.items():
            print(f"   {log_type}: {path}")

        print("\n" + "="*60)
        print("Simulation successful!")
        print("="*60)
        print("\nAnalyze results with:")
        print(f"  - CSV files in: artifacts/")
        print(f"  - Market data: artifacts/{run_name}_run1_market.csv")
        print(f"  - Agent beliefs: artifacts/{run_name}_run1_beliefs.csv")
        print(f"  - Event sources: artifacts/{run_name}_run1_sources.csv")

        # Plot and save PnL dashboard per agent if available
        if result.agent_pnl_history:
            dashboard_dir = Path("artifacts/dashboards")
            dashboard_dir.mkdir(parents=True, exist_ok=True)
            pnl_df = pd.DataFrame(result.agent_pnl_history)
            ax = pnl_df.plot(title=f"Agent PnL — {run_name}", figsize=(10, 5))
            ax.set_xlabel("Timestep")
            ax.set_ylabel("PnL")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            pnl_path = dashboard_dir / f"{run_name}_pnl.png"
            plt.savefig(pnl_path, dpi=150)
            plt.close()
            print(f"  - PnL chart: {pnl_path}")

        # Save trade log per agent if available
        if result.trade_log:
            trades_dir = Path("artifacts/trades")
            trades_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            for t in result.trade_log:
                if hasattr(t, "__dict__"):
                    rows.append({k: v for k, v in t.__dict__.items()})
                else:
                    try:
                        rows.append(dict(t))
                    except Exception:
                        rows.append({"trade": str(t)})
            trades_path = trades_dir / f"{run_name}_trades.csv"
            pd.DataFrame(rows).to_csv(trades_path, index=False)
            print(f"  - Trades log: {trades_path}")

    except Exception as e:
        print(f"\n[ERROR] Error during simulation: {e}")
        import traceback
        traceback.print_exc()

        print("\nTroubleshooting:")
        print("  1. Verify events JSON file has correct format")
        print("  2. Check that source_nodes in events match: twitter, news_feed, expert_analysis, reddit, discord")
        print("  3. Ensure Ollama is running for LLM agents: ollama serve")
        print("  4. Check that all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
