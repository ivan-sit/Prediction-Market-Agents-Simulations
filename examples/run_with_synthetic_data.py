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
import yaml

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
        'num_agents': None,  # None means use all agents from YAML or default to 3
        'max_timesteps': 100,
        'liquidity_param': 100.0,
        'run_name': 'prediction_sim',
        'random_seed': 42,
        'log_dir': 'simulation_logs',
        'personas_yaml': None,
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
                    'PERSONAS_YAML': ('personas_yaml', str),
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


def load_agents_from_yaml(yaml_path: Path):
    """
    Load agent configurations from YAML file.

    Expected format:
      agents:
        - agent_id: ...
          personality_prompt: ...
          subscriptions: [...]
          net_worth: "$X"
          ...

    Returns:
        List of agent config dicts
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    agent_configs = []
    for agent in data.get('agents', []):
        # Convert net_worth string like "$150,000" to float
        net_worth_str = agent.get('net_worth', '$10000')
        initial_cash = float(net_worth_str.replace('$', '').replace(',', ''))

        agent_configs.append({
            'agent_id': agent['agent_id'],
            'personality': agent['personality_prompt'],
            'subscriptions': agent['subscriptions'],
            'initial_cash': initial_cash
        })

    return agent_configs


def create_agents_with_subscriptions(portal_network, num_agents=None, yaml_path=None):
    """
    Create agents with different personalities and portal subscriptions.

    Args:
        portal_network: Portal network to subscribe agents to
        num_agents: Number of agents to create (limits from config). If None and using YAML, uses all agents from YAML.
        yaml_path: Optional path to YAML file with agent personas

    Returns:
        List of agent factory functions
    """
    # Try to load from YAML if provided
    if yaml_path and Path(yaml_path).exists():
        print(f"Loading agents from: {yaml_path}")
        base_agent_configs = load_agents_from_yaml(Path(yaml_path))
        print(f"Found {len(base_agent_configs)} agent personas in YAML")

        # If num_agents not specified, use all agents from YAML
        if num_agents is None:
            num_agents = len(base_agent_configs)
            print(f"Using all {num_agents} agents from YAML file")
    else:
        # Fallback to hardcoded configs
        if num_agents is None:
            num_agents = 3  # Default for hardcoded agents

        base_agent_configs = [
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

    # Build agent_configs with duplicates if needed
    agent_configs = []
    num_base = len(base_agent_configs)
    for i in range(num_agents):
        base_config = base_agent_configs[i % num_base].copy()
        # Add suffix for duplicates to create unique agent IDs
        if i >= num_base:
            duplicate_num = (i // num_base) + 1
            base_config['agent_id'] = f"{base_config['agent_id']}_v{duplicate_num}"
        agent_configs.append(base_config)

    if num_agents > num_base:
        print(f"Creating {num_agents} agents ({num_base} unique personas + {num_agents - num_base} duplicates)")
    else:
        print(f"Creating {num_agents} agents")

    # Subscribe agents to portals and collect subscription data
    subscriptions_data = []
    for config in agent_configs:
        portal_network.subscribe_agent(config['agent_id'], config['subscriptions'])
        subscriptions_data.append({
            'agent_id': config['agent_id'],
            'subscriptions': config['subscriptions'],
        })

    # Store subscriptions for later use (animation export)
    portal_network._agent_subscriptions_data = subscriptions_data

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
    run_name: str = 'synthetic_data_sim',
    read_only: bool = True,
    personas_yaml: str = None
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
        read_only: Whether to run in read-only mode
        personas_yaml: Optional path to YAML file with agent personas

    Returns:
        Configured SimulationEngine ready to run
    """

    # 1. Create portal network with standard portals
    print("Setting up information portal network...")
    portal_network = create_standard_portals()

    # 2. Create agents and subscribe them to portals
    print(f"Creating {num_agents} agents with different strategies...")
    agent_factories = create_agents_with_subscriptions(portal_network, num_agents, yaml_path=personas_yaml)

    # 3. Create market
    print(f"Setting up {market_type.upper()} market...")
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
    print(f"Loading events from: {events_path}")
    stream_factory = lambda: create_event_stream(str(events_path), read_only=read_only)

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
    print("Building simulation engine...")
    engine = SimulationEngine(
        stream_factory=stream_factory,
        portal_factory=lambda: portal_network,
        agent_factories=agent_factories,
        market_factory=market_factory,
        evaluator_factories=[],  # Add custom evaluators if needed
        runtime_config=runtime_config
    )

    # 7. Save agent subscriptions for animation visualization
    subscriptions_file = runtime_config.log_dir / f"{run_name}_run1_subscriptions.json"
    if hasattr(portal_network, '_agent_subscriptions_data'):
        import json
        with open(subscriptions_file, 'w') as f:
            json.dump(portal_network._agent_subscriptions_data, f, indent=2)

    print("Engine ready!\n")
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
        help=f"Number of agents to use. If personas YAML is provided and this is not set, uses all agents from YAML. (default: {env_config['num_agents'] if env_config['num_agents'] is not None else 'all from YAML or 3'})"
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
    parser.add_argument(
        '--no-read-only',
        dest='read_only',
        action='store_false',
        help='Allow event file to be consumed during simulation (default: read-only mode enabled)'
    )
    parser.add_argument(
        '--personas',
        type=str,
        default=env_config['personas_yaml'],
        help=f"Path to YAML file with agent personas (default from config.env: {env_config['personas_yaml']})"
    )
    parser.set_defaults(read_only=True)

    args = parser.parse_args()

    # Handle random seed
    if args.seed == -1:
        import random
        args.seed = random.randint(0, 999999)
        print(f"[RANDOM] Using random seed: {args.seed}")

    # Validate inputs
    if args.agents is not None and args.agents < 1:
        print("Warning: Number of agents must be at least 1")
        args.agents = 1

    # Check if events file exists
    events_path = Path(args.events)
    if not events_path.exists():
        print(f"Error: Events file not found: {events_path}")
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
    if args.agents is None and args.personas:
        print(f"Agents:         All from YAML ({args.personas})")
    elif args.agents is None:
        print(f"Agents:         3 (default)")
    else:
        print(f"Agents:         {args.agents}")
    print(f"Max timesteps:  {args.timesteps}")
    print(f"Liquidity:      {args.liquidity}")
    print(f"Random seed:    {args.seed}")
    print(f"Run name:       {run_name}")
    if args.personas:
        print(f"Personas:       {args.personas}")
    print("="*60 + "\n")

    # Build and run simulation
    try:
        engine = build_simulation_engine(
            events_path=str(events_path),
            market_type=args.market,
            num_agents=args.agents,
            timesteps=args.timesteps,
            liquidity_param=args.liquidity,
            run_name=run_name,
            read_only=args.read_only,
            personas_yaml=args.personas
        )

        print("Starting simulation...\n")
        print("-" * 60)

        result = engine.run_once(run_id=1, seed=args.seed)

        print("\n" + "-" * 60)
        print("Simulation complete!\n")

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

        # Get agent colors for consistent styling across charts
        agent_ids = list(pnl_df.columns) if result.agent_pnl_history else []
        agent_colors = {agent: f"C{i}" for i, agent in enumerate(agent_ids)}

        # 1. Price Trend Chart
        if result.prices:
            fig, ax = plt.subplots(figsize=(10, 5))
            timesteps = list(range(len(result.prices)))
            ax.plot(timesteps, result.prices, linewidth=2, color='#2E86AB', marker='o', markersize=4)
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Price")
            ax.set_title(f"Market Price — {run_name}")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            price_path = dashboard_dir / f"{run_name}_price.png"
            plt.savefig(price_path, dpi=150)
            plt.close()
            print(f"  - Price chart: {price_path}")

        # 2. Volume & Order Flow Chart
        if result.market_snapshots:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            timesteps = [s.get('timestep', i) for i, s in enumerate(result.market_snapshots)]
            volumes = [s.get('tick_volume', 0) for s in result.market_snapshots]
            net_flows = [s.get('net_flow', 0) for s in result.market_snapshots]

            # Volume bars
            ax1.bar(timesteps, volumes, color='#A23B72', alpha=0.8)
            ax1.set_ylabel("Volume (shares)")
            ax1.set_title(f"Trading Volume & Order Flow — {run_name}")
            ax1.grid(True, alpha=0.3)

            # Net flow bars (green=buy, red=sell)
            colors = ['green' if nf >= 0 else 'red' for nf in net_flows]
            ax2.bar(timesteps, net_flows, color=colors, alpha=0.8)
            ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax2.set_xlabel("Timestep")
            ax2.set_ylabel("Net Flow (+ buy / - sell)")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            volume_path = dashboard_dir / f"{run_name}_volume.png"
            plt.savefig(volume_path, dpi=150)
            plt.close()
            print(f"  - Volume chart: {volume_path}")

        # 3. Agent Beliefs Chart
        if result.belief_history:
            fig, ax = plt.subplots(figsize=(10, 5))

            # Agent beliefs
            belief_by_agent = {}
            for t, beliefs in enumerate(result.belief_history):
                for agent_id, belief in beliefs.items():
                    if agent_id not in belief_by_agent:
                        belief_by_agent[agent_id] = {}
                    belief_by_agent[agent_id][t] = belief

            for agent_id, beliefs in belief_by_agent.items():
                ts = sorted(beliefs.keys())
                vals = [beliefs[t] for t in ts]
                color = agent_colors.get(agent_id, None)
                ax.plot(ts, vals, linewidth=2, label=agent_id, color=color, marker='o', markersize=4)

            ax.set_xlabel("Timestep")
            ax.set_ylabel("Belief (Probability)")
            ax.set_title(f"Agent Beliefs — {run_name}")
            ax.set_ylim(0, 1)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            beliefs_path = dashboard_dir / f"{run_name}_beliefs.png"
            plt.savefig(beliefs_path, dpi=150)
            plt.close()
            print(f"  - Beliefs chart: {beliefs_path}")

        # 4. Price Returns Chart
        if result.prices and len(result.prices) > 1:
            fig, ax = plt.subplots(figsize=(10, 5))
            returns = [result.prices[i] - result.prices[i-1] for i in range(1, len(result.prices))]
            timesteps = list(range(1, len(result.prices)))
            colors = ['green' if r >= 0 else 'red' for r in returns]

            ax.bar(timesteps, returns, color=colors, alpha=0.8)
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Price Change")
            ax.set_title(f"Price Returns (Period-over-Period) — {run_name}")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            returns_path = dashboard_dir / f"{run_name}_returns.png"
            plt.savefig(returns_path, dpi=150)
            plt.close()
            print(f"  - Returns chart: {returns_path}")

        # 5. Trade Volume Chart (shares traded per agent per timestep)
        if result.trade_log:
            fig, ax = plt.subplots(figsize=(10, 5))

            # Sum shares traded per timestep per agent
            trade_volumes = {}
            for trade in result.trade_log:
                t = getattr(trade, 'timestamp', 0) if hasattr(trade, 'timestamp') else trade.get('timestamp', 0)
                agent = getattr(trade, 'agent_id', 'unknown') if hasattr(trade, 'agent_id') else trade.get('agent_id', 'unknown')
                shares = getattr(trade, 'shares', 0) if hasattr(trade, 'shares') else trade.get('shares', 0)
                if t not in trade_volumes:
                    trade_volumes[t] = {}
                trade_volumes[t][agent] = trade_volumes[t].get(agent, 0) + abs(shares)

            if trade_volumes:
                timesteps = sorted(trade_volumes.keys())
                agents = sorted(set(a for tv in trade_volumes.values() for a in tv.keys()))

                x = range(len(timesteps))
                width = 0.8 / len(agents) if agents else 0.8

                for i, agent in enumerate(agents):
                    volumes = [trade_volumes.get(t, {}).get(agent, 0) for t in timesteps]
                    offset = (i - len(agents)/2 + 0.5) * width
                    color = agent_colors.get(agent, f"C{i}")
                    ax.bar([xi + offset for xi in x], volumes, width, label=agent, color=color, alpha=0.8)

                ax.set_xticks(x)
                ax.set_xticklabels(timesteps)
                ax.set_xlabel("Timestep")
                ax.set_ylabel("Shares Traded")
                ax.set_title(f"Trading Volume by Agent — {run_name}")
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                trades_chart_path = dashboard_dir / f"{run_name}_trades.png"
                plt.savefig(trades_chart_path, dpi=150)
                plt.close()
                print(f"  - Trade volume chart: {trades_chart_path}")

        # 6. Agent Net Position Over Time Chart
        if result.market_snapshots and any('positions' in s for s in result.market_snapshots):
            fig, ax = plt.subplots(figsize=(10, 5))

            positions_over_time = {}
            for t, snapshot in enumerate(result.market_snapshots):
                positions = snapshot.get('positions', {})
                for agent_id, pos in positions.items():
                    if agent_id not in positions_over_time:
                        positions_over_time[agent_id] = {'timesteps': [], 'net': []}
                    positions_over_time[agent_id]['timesteps'].append(t)
                    # Net position: YES shares - NO shares (positive = bullish, negative = bearish)
                    net_pos = pos.get('YES', 0) - pos.get('NO', 0)
                    positions_over_time[agent_id]['net'].append(net_pos)

            if positions_over_time:
                for agent_id, data in positions_over_time.items():
                    color = agent_colors.get(agent_id, None)
                    ax.plot(data['timesteps'], data['net'], linewidth=2, label=agent_id,
                           color=color, marker='o', markersize=4)

                ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
                ax.set_xlabel("Timestep")
                ax.set_ylabel("Net Position (YES - NO shares)")
                ax.set_title(f"Agent Net Position — {run_name}")
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                positions_path = dashboard_dir / f"{run_name}_positions.png"
                plt.savefig(positions_path, dpi=150)
                plt.close()
                print(f"  - Positions chart: {positions_path}")

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
