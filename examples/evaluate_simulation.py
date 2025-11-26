"""
Evaluate prediction market simulation results.
Works with output from run_with_synthetic_data.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_simulation_results(artifacts_dir: Path, run_name: str):
    """Load simulation logs from artifacts directory"""

    # Find the run files (handle run1, run2 suffixes)
    market_files = list(artifacts_dir.glob(f"{run_name}_run*_market.json"))
    beliefs_files = list(artifacts_dir.glob(f"{run_name}_run*_beliefs.json"))
    trades_file = artifacts_dir / "trades" / f"{run_name}_trades.csv"

    if not market_files:
        print(f"[ERROR] No market files found matching: {run_name}_run*_market.json")
        return None

    # Use the latest run
    market_file = sorted(market_files)[-1]
    beliefs_file = sorted(beliefs_files)[-1] if beliefs_files else None

    print(f"Loading market data from: {market_file}")

    with open(market_file, 'r') as f:
        market_data = json.load(f)

    beliefs_data = None
    if beliefs_file and beliefs_file.exists():
        with open(beliefs_file, 'r') as f:
            beliefs_data = json.load(f)

    trades_data = None
    if trades_file.exists():
        trades_data = pd.read_csv(trades_file)
        print(f"Loading trades from: {trades_file}")

    return {
        'market': market_data,
        'beliefs': beliefs_data,
        'trades': trades_data
    }


def calculate_brier_score(predicted_probs: list, actual_outcome: float) -> float:
    """
    Brier Score: Mean squared error for probabilistic predictions.
    Lower is better. Range: [0, 1]
    """
    predicted_probs = np.array(predicted_probs)
    return float(np.mean((predicted_probs - actual_outcome) ** 2))


def calculate_log_score(predicted_probs: list, actual_outcome: float) -> float:
    """
    Logarithmic Score: Measures calibration of probabilistic predictions.
    Higher (less negative) is better.
    """
    predicted_probs = np.array(predicted_probs)
    # Clip to avoid log(0)
    predicted_probs = np.clip(predicted_probs, 1e-10, 1 - 1e-10)

    if actual_outcome >= 0.5:  # YES outcome
        return float(np.mean(np.log(predicted_probs)))
    else:  # NO outcome
        return float(np.mean(np.log(1 - predicted_probs)))


def calculate_agent_pnl(trades: pd.DataFrame, final_price: float) -> dict:
    """
    Calculate PnL for each agent based on their trades.

    PnL = (shares * final_price) - cost_paid
    For YES shares: value = shares * final_price
    For NO shares: value = shares * (1 - final_price)
    """
    agent_pnls = {}

    for agent_id in trades['agent_id'].unique():
        agent_trades = trades[trades['agent_id'] == agent_id]

        total_cost = 0
        yes_shares = 0
        no_shares = 0

        for _, trade in agent_trades.iterrows():
            total_cost += trade['cost']
            if trade['outcome'] == 'YES':
                yes_shares += trade['shares']
            else:
                no_shares += trade['shares']

        # Calculate final value
        yes_value = yes_shares * final_price
        no_value = no_shares * (1 - final_price)
        total_value = yes_value + no_value

        pnl = total_value - total_cost

        agent_pnls[agent_id] = {
            'total_cost': total_cost,
            'yes_shares': yes_shares,
            'no_shares': no_shares,
            'yes_value': yes_value,
            'no_value': no_value,
            'total_value': total_value,
            'pnl': pnl,
            'return_pct': (pnl / total_cost * 100) if total_cost > 0 else 0
        }

    return agent_pnls


def calculate_price_volatility(prices: list) -> float:
    """Calculate price volatility as standard deviation of returns"""
    if len(prices) < 2:
        return 0.0
    returns = np.diff(prices)
    return float(np.std(returns))


def calculate_price_discovery_speed(prices: list, final_price: float, threshold: float = 0.05) -> int:
    """
    Calculate how many timesteps until price is within threshold of final price.
    """
    for i, price in enumerate(prices):
        if abs(price - final_price) <= threshold:
            return i
    return len(prices)


def evaluate_simulation(
    artifacts_dir: Path,
    run_name: str,
    actual_outcome: float = 1.0,  # 1.0 = YES happened, 0.0 = NO happened
    outcome_name: str = "YES"
):
    """
    Comprehensive evaluation of simulation results.

    Args:
        artifacts_dir: Directory containing simulation artifacts
        run_name: Name of the simulation run
        actual_outcome: Actual probability of the outcome (1.0 if YES won, 0.0 if NO won)
        outcome_name: Name of the winning outcome for display
    """
    print("\n" + "=" * 70)
    print(f"PREDICTION MARKET SIMULATION EVALUATION")
    print(f"Run: {run_name}")
    print("=" * 70)

    # Load data
    results = load_simulation_results(artifacts_dir, run_name)
    if not results:
        return None

    market_data = results['market']
    beliefs_data = results['beliefs']
    trades_data = results['trades']

    # Extract price history
    prices = [m['price'] for m in market_data]
    timesteps = [m['timestep'] for m in market_data]

    final_price = prices[-1]
    initial_price = prices[0]

    print(f"\n[MARKET SUMMARY]")
    print("-" * 70)
    print(f"Timesteps:        {len(timesteps)}")
    print(f"Initial Price:    {initial_price:.4f}")
    print(f"Final Price:      {final_price:.4f}")
    print(f"Price Change:     {final_price - initial_price:+.4f}")
    print(f"Price Range:      [{min(prices):.4f}, {max(prices):.4f}]")

    # Calculate evaluation metrics
    report = {}

    # 1. Prediction Quality Metrics
    print(f"\n[1] PREDICTION QUALITY METRICS")
    print("-" * 70)

    brier_score = calculate_brier_score(prices, actual_outcome)
    log_score = calculate_log_score(prices, actual_outcome)

    report['brier_score'] = brier_score
    report['log_score'] = log_score

    print(f"Brier Score:      {brier_score:.4f}  (lower = better, 0 = perfect)")
    print(f"Log Score:        {log_score:.4f}  (higher = better, 0 = perfect)")
    print(f"Actual Outcome:   {outcome_name} (prob={actual_outcome})")
    print(f"Prediction Error: {abs(final_price - actual_outcome):.4f}")

    # 2. Market Efficiency
    print(f"\n[2] MARKET EFFICIENCY")
    print("-" * 70)

    volatility = calculate_price_volatility(prices)
    discovery_speed = calculate_price_discovery_speed(prices, actual_outcome, threshold=0.1)

    total_volume = market_data[-1].get('total_volume', 0)
    total_trades = market_data[-1].get('num_trades', 0)

    report['volatility'] = volatility
    report['discovery_speed'] = discovery_speed
    report['total_volume'] = total_volume
    report['total_trades'] = total_trades

    print(f"Price Volatility:        {volatility:.4f}")
    print(f"Price Discovery Speed:   {discovery_speed} timesteps (to within 10% of outcome)")
    print(f"Total Volume:            {total_volume:,.0f} shares")
    print(f"Total Trades:            {total_trades}")

    # 3. Agent Performance
    if trades_data is not None and len(trades_data) > 0:
        print(f"\n[3] AGENT PERFORMANCE")
        print("-" * 70)

        agent_pnls = calculate_agent_pnl(trades_data, actual_outcome)
        report['agent_performance'] = agent_pnls

        print(f"{'Agent':<25} {'Cost':>12} {'Value':>12} {'PnL':>12} {'Return':>10}")
        print("-" * 70)

        for agent_id, metrics in sorted(agent_pnls.items(), key=lambda x: x[1]['pnl'], reverse=True):
            print(f"{agent_id:<25} ${metrics['total_cost']:>10,.2f} ${metrics['total_value']:>10,.2f} "
                  f"${metrics['pnl']:>10,.2f} {metrics['return_pct']:>9.1f}%")

        # Agent trading stats
        print(f"\n[4] AGENT TRADING BEHAVIOR")
        print("-" * 70)

        for agent_id in trades_data['agent_id'].unique():
            agent_trades = trades_data[trades_data['agent_id'] == agent_id]

            total_shares = agent_trades['shares'].sum()
            avg_price = (agent_trades['cost'] / agent_trades['shares']).mean()
            yes_trades = len(agent_trades[agent_trades['outcome'] == 'YES'])
            no_trades = len(agent_trades[agent_trades['outcome'] == 'NO'])

            print(f"\n{agent_id}:")
            print(f"  Total Trades:    {len(agent_trades)}")
            print(f"  Total Shares:    {total_shares:,.0f}")
            print(f"  YES Trades:      {yes_trades}")
            print(f"  NO Trades:       {no_trades}")
            print(f"  Avg Cost/Share:  ${avg_price:.4f}")

    # 4. Belief Analysis
    if beliefs_data:
        print(f"\n[5] AGENT BELIEF ANALYSIS")
        print("-" * 70)

        belief_by_agent = {}
        for entry in beliefs_data:
            agent_id = entry['agent_id']
            if agent_id not in belief_by_agent:
                belief_by_agent[agent_id] = []
            belief_by_agent[agent_id].append(entry['belief'])

        for agent_id, beliefs in belief_by_agent.items():
            avg_belief = np.mean(beliefs)
            belief_std = np.std(beliefs)
            belief_error = abs(avg_belief - actual_outcome)

            print(f"{agent_id}:")
            print(f"  Avg Belief:      {avg_belief:.4f}")
            print(f"  Belief Std Dev:  {belief_std:.4f}")
            print(f"  Belief Error:    {belief_error:.4f} (vs actual outcome)")

    # Summary
    print(f"\n[SUMMARY]")
    print("=" * 70)
    report['final_price'] = final_price
    report['actual_outcome'] = actual_outcome
    report['prediction_error'] = abs(final_price - actual_outcome)

    accuracy_rating = "EXCELLENT" if report['prediction_error'] < 0.05 else \
                      "GOOD" if report['prediction_error'] < 0.1 else \
                      "FAIR" if report['prediction_error'] < 0.2 else "POOR"

    print(f"Final Price:       {final_price:.4f}")
    print(f"Actual Outcome:    {actual_outcome:.4f}")
    print(f"Prediction Error:  {report['prediction_error']:.4f}")
    print(f"Accuracy Rating:   {accuracy_rating}")

    return report


def plot_evaluation(artifacts_dir: Path, run_name: str, output_dir: Path):
    """Generate evaluation visualization plots"""

    results = load_simulation_results(artifacts_dir, run_name)
    if not results:
        return

    market_data = results['market']
    trades_data = results['trades']

    prices = [m['price'] for m in market_data]
    timesteps = [m['timestep'] for m in market_data]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Price Evolution with Confidence Bands
    ax1 = axes[0, 0]
    ax1.plot(timesteps, prices, linewidth=2, color='#2E86AB', marker='o')
    ax1.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='YES Win (1.0)')
    ax1.axhline(y=0.0, color='red', linestyle='--', alpha=0.5, label='NO Win (0.0)')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Market Price')
    ax1.set_title('Price Evolution')
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Volume Over Time
    ax2 = axes[0, 1]
    volumes = [m.get('tick_volume', 0) for m in market_data]
    ax2.bar(timesteps, volumes, color='#A23B72', alpha=0.8)
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Volume (Shares)')
    ax2.set_title('Trading Volume per Timestep')
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Agent Trading Volume
    if trades_data is not None:
        ax3 = axes[1, 0]
        agent_volumes = trades_data.groupby('agent_id')['shares'].sum()
        colors = plt.cm.Set2(np.linspace(0, 1, len(agent_volumes)))
        ax3.bar(agent_volumes.index, agent_volumes.values, color=colors)
        ax3.set_xlabel('Agent')
        ax3.set_ylabel('Total Shares Traded')
        ax3.set_title('Trading Volume by Agent')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Cumulative Returns
    ax4 = axes[1, 1]
    returns = np.diff(prices)
    cumulative_returns = np.cumsum(returns)
    ax4.plot(timesteps[1:], cumulative_returns, linewidth=2, color='#2E86AB')
    ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax4.fill_between(timesteps[1:], 0, cumulative_returns,
                     where=np.array(cumulative_returns)>=0, color='green', alpha=0.3)
    ax4.fill_between(timesteps[1:], 0, cumulative_returns,
                     where=np.array(cumulative_returns)<0, color='red', alpha=0.3)
    ax4.set_xlabel('Timestep')
    ax4.set_ylabel('Cumulative Price Change')
    ax4.set_title('Cumulative Price Movement')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_name}_evaluation.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\n[PLOT] Saved evaluation plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate prediction market simulation')
    parser.add_argument('--run-name', type=str, default='prediction_sim',
                        help='Name of the simulation run')
    parser.add_argument('--artifacts-dir', type=str, default='artifacts',
                        help='Directory containing simulation artifacts')
    parser.add_argument('--actual-outcome', type=float, default=1.0,
                        help='Actual outcome probability (1.0=YES won, 0.0=NO won)')
    parser.add_argument('--outcome-name', type=str, default='YES',
                        help='Name of the winning outcome')
    parser.add_argument('--plot', action='store_true',
                        help='Generate evaluation plots')

    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)

    # Run evaluation
    report = evaluate_simulation(
        artifacts_dir=artifacts_dir,
        run_name=args.run_name,
        actual_outcome=args.actual_outcome,
        outcome_name=args.outcome_name
    )

    # Generate plots if requested
    if args.plot and report:
        plot_evaluation(
            artifacts_dir=artifacts_dir,
            run_name=args.run_name,
            output_dir=artifacts_dir / "dashboards"
        )

    return report


if __name__ == "__main__":
    main()
