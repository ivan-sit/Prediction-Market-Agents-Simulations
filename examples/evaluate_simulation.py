"""
Example script for evaluating prediction market simulation results.
Demonstrates proper evaluation following AgentSociety Challenge methodology.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prediction_market_sim.evaluation_metrics import (
    PredictionMarketEvaluator,
    run_ablation_study
)


def load_simulation_results(log_dir: Path, run_name: str):
    """Load simulation logs and extract data for evaluation"""

    # Load CSV logs
    trades_file = log_dir / f"{run_name}_trades.csv"
    prices_file = log_dir / f"{run_name}_prices.csv"
    agents_file = log_dir / f"{run_name}_agents.csv"

    if not trades_file.exists():
        print(f"[ERROR] Trades file not found: {trades_file}")
        return None

    trades = pd.read_csv(trades_file)
    prices = pd.read_csv(prices_file) if prices_file.exists() else None
    agents = pd.read_csv(agents_file) if agents_file.exists() else None

    return {
        'trades': trades,
        'prices': prices,
        'agents': agents
    }


def extract_agent_pnls_from_logs(trades: pd.DataFrame) -> dict:
    """
    Extract PnL history for each agent from trade logs.

    Returns:
        Dict mapping agent_id to list of cumulative PnL values
    """
    agent_pnls = {}

    for agent_id in trades['agent_id'].unique():
        agent_trades = trades[trades['agent_id'] == agent_id].sort_values('timestep')

        # Calculate cumulative PnL
        cumulative_pnl = agent_trades['pnl'].cumsum().tolist()

        agent_pnls[agent_id] = cumulative_pnl if cumulative_pnl else [0.0]

    return agent_pnls


def prepare_ground_truth(events_file: Path, actual_winner: str, actual_prob: float) -> dict:
    """
    Prepare ground truth data for evaluation.

    Args:
        events_file: Path to events JSON file
        actual_winner: The actual winner (e.g., "Mamdani")
        actual_prob: Final probability/vote share (e.g., 0.504 for 50.4%)

    Returns:
        Ground truth dictionary
    """
    with open(events_file, 'r') as f:
        events_data = json.load(f)

    return {
        'winner': actual_winner,
        'final_prob': actual_prob,
        'events': events_data['events']
    }


def evaluate_nyc_mayor_simulation(log_dir: Path, run_name: str = "nyc_mayor_sim"):
    """
    Evaluate NYC Mayoral Election simulation with ground truth.
    """
    print("=" * 70)
    print("NYC MAYORAL ELECTION SIMULATION EVALUATION")
    print("=" * 70)

    # Load simulation results
    results = load_simulation_results(log_dir, run_name)
    if not results:
        return

    trades = results['trades']
    prices = results['prices']

    # Ground truth: Mamdani won with 50.4%
    ground_truth = prepare_ground_truth(
        events_file=Path("data/nyc_mayoral_election_2025.json"),
        actual_winner="Mamdani",
        actual_prob=0.504
    )

    # Extract agent PnLs
    agent_pnls = extract_agent_pnls_from_logs(trades)

    # Create evaluator
    evaluator = PredictionMarketEvaluator(
        simulation_logs=trades,
        ground_truth=ground_truth
    )

    # Generate comprehensive evaluation report
    report = evaluator.generate_evaluation_report(
        market_prices=prices,
        agent_pnls=agent_pnls,
        trades=trades
    )

    # Print results
    print("\n[1] PREDICTION QUALITY METRICS (analogous to RMSE)")
    print("-" * 70)
    print(f"Brier Score:     {report['brier_score']:.4f}  (lower = better calibration)")
    print(f"Log Score:       {report['log_score']:.4f}  (higher = better)")

    print("\n[2] MARKET ACCURACY (analogous to HR@K)")
    print("-" * 70)
    print(f"Accuracy @10:    {report['prediction_accuracy_at_10']:.2f}  (1.0 = correct winner in final 10 steps)")
    print(f"Accuracy @50:    {report['prediction_accuracy_at_50']:.2f}  (1.0 = correct winner in final 50 steps)")

    print("\n[3] AGENT PROFITABILITY (analogous to sentiment alignment)")
    print("-" * 70)
    for agent_id, metrics in sorted(report['agent_profitability'].items(),
                                    key=lambda x: x[1]['final_pnl'], reverse=True):
        print(f"{agent_id:30s}  PnL: ${metrics['final_pnl']:>10,.2f}  "
              f"Sharpe: {metrics['sharpe_ratio']:>6.2f}  "
              f"Max DD: ${metrics['max_drawdown']:>8,.2f}")

    print("\n[4] TRADING BEHAVIOR BY PERSONALITY")
    print("-" * 70)
    for agent_type, behavior in report['trading_behavior'].items():
        print(f"\n{agent_type}:")
        print(f"  Total Trades:       {behavior['total_trades']}")
        print(f"  Avg Position Size:  {behavior['avg_position_size']:.2f}")
        print(f"  Trade Frequency:    {behavior['trade_frequency']:.4f} trades/timestep")
        print(f"  Contrarian Ratio:   {behavior['contrarian_ratio']:.2%}")
        print(f"  Momentum Ratio:     {behavior['momentum_ratio']:.2%}")

    print("\n[5] MARKET EFFICIENCY")
    print("-" * 70)
    efficiency = report['market_efficiency']
    print(f"Avg Price Discovery Speed: {efficiency['avg_price_discovery_speed']:.2f} timesteps")
    print(f"Price Volatility:          {efficiency['price_volatility']:.4f}")

    print("\n[6] SUMMARY")
    print("-" * 70)
    print(f"Total Trades:          {report['total_trades']}")
    print(f"Final Market Price:    {report['final_market_price']:.4f}")
    print(f"Actual Outcome:        {report['actual_outcome_prob']:.4f}")
    print(f"Price Error:           {abs(report['final_market_price'] - report['actual_outcome_prob']):.4f}")

    return report


def evaluate_nba_finals_simulation(log_dir: Path, run_name: str = "nba_finals_sim"):
    """
    Evaluate NBA Finals simulation with ground truth.
    """
    print("\n" + "=" * 70)
    print("NBA FINALS 2025 SIMULATION EVALUATION")
    print("=" * 70)

    # Load simulation results
    results = load_simulation_results(log_dir, run_name)
    if not results:
        return

    trades = results['trades']
    prices = results['prices']

    # Ground truth: Thunder won 4-3
    ground_truth = prepare_ground_truth(
        events_file=Path("data/nba_finals_2025_okc_pacers.json"),
        actual_winner="Thunder",
        actual_prob=1.0  # Binary outcome: Thunder won
    )

    agent_pnls = extract_agent_pnls_from_logs(trades)

    evaluator = PredictionMarketEvaluator(
        simulation_logs=trades,
        ground_truth=ground_truth
    )

    report = evaluator.generate_evaluation_report(
        market_prices=prices,
        agent_pnls=agent_pnls,
        trades=trades
    )

    # Similar printing as NYC evaluation
    print("\n[PREDICTION QUALITY]")
    print(f"Brier Score: {report['brier_score']:.4f}")
    print(f"Log Score:   {report['log_score']:.4f}")

    print("\n[MARKET ACCURACY]")
    print(f"Correctly predicted Thunder win in final 10 steps: {report['prediction_accuracy_at_10']}")

    print("\n[TOP PERFORMING AGENTS]")
    top_agents = sorted(report['agent_profitability'].items(),
                       key=lambda x: x[1]['final_pnl'], reverse=True)[:5]
    for agent_id, metrics in top_agents:
        print(f"  {agent_id}: ${metrics['final_pnl']:,.2f}")

    return report


def run_ablation_studies():
    """
    Run ablation studies as required by project spec:
    - With/without memory
    - Varying personality prompts
    - Different exploration strategies
    """
    print("\n" + "=" * 70)
    print("ABLATION STUDY: TESTING DIFFERENT CONFIGURATIONS")
    print("=" * 70)

    configurations = [
        {
            'name': 'baseline_no_memory',
            'memory_enabled': False,
            'personality_type': 'generic',
            'exploration': 'random'
        },
        {
            'name': 'with_memory',
            'memory_enabled': True,
            'personality_type': 'generic',
            'exploration': 'random'
        },
        {
            'name': 'custom_personalities',
            'memory_enabled': True,
            'personality_type': 'custom',
            'exploration': 'random'
        },
        {
            'name': 'contrarian_strategy',
            'memory_enabled': True,
            'personality_type': 'contrarian',
            'exploration': 'strategic'
        }
    ]

    print("\nConfigurations to test:")
    for i, config in enumerate(configurations, 1):
        print(f"  {i}. {config['name']}")
        print(f"     - Memory: {config['memory_enabled']}")
        print(f"     - Personality: {config['personality_type']}")
        print(f"     - Exploration: {config['exploration']}")

    print("\n[NOTE] To run ablation study, execute simulations with each config")
    print("       Then use evaluator.compare_with_baseline() to compare results")

    # Example comparison
    print("\nExample comparison code:")
    print("""
    # After running baseline and experimental simulations:
    baseline_pnls = extract_agent_pnls_from_logs(baseline_trades)
    experimental_pnls = extract_agent_pnls_from_logs(experimental_trades)

    comparison = evaluator.compare_with_baseline(baseline_pnls, experimental_pnls)
    print(f"Improvement: {comparison['percent_improvement']:.2f}%")
    """)


def plot_evaluation_comparison(nyc_report: dict, nba_report: dict):
    """
    Visualize evaluation metrics across both simulations.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Brier Scores
    ax1 = axes[0, 0]
    scenarios = ['NYC Mayor', 'NBA Finals']
    brier_scores = [nyc_report['brier_score'], nba_report['brier_score']]
    ax1.bar(scenarios, brier_scores, color=['#1f77b4', '#ff7f0e'])
    ax1.set_ylabel('Brier Score (lower = better)')
    ax1.set_title('Prediction Quality Comparison')
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Accuracy @K
    ax2 = axes[0, 1]
    x = np.arange(len(scenarios))
    width = 0.35
    acc_10 = [nyc_report['prediction_accuracy_at_10'], nba_report['prediction_accuracy_at_10']]
    acc_50 = [nyc_report['prediction_accuracy_at_50'], nba_report['prediction_accuracy_at_50']]

    ax2.bar(x - width/2, acc_10, width, label='Accuracy @10', color='#2ca02c')
    ax2.bar(x + width/2, acc_50, width, label='Accuracy @50', color='#d62728')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Market Accuracy by Horizon')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Agent profitability distribution (NYC)
    ax3 = axes[1, 0]
    nyc_pnls = [m['final_pnl'] for m in nyc_report['agent_profitability'].values()]
    ax3.hist(nyc_pnls, bins=15, color='#1f77b4', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Final PnL ($)')
    ax3.set_ylabel('Number of Agents')
    ax3.set_title('NYC Mayor: Agent PnL Distribution')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # Plot 4: Agent profitability distribution (NBA)
    ax4 = axes[1, 1]
    nba_pnls = [m['final_pnl'] for m in nba_report['agent_profitability'].values()]
    ax4.hist(nba_pnls, bins=15, color='#ff7f0e', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Final PnL ($)')
    ax4.set_ylabel('Number of Agents')
    ax4.set_title('NBA Finals: Agent PnL Distribution')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/evaluation_comparison.png', dpi=300, bbox_inches='tight')
    print("\n[PLOT] Saved evaluation comparison to output/evaluation_comparison.png")


if __name__ == "__main__":
    # Set paths
    LOG_DIR = Path("simulation_logs")
    OUTPUT_DIR = Path("output")
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("PREDICTION MARKET SIMULATION - COMPREHENSIVE EVALUATION")
    print("Following AgentSociety Challenge Methodology")
    print("=" * 70)

    # Evaluate both simulations
    nyc_report = evaluate_nyc_mayor_simulation(LOG_DIR, "nyc_mayor_sim")
    nba_report = evaluate_nba_finals_simulation(LOG_DIR, "nba_finals_sim")

    # Run ablation studies
    run_ablation_studies()

    # Generate comparison plots
    if nyc_report and nba_report:
        plot_evaluation_comparison(nyc_report, nba_report)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print("\nFor your CS245 report, focus on:")
    print("1. Brier Score / Log Score (analogous to RMSE)")
    print("2. Prediction Accuracy @K (analogous to HR@K)")
    print("3. Trading Behavior Analysis (analogous to sentiment alignment)")
    print("4. Ablation study results (with/without memory, different personalities)")
    print("5. Comparison of simulated vs real market prices (if Kalshi data available)")
