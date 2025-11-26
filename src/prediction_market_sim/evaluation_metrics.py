"""
Evaluation metrics for prediction market agent simulation.
Adapted from AgentSociety Challenge framework for prediction market context.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import pandas as pd


class PredictionMarketEvaluator:
    """
    Evaluation tool for prediction market simulations.
    Analogous to AgentSociety's EvaluationTool but adapted for prediction markets.
    """

    def __init__(self, simulation_logs: pd.DataFrame, ground_truth: Dict[str, Any]):
        """
        Args:
            simulation_logs: DataFrame with columns [timestep, agent_id, action, price, etc.]
            ground_truth: Dict with actual outcomes (e.g., {"winner": "Mamdani", "final_prob": 0.504})
        """
        self.logs = simulation_logs
        self.ground_truth = ground_truth

    def calculate_brier_score(self, predicted_probs: List[float], actual_outcome: int) -> float:
        """
        Brier Score: Analogous to RMSE for probabilistic predictions.
        Lower is better. Range: [0, 1]

        Formula: BS = (1/N) * sum((predicted_prob - actual_outcome)^2)

        Args:
            predicted_probs: List of predicted probabilities over time
            actual_outcome: 1 if event happened, 0 if not

        Returns:
            Brier score (lower = better calibration)
        """
        predicted_probs = np.array(predicted_probs)
        return np.mean((predicted_probs - actual_outcome) ** 2)

    def calculate_log_score(self, predicted_probs: List[float], actual_outcome: int) -> float:
        """
        Logarithmic Score: Measures calibration of probabilistic predictions.
        Higher is better.

        Formula: LS = (1/N) * sum(actual * log(pred) + (1-actual) * log(1-pred))
        """
        predicted_probs = np.array(predicted_probs)
        # Clip to avoid log(0)
        predicted_probs = np.clip(predicted_probs, 1e-10, 1 - 1e-10)

        if actual_outcome == 1:
            return np.mean(np.log(predicted_probs))
        else:
            return np.mean(np.log(1 - predicted_probs))

    def calculate_prediction_accuracy_at_k(self, market_prices: pd.DataFrame,
                                           winner: str, k_timesteps: int = 10) -> float:
        """
        Prediction Accuracy @K: Analogous to HR@K.
        "Did the market correctly identify the winner in the last K timesteps?"

        Args:
            market_prices: DataFrame with [timestep, outcome, probability]
            winner: The actual winner
            k_timesteps: Number of final timesteps to check

        Returns:
            1.0 if winner had highest probability in last K timesteps, else 0.0
        """
        final_k_prices = market_prices.tail(k_timesteps)

        # Check if winner had highest average probability in final K timesteps
        avg_probs = final_k_prices.groupby('outcome')['probability'].mean()
        predicted_winner = avg_probs.idxmax()

        return 1.0 if predicted_winner == winner else 0.0

    def calculate_agent_profitability_metrics(self, agent_pnls: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Agent profitability analysis.

        Returns:
            Dictionary with metrics per agent type/personality
        """
        results = {}

        for agent_id, pnl_history in agent_pnls.items():
            final_pnl = pnl_history[-1]
            max_drawdown = self._calculate_max_drawdown(pnl_history)
            sharpe_ratio = self._calculate_sharpe_ratio(pnl_history)

            results[agent_id] = {
                'final_pnl': final_pnl,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'profitable': final_pnl > 0
            }

        return results

    def _calculate_max_drawdown(self, pnl_history: List[float]) -> float:
        """Calculate maximum drawdown from peak"""
        pnl_array = np.array(pnl_history)
        cummax = np.maximum.accumulate(pnl_array)
        drawdown = cummax - pnl_array
        return np.max(drawdown)

    def _calculate_sharpe_ratio(self, pnl_history: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio (return/risk)"""
        returns = np.diff(pnl_history)
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        return (np.mean(returns) - risk_free_rate) / np.std(returns)

    def calculate_market_efficiency_metrics(self, prices: pd.DataFrame, events: pd.DataFrame) -> Dict[str, float]:
        """
        Market efficiency: How quickly does price converge to true probability after events?

        Returns:
            - price_discovery_speed: Average time for price to stabilize after events
            - volatility: Price volatility measure
            - liquidity: Average bid-ask spread or trade volume
        """
        # Calculate price changes after major events
        event_times = events['initial_time'].values
        price_changes = []

        for event_time in event_times:
            # Get prices 5 timesteps before and 10 timesteps after event
            before = prices[prices['timestep'] == event_time - 1]['probability'].values
            after = prices[(prices['timestep'] > event_time) &
                          (prices['timestep'] <= event_time + 10)]['probability'].values

            if len(before) > 0 and len(after) > 0:
                # Time to stabilize (when price change < 1%)
                stabilization_time = self._find_stabilization_time(after)
                price_changes.append(stabilization_time)

        volatility = prices.groupby('outcome')['probability'].std().mean()

        return {
            'avg_price_discovery_speed': np.mean(price_changes) if price_changes else np.inf,
            'price_volatility': volatility
        }

    def _find_stabilization_time(self, price_series: np.ndarray, threshold: float = 0.01) -> int:
        """Find how many timesteps until price stabilizes"""
        for i in range(1, len(price_series)):
            if abs(price_series[i] - price_series[i-1]) < threshold:
                return i
        return len(price_series)

    def calculate_trading_behavior_metrics(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze trading behavior patterns by agent personality.
        Analogous to "sentiment alignment" in AgentSociety.

        Returns:
            Metrics about how different personalities trade
        """
        behavior_metrics = {}

        for agent_type in trades['agent_type'].unique():
            agent_trades = trades[trades['agent_type'] == agent_type]

            behavior_metrics[agent_type] = {
                'total_trades': len(agent_trades),
                'avg_position_size': agent_trades['quantity'].mean(),
                'trade_frequency': len(agent_trades) / trades['timestep'].max(),
                'contrarian_ratio': self._calculate_contrarian_ratio(agent_trades),
                'momentum_ratio': self._calculate_momentum_ratio(agent_trades)
            }

        return behavior_metrics

    def _calculate_contrarian_ratio(self, trades: pd.DataFrame) -> float:
        """Ratio of trades against prevailing price trend"""
        # Simplified: buy when price falling, sell when price rising
        contrarian_count = 0
        for i in range(1, len(trades)):
            price_change = trades.iloc[i]['price'] - trades.iloc[i-1]['price']
            action = trades.iloc[i]['action']

            if (action == 'buy' and price_change < 0) or (action == 'sell' and price_change > 0):
                contrarian_count += 1

        return contrarian_count / len(trades) if len(trades) > 0 else 0.0

    def _calculate_momentum_ratio(self, trades: pd.DataFrame) -> float:
        """Ratio of trades following price trend"""
        return 1.0 - self._calculate_contrarian_ratio(trades)

    def generate_evaluation_report(self, market_prices: pd.DataFrame,
                                   agent_pnls: Dict[str, List[float]],
                                   trades: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        Similar to simulator.evaluate() in AgentSociety.

        Returns:
            Complete evaluation metrics dictionary
        """
        winner = self.ground_truth.get('winner')
        final_prob = self.ground_truth.get('final_prob')

        # Get market probabilities over time for winner
        winner_probs = market_prices[market_prices['outcome'] == winner]['probability'].values

        report = {
            # Prediction Quality Metrics (analogous to RMSE)
            'brier_score': self.calculate_brier_score(winner_probs, 1),
            'log_score': self.calculate_log_score(winner_probs, 1),

            # Accuracy Metrics (analogous to HR@K)
            'prediction_accuracy_at_10': self.calculate_prediction_accuracy_at_k(
                market_prices, winner, k_timesteps=10
            ),
            'prediction_accuracy_at_50': self.calculate_prediction_accuracy_at_k(
                market_prices, winner, k_timesteps=50
            ),

            # Agent Performance (analogous to sentiment alignment)
            'agent_profitability': self.calculate_agent_profitability_metrics(agent_pnls),
            'trading_behavior': self.calculate_trading_behavior_metrics(trades),

            # Market Efficiency
            'market_efficiency': self.calculate_market_efficiency_metrics(
                market_prices,
                pd.DataFrame(self.ground_truth.get('events', []))
            ),

            # Summary Statistics
            'total_trades': len(trades),
            'final_market_price': winner_probs[-1] if len(winner_probs) > 0 else None,
            'actual_outcome_prob': final_prob
        }

        return report

    def compare_with_baseline(self, baseline_agent_pnls: Dict[str, List[float]],
                              experimental_agent_pnls: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Ablation study: Compare experimental agents vs baseline.
        As required by project spec: "compare against baseline agents"

        Returns:
            Improvement metrics
        """
        baseline_final = np.mean([pnl[-1] for pnl in baseline_agent_pnls.values()])
        experimental_final = np.mean([pnl[-1] for pnl in experimental_agent_pnls.values()])

        improvement = (experimental_final - baseline_final) / abs(baseline_final) if baseline_final != 0 else 0

        return {
            'baseline_avg_pnl': baseline_final,
            'experimental_avg_pnl': experimental_final,
            'percent_improvement': improvement * 100
        }


def run_ablation_study(config_variations: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Run ablation study as required by project spec.

    Example variations:
    - With/without memory
    - With/without context retrieval
    - Different personality prompts
    - Different exploration strategies

    Args:
        config_variations: List of configuration dictionaries to test

    Returns:
        DataFrame comparing results across configurations
    """
    results = []

    for config in config_variations:
        # Run simulation with this configuration
        # (This would call your main simulation script)
        print(f"Running simulation with config: {config['name']}")

        # Placeholder - you'd actually run the simulation here
        result = {
            'config_name': config['name'],
            'has_memory': config.get('memory_enabled', False),
            'personality_type': config.get('personality_type', 'baseline'),
            # Add actual metrics after simulation runs
        }
        results.append(result)

    return pd.DataFrame(results)
