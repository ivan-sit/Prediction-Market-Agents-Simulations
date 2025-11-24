from __future__ import annotations

import sys
import random
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prediction_market_sim import SimulationEngine, SimulationRuntimeConfig
from prediction_market_sim.simulation import MarketOrder
from prediction_market_sim.market import LMSRMarketAdapter
from prediction_market_sim.agents import create_prediction_agent


def _generate_run_name(
    agents: int,
    timesteps: int,
    liquidity: float,
    messages: int,
    volatility: float,
    cash: float,
) -> str:
    """Generate procedural run name with parameters and timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return (
        f"lmsr_a{agents}_t{timesteps}_l{liquidity}_m{messages}_"
        f"v{volatility:.1f}_c{cash:.0f}_{timestamp}"
    )


class RealWorldNewsStream:

    def __init__(self, *, max_messages: int = 50, volatility: float = 0.3):
        self.max_messages = max_messages
        self.volatility = volatility
        self._message_count = 0
        self._rng = random.Random(0)

        self._sources = ["reuters", "twitter", "analyst_report", "insider"]
        self._reliability = {
            "reuters": 0.85,
            "twitter": 0.60,
            "analyst_report": 0.75,
            "insider": 0.90
        }

    @property
    def finished(self) -> bool:
        return self._message_count >= self.max_messages

    def bootstrap(self, *, seed: int | None = None) -> None:
        self._message_count = 0
        if seed is not None:
            self._rng.seed(seed)

    def next_batch(self) -> List[Mapping[str, object]]:
        if self.finished:
            return []

        num_messages = self._rng.randint(1, 3)
        messages = []

        for _ in range(num_messages):
            if self._message_count >= self.max_messages:
                break

            source = self._rng.choice(self._sources)
            sentiment = self._rng.gauss(0, self.volatility)
            sentiment = max(-1, min(1, sentiment))

            message = {
                "timestamp": self._message_count,
                "source_id": source,
                "content": f"Signal from {source}: {sentiment:+.2f}",
                "sentiment": sentiment,
                "reliability": self._reliability[source],
                "topic": "election_outcome"
            }

            messages.append(message)
            self._message_count += 1

        return messages


class InformationPortalNetwork:

    def __init__(self, *, agent_subscriptions: Mapping[str, List[str]]):
        self.subscriptions = dict(agent_subscriptions)

    def route(self, messages: Iterable[Mapping[str, object]]) -> Mapping[str, List[dict]]:
        routed = {agent_id: [] for agent_id in self.subscriptions}

        for message in messages:
            source = message.get("source_id", "unknown")

            for agent_id, subscribed_sources in self.subscriptions.items():
                if source in subscribed_sources or "all" in subscribed_sources:
                    routed[agent_id].append(dict(message))

        return routed

    def ingest_agent_feedback(self, agent_id: str, payload: Mapping[str, object]) -> None:
        pass


class PredictionQualityEvaluator:

    def __init__(self, *, true_probability: float = 0.65):
        self.true_probability = true_probability
        self._brier_scores: List[float] = []
        self._price_errors: List[float] = []

    def on_tick(
        self,
        *,
        timestep: int,
        price: float,
        agent_beliefs: Mapping[str, float],
        market_snapshot: Mapping[str, object],
    ) -> None:
        brier = (price - self.true_probability) ** 2
        self._brier_scores.append(brier)

        price_error = abs(price - self.true_probability)
        self._price_errors.append(price_error)

    def finalize(self) -> Mapping[str, float]:
        if not self._brier_scores:
            return {}

        return {
            "mean_brier_score": sum(self._brier_scores) / len(self._brier_scores),
            "final_brier_score": self._brier_scores[-1],
            "mean_price_error": sum(self._price_errors) / len(self._price_errors),
            "final_price_error": self._price_errors[-1],
        }


def build_lmsr_engine_simple() -> SimulationEngine:

    agent_ids = ["conservative_1", "aggressive_1"]

    subscriptions = {
        "conservative_1": ["all"],
        "aggressive_1": ["reuters", "twitter"],
    }

    return SimulationEngine(
        stream_factory=lambda: RealWorldNewsStream(max_messages=50, volatility=0.2),
        portal_factory=lambda: InformationPortalNetwork(agent_subscriptions=subscriptions),
        agent_factories=[
            lambda: create_prediction_agent(
                agent_id="conservative_1",
                personality="Conservative trader who weighs evidence carefully",
                initial_cash=10000.0
            ),
            lambda: create_prediction_agent(
                agent_id="aggressive_1",
                personality="Aggressive trader who takes bold positions",
                initial_cash=10000.0
            ),
        ],
        market_factory=lambda: LMSRMarketAdapter(liquidity_param=50.0),
        evaluator_factories=[lambda: PredictionQualityEvaluator(true_probability=0.68)],
        runtime_config=SimulationRuntimeConfig(
            max_timesteps=50,
            run_name="lmsr_simple_demo",
            log_dir=Path("simulation_logs"),
            enable_logging=True,
            save_logs_as_csv=True,
            save_logs_as_json=True,
        ),
    )


def build_lmsr_engine(
    num_agents: int = 2,
    timesteps: int = 5,
    liquidity: float = 50.0,
    max_messages: int = 100,
    volatility: float = 0.2,
    initial_cash: float = 10000.0,
    run_name: str = "lmsr_demo",
) -> SimulationEngine:

    personalities = [
        "Conservative Bayesian trader who carefully weighs evidence before making decisions",
        "Aggressive trader who takes bold positions based on strong signals",
        "Momentum trader who follows price trends and market sentiment",
        "Contrarian trader who bets against the crowd and looks for mispricing",
        "Risk-averse trader who prioritizes capital preservation",
        "Speculative trader who seeks high-risk high-reward opportunities",
        "Technical analyst who relies on price patterns and indicators",
        "Fundamental analyst who focuses on underlying event data",
    ]

    agent_factories = []
    subscriptions = {}

    for i in range(num_agents):
        agent_id = f"agent_{i+1}"
        personality = personalities[i % len(personalities)]

        agent_factories.append(
            lambda aid=agent_id, pers=personality: create_prediction_agent(
                agent_id=aid,
                personality=pers,
                initial_cash=initial_cash
            )
        )

        if i == 0:
            subscriptions[agent_id] = ["all"]
        elif i % 3 == 0:
            subscriptions[agent_id] = ["reuters", "twitter", "analyst_report"]
        elif i % 3 == 1:
            subscriptions[agent_id] = ["reuters", "twitter"]
        else:
            subscriptions[agent_id] = ["twitter"]

    return SimulationEngine(
        stream_factory=lambda: RealWorldNewsStream(max_messages=max_messages, volatility=volatility),
        portal_factory=lambda: InformationPortalNetwork(agent_subscriptions=subscriptions),
        agent_factories=agent_factories,
        market_factory=lambda: LMSRMarketAdapter(liquidity_param=liquidity),
        evaluator_factories=[lambda: PredictionQualityEvaluator(true_probability=0.68)],
        runtime_config=SimulationRuntimeConfig(
            max_timesteps=timesteps,
            run_name=run_name,
            log_dir=Path("simulation_logs"),
            enable_logging=True,
            save_logs_as_csv=True,
            save_logs_as_json=True,
        ),
    )


def main() -> None:

    parser = argparse.ArgumentParser(description="Run LMSR Prediction Market Simulation with LLM Agents")
    parser.add_argument("--agents", type=int, default=10, help="Number of agents (default: 4)")
    parser.add_argument("--timesteps", type=int, default=50, help="Number of timesteps (default: 100)")
    parser.add_argument("--liquidity", type=float, default=50.0, help="LMSR liquidity parameter (default: 50.0)")
    parser.add_argument("--messages", type=int, default=100, help="Max messages from data stream (default: 100)")
    parser.add_argument("--volatility", type=float, default=0.2, help="Market volatility (default: 0.2)")
    parser.add_argument("--cash", type=float, default=10000.0, help="Initial cash per agent (default: 10000.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    print("\nRUNNING LMSR PREDICTION MARKET SIMULATION")
    print(f"Agents: {args.agents}, Timesteps: {args.timesteps}, Liquidity: {args.liquidity}\n")

    run_name = _generate_run_name(
        agents=args.agents,
        timesteps=args.timesteps,
        liquidity=args.liquidity,
        messages=args.messages,
        volatility=args.volatility,
        cash=args.cash,
    )

    engine = build_lmsr_engine(
        num_agents=args.agents,
        timesteps=args.timesteps,
        liquidity=args.liquidity,
        max_messages=args.messages,
        volatility=args.volatility,
        initial_cash=args.cash,
        run_name=run_name,
    )

    result = engine.run_once(run_id=1, seed=args.seed)

    print(f"\nRan {len(result.prices)} timesteps")
    print(f"Initial price: {result.prices[0]:.4f}")
    print(f"Final price: {result.prices[-1]:.4f}")
    print(f"Price change: {result.prices[-1] - result.prices[0]:+.4f}")

    if result.summary_stats:
        print(f"\nSummary Statistics:")
        for key, value in result.summary_stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")

    if result.log_files:
        print(f"\nSaved log files:")
        for log_type, path in result.log_files.items():
            print(f"   {log_type}: {path}")

    for name, metrics in result.evaluator_metrics.items():
        print(f"\n{name}:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.4f}")

    if result.market_snapshots:
        final_snapshot = result.market_snapshots[-1]
        print(f"\nFinal Market State:")
        print(f"   YES Price: {final_snapshot.get('yes_price', 'N/A'):.4f}")
        print(f"   NO Price: {final_snapshot.get('no_price', 'N/A'):.4f}")
        print(f"   Total Volume: {final_snapshot.get('total_volume', 0):.2f} shares")
        print(f"   Number of Trades: {final_snapshot.get('num_trades', 0)}")

    print("\nSIMULATION COMPLETE")
    print("\nCheck the 'simulation_logs/' directory for detailed CSV and JSON files.")
    print("You can analyze:")
    print("  - market_df: Market prices, volume, net flow over time")
    print("  - beliefs_df: Agent beliefs and convergence metrics")
    print("  - sources_df: Information signals fed to agents")
    print("\nVisualize with: python examples/visualize_results.py")


if __name__ == "__main__":
    main()
