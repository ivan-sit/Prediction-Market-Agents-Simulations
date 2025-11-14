"""Enhanced demo showcasing all simulation features.

This example demonstrates:
- Full orderbook implementation with bid/ask matching
- LMSR market maker for prediction markets
- Comprehensive data logging (market_df, belief_df, agent_meta_df, sources)
- Multiple agent types with different strategies
- Net flow tracking, spread monitoring, and position tracking

Run with:
    PYTHONPATH=src python examples/demo_full_simulation.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

from prediction_market_sim import SimulationEngine, SimulationRuntimeConfig
from prediction_market_sim.simulation import MarketOrder
from prediction_market_sim.market import LMSRMarketAdapter


# ============================================================================
# DATA SOURCE: Simulated news/signal stream
# ============================================================================

class RealWorldNewsStream:
    """Simulates a stream of news/signals affecting market beliefs.
    
    Each message has a sentiment (-1 to +1) and strength that agents can interpret.
    """

    def __init__(self, *, max_messages: int = 50, volatility: float = 0.3):
        self.max_messages = max_messages
        self.volatility = volatility
        self._message_count = 0
        self._rng = random.Random(0)
        
        # Simulate different news sources with varying reliability
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

        # Generate 1-3 messages per timestep
        num_messages = self._rng.randint(1, 3)
        messages = []
        
        for _ in range(num_messages):
            if self._message_count >= self.max_messages:
                break
                
            source = self._rng.choice(self._sources)
            sentiment = self._rng.gauss(0, self.volatility)
            sentiment = max(-1, min(1, sentiment))  # Clip to [-1, 1]
            
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


# ============================================================================
# PORTAL NETWORK: Routes messages to agents
# ============================================================================

class InformationPortalNetwork:
    """Routes messages to agents based on subscriptions and access levels.
    
    - Premium agents get all sources
    - Basic agents only get public sources (reuters, twitter)
    """

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
        # Could use this for agent communication or meta-learning
        pass


# ============================================================================
# AGENTS: Different trading strategies
# ============================================================================

@dataclass
class BayesianAgent:
    """Agent that updates beliefs using Bayesian-style inference."""

    agent_id: str
    learning_rate: float = 0.10
    baseline: float = 0.5
    confidence_threshold: float = 0.03
    max_order_size: float = 10.0
    risk_aversion: float = 1.0

    def __post_init__(self) -> None:
        self._belief = self.baseline
        self._confidence = 0.5

    def ingest(self, messages: Sequence[Mapping[str, object]]) -> None:
        """Update belief based on weighted signals."""
        for message in messages:
            sentiment = message.get("sentiment", 0.0)
            reliability = message.get("reliability", 0.5)
            
            # Weight signal by reliability
            weighted_signal = sentiment * reliability * self.learning_rate
            
            # Update belief
            self._belief = min(max(self._belief + weighted_signal, 0.01), 0.99)
            
            # Update confidence based on signal strength
            self._confidence = min(1.0, self._confidence + abs(sentiment) * 0.1)

    def update_belief(self, timestep: int, market_price: float) -> float:
        # Mean revert slightly towards market price (learning from market)
        market_weight = 0.05
        self._belief = (1 - market_weight) * self._belief + market_weight * market_price
        return self._belief

    def generate_order(self, belief: float, market_price: float) -> MarketOrder | None:
        delta = belief - market_price
        
        # Only trade if delta exceeds confidence threshold
        if abs(delta) < self.confidence_threshold:
            return None

        side = "buy" if delta > 0 else "sell"
        
        # Size based on delta and confidence, adjusted for risk aversion
        size = min(
            self.max_order_size,
            abs(delta) * self._confidence * self.max_order_size / self.risk_aversion
        )
        
        # Limit price: willing to pay/sell at our belief
        limit_price = belief
        
        return MarketOrder(
            agent_id=self.agent_id,
            side=side,
            size=size,
            limit_price=limit_price,
            confidence=self._confidence,
            metadata={"strategy": "bayesian", "delta": delta}
        )


@dataclass
class MomentumAgent:
    """Agent that follows price momentum."""

    agent_id: str
    lookback: int = 5
    momentum_threshold: float = 0.02
    max_order_size: float = 8.0

    def __post_init__(self) -> None:
        self._price_history: List[float] = []
        self._belief = 0.5

    def ingest(self, messages: Sequence[Mapping[str, object]]) -> None:
        # Momentum agent doesn't use messages, only price action
        pass

    def update_belief(self, timestep: int, market_price: float) -> float:
        self._price_history.append(market_price)
        
        if len(self._price_history) > self.lookback:
            self._price_history.pop(0)
        
        if len(self._price_history) >= 2:
            # Calculate momentum
            momentum = self._price_history[-1] - self._price_history[0]
            
            # Extrapolate
            self._belief = min(max(market_price + momentum, 0.01), 0.99)
        else:
            self._belief = market_price
            
        return self._belief

    def generate_order(self, belief: float, market_price: float) -> MarketOrder | None:
        if len(self._price_history) < 2:
            return None
            
        delta = belief - market_price
        
        if abs(delta) < self.momentum_threshold:
            return None

        side = "buy" if delta > 0 else "sell"
        size = min(self.max_order_size, abs(delta) * 20)
        
        return MarketOrder(
            agent_id=self.agent_id,
            side=side,
            size=size,
            limit_price=belief,
            confidence=min(1.0, abs(delta) * 5),
            metadata={"strategy": "momentum"}
        )


@dataclass
class NoiseTrader:
    """Agent that trades randomly (noise trader)."""

    agent_id: str
    trade_probability: float = 0.3
    max_order_size: float = 5.0

    def __post_init__(self) -> None:
        self._rng = random.Random()
        self._belief = 0.5

    def ingest(self, messages: Sequence[Mapping[str, object]]) -> None:
        # Noise trader incorporates some random signals
        if messages and self._rng.random() < 0.5:
            sentiment = messages[0].get("sentiment", 0)
            self._belief = min(max(self._belief + sentiment * 0.05, 0.01), 0.99)

    def update_belief(self, timestep: int, market_price: float) -> float:
        # Add random walk
        self._belief += self._rng.gauss(0, 0.05)
        self._belief = min(max(self._belief, 0.01), 0.99)
        return self._belief

    def generate_order(self, belief: float, market_price: float) -> MarketOrder | None:
        if self._rng.random() > self.trade_probability:
            return None

        side = "buy" if self._rng.random() < 0.5 else "sell"
        size = self._rng.uniform(1, self.max_order_size)
        limit_price = market_price + self._rng.gauss(0, 0.05)
        limit_price = min(max(limit_price, 0.01), 0.99)
        
        return MarketOrder(
            agent_id=self.agent_id,
            side=side,
            size=size,
            limit_price=limit_price,
            confidence=0.3,
            metadata={"strategy": "noise"}
        )


# ============================================================================
# EVALUATOR: Track prediction quality
# ============================================================================

class PredictionQualityEvaluator:
    """Evaluates how well agents' beliefs converge to truth."""

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
        # Brier score for market price
        brier = (price - self.true_probability) ** 2
        self._brier_scores.append(brier)
        
        # Price error
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


# ============================================================================
# BUILD ENGINE
# ============================================================================

def build_lmsr_engine_simple() -> SimulationEngine:
    """Build simulation with LMSR (simpler agent set for testing)."""
    
    agent_ids = ["bayesian_1", "bayesian_2", "momentum_1", "noise_1"]
    
    subscriptions = {
        "bayesian_1": ["all"],
        "bayesian_2": ["reuters", "twitter", "analyst_report"],
        "momentum_1": ["reuters", "twitter"],
        "noise_1": ["twitter"],
    }
    
    return SimulationEngine(
        stream_factory=lambda: RealWorldNewsStream(max_messages=100, volatility=0.2),
        portal_factory=lambda: InformationPortalNetwork(agent_subscriptions=subscriptions),
        agent_factories=[
            lambda: BayesianAgent(agent_id="bayesian_1", learning_rate=0.12, max_order_size=15.0),
            lambda: BayesianAgent(agent_id="bayesian_2", learning_rate=0.10, max_order_size=12.0),
            lambda: MomentumAgent(agent_id="momentum_1", lookback=5, max_order_size=10.0),
            lambda: NoiseTrader(agent_id="noise_1", trade_probability=0.4, max_order_size=6.0),
        ],
        market_factory=lambda: LMSRMarketAdapter(liquidity_param=50.0),
        evaluator_factories=[lambda: PredictionQualityEvaluator(true_probability=0.68)],
        runtime_config=SimulationRuntimeConfig(
            max_timesteps=100,
            run_name="lmsr_simple_demo",
            log_dir=Path("simulation_logs"),
            enable_logging=True,
            save_logs_as_csv=True,
            save_logs_as_json=True,
        ),
    )


def build_lmsr_engine() -> SimulationEngine:
    """Build simulation with LMSR market maker."""
    
    agent_ids = ["bayesian_1", "bayesian_2", "momentum_1", "noise_1"]
    
    subscriptions = {
        "bayesian_1": ["all"],
        "bayesian_2": ["reuters", "twitter", "analyst_report"],
        "momentum_1": ["reuters", "twitter"],
        "noise_1": ["twitter"],
    }
    
    return SimulationEngine(
        stream_factory=lambda: RealWorldNewsStream(max_messages=100, volatility=0.2),
        portal_factory=lambda: InformationPortalNetwork(agent_subscriptions=subscriptions),
        agent_factories=[
            lambda: BayesianAgent(agent_id="bayesian_1", learning_rate=0.12, max_order_size=15.0),
            lambda: BayesianAgent(agent_id="bayesian_2", learning_rate=0.10, max_order_size=12.0),
            lambda: MomentumAgent(agent_id="momentum_1", lookback=5, max_order_size=10.0),
            lambda: NoiseTrader(agent_id="noise_1", trade_probability=0.4, max_order_size=6.0),
        ],
        market_factory=lambda: LMSRMarketAdapter(liquidity_param=50.0),
        evaluator_factories=[lambda: PredictionQualityEvaluator(true_probability=0.68)],
        runtime_config=SimulationRuntimeConfig(
            max_timesteps=100,
            run_name="lmsr_demo",
            log_dir=Path("simulation_logs"),
            enable_logging=True,
            save_logs_as_csv=True,
            save_logs_as_json=True,
        ),
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Run LMSR prediction market simulation."""
    
    print("=" * 80)
    print("RUNNING LMSR PREDICTION MARKET SIMULATION")
    print("=" * 80)
    
    engine = build_lmsr_engine()
    result = engine.run_once(run_id=1, seed=42)
    
    print(f"\n✓ Ran {len(result.prices)} timesteps")
    print(f"✓ Initial price: {result.prices[0]:.4f}")
    print(f"✓ Final price: {result.prices[-1]:.4f}")
    print(f"✓ Price change: {result.prices[-1] - result.prices[0]:+.4f}")
    
    if result.summary_stats:
        print(f"\n📊 Summary Statistics:")
        for key, value in result.summary_stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
    
    if result.log_files:
        print(f"\n💾 Saved log files:")
        for log_type, path in result.log_files.items():
            print(f"   {log_type}: {path}")
    
    for name, metrics in result.evaluator_metrics.items():
        print(f"\n📈 {name}:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.4f}")
    
    # Show final market state
    if result.market_snapshots:
        final_snapshot = result.market_snapshots[-1]
        print(f"\n🏦 Final Market State:")
        print(f"   YES Price: {final_snapshot.get('yes_price', 'N/A'):.4f}")
        print(f"   NO Price: {final_snapshot.get('no_price', 'N/A'):.4f}")
        print(f"   Total Volume: {final_snapshot.get('total_volume', 0):.2f} shares")
        print(f"   Number of Trades: {final_snapshot.get('num_trades', 0)}")
    
    print("\n" + "=" * 80)
    print("✅ SIMULATION COMPLETE")
    print("=" * 80)
    print("\nCheck the 'simulation_logs/' directory for detailed CSV and JSON files.")
    print("You can analyze:")
    print("  - market_df: Market prices, volume, net flow over time")
    print("  - beliefs_df: Agent beliefs and convergence metrics")
    print("  - sources_df: Information signals fed to agents")
    print("\nVisualize with: python examples/visualize_results.py")


if __name__ == "__main__":
    main()

