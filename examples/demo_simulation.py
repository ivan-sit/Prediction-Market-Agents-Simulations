"""Minimal demo showcasing the SimulationEngine orchestrator.

Run with:

    PYTHONPATH=src python examples/demo_simulation.py

All components below are lightweight stubs so teammates can focus on their own
modules while still exercising the high-level loop.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence

from prediction_market_sim import SimulationEngine, SimulationRuntimeConfig
from prediction_market_sim.simulation import MarketOrder


class DemoMessageStream:
    """Deterministic stream that emits one message per timestep."""

    def __init__(self, *, max_timesteps: int = 12):
        self.max_timesteps = max_timesteps
        self._t = 0
        self._rng = random.Random(0)

    @property
    def finished(self) -> bool:
        return self._t >= self.max_timesteps

    def bootstrap(self, *, seed: int | None = None) -> None:
        self._t = 0
        if seed is not None:
            self._rng.seed(seed)

    def next_batch(self) -> List[Mapping[str, object]]:
        if self.finished:
            return []

        topic = "election"
        sentiment = self._rng.choice([-1, 1])
        payload = {
            "timestamp": self._t,
            "source_id": "portal_main",
            "topic": topic,
            "content": f"Signal {sentiment:+d} @t={self._t}",
            "signal": sentiment,
        }

        self._t += 1
        return [payload]


class DemoPortalNetwork:
    """Broadcasts messages from each source to all registered agents."""

    def __init__(self, *, agent_ids: Sequence[str]):
        self.agent_ids = list(agent_ids)

    def route(self, messages: Iterable[Mapping[str, object]]) -> Mapping[str, List[dict]]:
        routed = {agent_id: [] for agent_id in self.agent_ids}
        for message in messages:
            for agent_id in self.agent_ids:
                routed[agent_id].append(dict(message))
        return routed

    def ingest_agent_feedback(self, agent_id: str, payload: Mapping[str, object]) -> None:
        # Demo portal ignores feedback.
        return


@dataclass
class DemoAgent:
    """Toy agent that adjusts belief based on incoming signal direction."""

    agent_id: str
    learning_rate: float = 0.05
    baseline: float = 0.5

    def __post_init__(self) -> None:
        self._belief = self.baseline

    def ingest(self, messages: Sequence[Mapping[str, object]]) -> None:
        for message in messages:
            sentiment = message.get("signal", 0)
            self._belief = min(max(self._belief + self.learning_rate * sentiment, 0.01), 0.99)

    def update_belief(self, timestep: int, market_price: float) -> float:
        return self._belief

    def generate_order(self, belief: float, market_price: float) -> MarketOrder | None:
        delta = belief - market_price
        if abs(delta) < 0.02:
            return None

        side = "buy" if delta > 0 else "sell"
        size = min(1.0, abs(delta) * 10)
        limit_price = belief
        return MarketOrder(
            agent_id=self.agent_id,
            side=side,
            size=size,
            limit_price=limit_price,
            confidence=min(1.0, abs(delta) * 5),
            metadata={"kind": "demo"},
        )


class DemoMarketAdapter:
    """Simple impact model: price moves with signed order flow."""

    def __init__(self, *, initial_price: float = 0.5, impact: float = 0.03):
        self._price = initial_price
        self._impact = impact
        self._last_net_flow = 0.0

    def submit_orders(self, orders: Sequence[MarketOrder], timestep: int) -> None:
        net_flow = 0.0
        for order in orders:
            net_flow += order.size if order.side == "buy" else -order.size
        self._last_net_flow = net_flow
        self._price = min(max(self._price + self._impact * net_flow, 0.01), 0.99)

    def current_price(self) -> float:
        return float(self._price)

    def snapshot(self) -> Mapping[str, object]:
        return {"price": self._price, "net_flow": self._last_net_flow}


class DemoEvaluator:
    """Tracks average belief-price gap across the run."""

    def __init__(self) -> None:
        self._gaps: List[float] = []
        self._prices: List[float] = []

    def on_tick(
        self,
        *,
        timestep: int,
        price: float,
        agent_beliefs: Mapping[str, float],
        market_snapshot: Mapping[str, object],
    ) -> None:
        mean_belief = sum(agent_beliefs.values()) / max(len(agent_beliefs), 1)
        self._gaps.append(abs(mean_belief - price))
        self._prices.append(price)

    def finalize(self) -> Mapping[str, float]:
        if not self._gaps:
            return {}
        avg_gap = sum(self._gaps) / len(self._gaps)
        avg_price = sum(self._prices) / len(self._prices)
        return {"avg_belief_price_gap": avg_gap, "avg_price": avg_price}


def build_engine() -> SimulationEngine:
    agent_ids = ["agent_kelly", "agent_llm"]

    return SimulationEngine(
        stream_factory=lambda: DemoMessageStream(max_timesteps=20),
        portal_factory=lambda: DemoPortalNetwork(agent_ids=agent_ids),
        agent_factories=[
            lambda: DemoAgent(agent_id="agent_kelly", learning_rate=0.06),
            lambda: DemoAgent(agent_id="agent_llm", learning_rate=0.08),
        ],
        market_factory=lambda: DemoMarketAdapter(initial_price=0.5),
        evaluator_factories=[lambda: DemoEvaluator()],
        runtime_config=SimulationRuntimeConfig(max_timesteps=20),
    )


def main() -> None:
    engine = build_engine()
    result = engine.run_once(run_id=1, seed=2024)
    print(f"Ran {len(result.prices)} timesteps.")
    print(f"Final price: {result.prices[-1]:.3f}")
    for name, metrics in result.evaluator_metrics.items():
        print(f"{name}: {metrics}")


if __name__ == "__main__":
    main()
