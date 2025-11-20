"""Lightweight simulation smoke test avoiding LLM dependencies.

Run with:
    PYTHONPATH=src python examples/smoke_test.py

It wires stub agents/streams into the SimulationEngine + LMSR adapter to
ensure the orchestrator still runs end-to-end after merges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence

from prediction_market_sim import SimulationEngine, SimulationRuntimeConfig
from prediction_market_sim.simulation import MarketOrder
from prediction_market_sim.market import LMSRMarketAdapter


class FixedStream:
    """Emits a deterministic signal each timestep for N steps."""

    def __init__(self, *, max_timesteps: int = 20):
        self.max_timesteps = max_timesteps
        self._t = 0

    @property
    def finished(self) -> bool:
        return self._t >= self.max_timesteps

    def bootstrap(self, *, seed: int | None = None) -> None:
        self._t = 0

    def next_batch(self) -> List[Mapping[str, object]]:
        if self.finished:
            return []
        payload = {
            "timestamp": self._t,
            "source_id": "smoke_stream",
            "signal": 1 if self._t % 2 == 0 else -1,
        }
        self._t += 1
        return [payload]


class BroadcastPortal:
    def __init__(self, *, agent_ids: Sequence[str]):
        self.agent_ids = list(agent_ids)

    def route(self, messages: Iterable[Mapping[str, object]]) -> Mapping[str, List[dict]]:
        routed = {agent: [] for agent in self.agent_ids}
        for message in messages:
            for agent in self.agent_ids:
                routed[agent].append(dict(message))
        return routed

    def ingest_agent_feedback(self, agent_id: str, payload: Mapping[str, object]) -> None:
        return


@dataclass
class StubAgent:
    agent_id: str
    learning_rate: float = 0.05
    belief: float = 0.5

    def ingest(self, messages: Sequence[Mapping[str, object]]) -> None:
        for message in messages:
            sentiment = message.get("signal", 0)
            self.belief = min(max(self.belief + self.learning_rate * sentiment, 0.01), 0.99)

    def update_belief(self, timestep: int, market_price: float) -> float:
        return self.belief

    def generate_order(self, belief: float, market_price: float) -> MarketOrder | None:
        delta = belief - market_price
        if abs(delta) < 0.02:
            return None
        side = "buy" if delta > 0 else "sell"
        return MarketOrder(
            agent_id=self.agent_id,
            side=side,
            size=min(1.0, abs(delta) * 10),
            limit_price=belief,
            confidence=min(1.0, abs(delta) * 5),
        )


def build_engine() -> SimulationEngine:
    agent_ids = ["stub_alpha", "stub_beta"]
    return SimulationEngine(
        stream_factory=lambda: FixedStream(max_timesteps=10),
        portal_factory=lambda: BroadcastPortal(agent_ids=agent_ids),
        agent_factories=[lambda i=i: StubAgent(agent_id=i) for i in agent_ids],
        market_factory=lambda: LMSRMarketAdapter(liquidity_param=80.0),
        evaluator_factories=[],
        runtime_config=SimulationRuntimeConfig(max_timesteps=10, enable_logging=False),
    )


def main() -> None:
    engine = build_engine()
    result = engine.run_once(run_id=1, seed=123)
    print(f"✓ Ran {len(result.prices)} timesteps. Final price: {result.prices[-1]:.4f}")


if __name__ == "__main__":
    main()
