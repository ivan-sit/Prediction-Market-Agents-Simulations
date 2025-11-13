"""Protocol definitions that the high-level simulator depends on.

All concrete implementations are developed by teammate-owned modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Protocol, Sequence


@dataclass(slots=True)
class MarketOrder:
    """Lightweight order representation shared across agents and markets."""

    agent_id: str
    side: str  # "buy" or "sell"
    size: float
    limit_price: float
    confidence: float
    metadata: Mapping[str, object] | None = None


class MessageStream(Protocol):
    """Produces source messages each simulation tick."""

    @property
    def finished(self) -> bool:
        ...

    def bootstrap(self, *, seed: int | None = None) -> None:
        ...

    def next_batch(self) -> List[Mapping[str, object]]:
        ...


class PortalNetwork(Protocol):
    """Routes stream messages to agent inboxes."""

    def route(self, messages: Iterable[Mapping[str, object]]) -> Mapping[str, List[dict]]:
        ...

    def ingest_agent_feedback(self, agent_id: str, payload: Mapping[str, object]) -> None:
        ...


class Agent(Protocol):
    """Minimal surface expected from AgentSocietyChallenge-compatible agents."""

    agent_id: str

    def ingest(self, messages: Sequence[Mapping[str, object]]) -> None:
        ...

    def update_belief(self, timestep: int, market_price: float) -> float:
        ...

    def generate_order(self, belief: float, market_price: float) -> MarketOrder | None:
        ...


class MarketAdapter(Protocol):
    """Wraps a real order-book implementation (external dependency)."""

    def submit_orders(self, orders: Sequence[MarketOrder], timestep: int) -> None:
        ...

    def current_price(self) -> float:
        ...

    def snapshot(self) -> Mapping[str, object]:
        ...


class Evaluator(Protocol):
    """Consumes state transitions to compute metrics (Brier, ECE, etc.)."""

    def on_tick(
        self,
        *,
        timestep: int,
        price: float,
        agent_beliefs: Mapping[str, float],
        market_snapshot: Mapping[str, object],
    ) -> None:
        ...

    def finalize(self) -> Mapping[str, float]:
        ...
