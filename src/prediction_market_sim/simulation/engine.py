"""High-level simulation orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from .interfaces import (
    Agent,
    Evaluator,
    MarketAdapter,
    MarketOrder,
    MessageStream,
    PortalNetwork,
)
from .logging import SimulationLogger


Factory = Callable[[], object]


@dataclass(slots=True)
class SimulationRuntimeConfig:
    """Controls runtime behavior for the simulator."""

    max_timesteps: int
    log_dir: Path = Path("artifacts")
    run_name: str = "default"
    log_every: int = 1
    stop_when_stream_finishes: bool = True
    enable_logging: bool = True
    save_logs_as_csv: bool = True
    save_logs_as_json: bool = True


@dataclass(slots=True)
class SimulationResult:
    """Structured output for downstream evaluation/reporting."""

    run_id: int
    prices: List[float] = field(default_factory=list)
    belief_history: List[Mapping[str, float]] = field(default_factory=list)
    market_snapshots: List[Mapping[str, object]] = field(default_factory=list)
    evaluator_metrics: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    logger: Optional[SimulationLogger] = None
    log_files: Mapping[str, Path] = field(default_factory=dict)
    summary_stats: Mapping[str, object] = field(default_factory=dict)


class SimulationEngine:
    """Coordinates agents, portals, markets, and evaluators."""

    def __init__(
        self,
        *,
        stream_factory: Callable[[], MessageStream],
        portal_factory: Callable[[], PortalNetwork],
        agent_factories: Sequence[Callable[[], Agent]],
        market_factory: Callable[[], MarketAdapter],
        evaluator_factories: Sequence[Callable[[], Evaluator]],
        runtime_config: SimulationRuntimeConfig,
    ) -> None:
        self._stream_factory = stream_factory
        self._portal_factory = portal_factory
        self._agent_factories = list(agent_factories)
        self._market_factory = market_factory
        self._evaluator_factories = list(evaluator_factories)
        self._config = runtime_config

    def run_many(self, *, num_runs: int, seeds: Iterable[int] | None = None) -> List[SimulationResult]:
        """Convenience helper for multi-run experiments."""

        seeds = list(seeds) if seeds is not None else [None] * num_runs
        if len(seeds) != num_runs:
            raise ValueError("Length of seeds iterable must match num_runs.")

        results: List[SimulationResult] = []
        for run_id, seed in enumerate(seeds, start=1):
            results.append(self.run_once(run_id=run_id, seed=seed))
        return results

    def run_once(self, *, run_id: int = 1, seed: int | None = None) -> SimulationResult:
        """Execute a single simulation loop."""

        stream = self._stream_factory()
        portal = self._portal_factory()
        agents = [factory() for factory in self._agent_factories]
        market = self._market_factory()
        evaluators = [factory() for factory in self._evaluator_factories]

        stream.bootstrap(seed=seed)
        result = SimulationResult(run_id=run_id)

        # Initialize logger if enabled
        logger: Optional[SimulationLogger] = None
        if self._config.enable_logging:
            run_name = f"{self._config.run_name}_run{run_id}"
            logger = SimulationLogger(log_dir=self._config.log_dir, run_id=run_name)
            result.logger = logger

        current_price = market.current_price()
        timestep = 0
        while timestep < self._config.max_timesteps:
            if stream.finished and self._config.stop_when_stream_finishes:
                break

            messages = stream.next_batch()
            routed = portal.route(messages) if messages else {}
            belief_snapshot: Dict[str, float] = {}
            orders: List[MarketOrder] = []

            # Log source messages
            if logger and messages:
                for message in messages:
                    logger.log_source_message(timestep, message)

            for agent in agents:
                inbox = routed.get(agent.agent_id, [])
                if inbox:
                    agent.ingest(inbox)
                belief = agent.update_belief(timestep, current_price)
                belief_snapshot[agent.agent_id] = belief

                maybe_order = agent.generate_order(belief, current_price)
                if maybe_order is not None:
                    orders.append(maybe_order)

            if orders:
                market.submit_orders(orders, timestep)

            current_price = market.current_price()
            snapshot = market.snapshot()
            result.prices.append(current_price)
            result.belief_history.append(belief_snapshot)
            result.market_snapshots.append(snapshot)

            # Log market state and beliefs
            if logger and timestep % self._config.log_every == 0:
                logger.log_market_state(timestep, current_price, snapshot)
                logger.log_beliefs(timestep, belief_snapshot, current_price)

            for evaluator in evaluators:
                evaluator.on_tick(
                    timestep=timestep,
                    price=current_price,
                    agent_beliefs=belief_snapshot,
                    market_snapshot=snapshot,
                )

            timestep += 1

        metrics: MutableMapping[str, Mapping[str, float]] = {}
        for evaluator in evaluators:
            name = evaluator.__class__.__name__
            metrics[name] = evaluator.finalize()
        result.evaluator_metrics = metrics

        # Save logs and generate summary
        if logger:
            if self._config.save_logs_as_csv:
                result.log_files.update(logger.save_to_csv())
            if self._config.save_logs_as_json:
                result.log_files.update(logger.save_to_json())
            result.summary_stats = logger.get_summary_stats()

        return result
