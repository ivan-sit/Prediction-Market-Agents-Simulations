"""High-level simulation orchestrator."""

from __future__ import annotations

import time
import concurrent.futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence
from tqdm import tqdm

try:  # Optional - only used if source nodes are present
    from prediction_market_sim.data_sources.source_node import set_current_time
except Exception:  # pragma: no cover - keep simulation runnable without data_sources
    def set_current_time(_: int) -> None:
        return

from .interfaces import (
    Agent,
    Evaluator,
    MarketAdapter,
    MarketOrder,
    MessageStream,
    PortalNetwork,
)
from .logging import SimulationLogger, InformationFlowLogger, create_flow_logger


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
    enable_flow_logging: bool = False  # Enable information flow logging for animation


@dataclass(slots=True)
class SimulationResult:
    """Structured output for downstream evaluation/reporting."""

    run_id: int
    prices: List[float] = field(default_factory=list)
    belief_history: List[Mapping[str, float]] = field(default_factory=list)
    market_snapshots: List[Mapping[str, object]] = field(default_factory=list)
    evaluator_metrics: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    logger: Optional[SimulationLogger] = None
    flow_logger: Optional[InformationFlowLogger] = None  # For animation data
    log_files: Mapping[str, Path] = field(default_factory=dict)
    summary_stats: Mapping[str, object] = field(default_factory=dict)
    agent_pnl_history: List[Mapping[str, float]] = field(default_factory=list)
    trade_log: List[object] = field(default_factory=list)


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
        
        # Inject market into agents if they support it
        for agent in agents:
            if hasattr(agent, 'set_market'):
                agent.set_market(market)
        
        evaluators = [factory() for factory in self._evaluator_factories]

        stream.bootstrap(seed=seed)
        result = SimulationResult(run_id=run_id)

        # Initialize logger if enabled
        logger: Optional[SimulationLogger] = None
        if self._config.enable_logging:
            run_name = f"{self._config.run_name}_run{run_id}"
            logger = SimulationLogger(log_dir=self._config.log_dir, run_id=run_name)
            result.logger = logger

        # Initialize flow logger for animation if enabled
        flow_logger: Optional[InformationFlowLogger] = None
        if self._config.enable_flow_logging:
            run_name = f"{self._config.run_name}_run{run_id}"
            flow_logger = create_flow_logger(run_id=run_name, log_dir=self._config.log_dir)
            result.flow_logger = flow_logger

            # Register agent subscriptions for network topology
            # Also get persona name for display
            for agent in agents:
                subscriptions = []
                persona_name = None

                if hasattr(agent, 'persona') and agent.persona:
                    subscriptions = agent.persona.get('subscriptions', ['all'])
                    persona_name = agent.persona.get('name', None)
                elif hasattr(agent, 'subscriptions'):
                    subscriptions = list(agent.subscriptions)
                else:
                    subscriptions = ['all']

                # Expand "all" to actual source list for clearer visualization
                if 'all' in subscriptions:
                    # Default sources used in demo
                    subscriptions = ['reuters', 'twitter', 'analyst_report', 'insider']

                flow_logger.register_agent_subscriptions(
                    agent.agent_id,
                    subscriptions,
                    display_name=persona_name
                )

        current_price = market.current_price()
        timestep = 0

        with tqdm(total=self._config.max_timesteps, desc=f"Run {run_id}", unit="step", leave=True, position=0) as pbar_timestep:
            while timestep < self._config.max_timesteps:
                if stream.finished and self._config.stop_when_stream_finishes:
                    break

                set_current_time(timestep)

                timestep_start = time.time()

                # Reset per-tick market stats at start of each timestep
                if hasattr(market, 'reset_tick_stats'):
                    market.reset_tick_stats()

                messages = stream.next_batch()
                routed = portal.route(messages) if messages else {}
                belief_snapshot: Dict[str, float] = {}
                orders: List[MarketOrder] = []

                if logger and messages:
                    for message in messages:
                        logger.log_source_message(timestep, message)

                # Log information flow: routing of messages to agents
                if flow_logger and messages:
                    flow_logger.log_routing(timestep, messages, routed)

                pbar_timestep.write(f"[Timestep {timestep}] Processing {len(agents)} agents, {len(messages)} messages...")

                # Track previous beliefs for flow logging
                prev_beliefs = dict(flow_logger.agent_beliefs) if flow_logger else {}

                # Parallel agent processing
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_to_agent = {}
                    for agent in agents:
                        inbox = routed.get(agent.agent_id, [])
                        future = executor.submit(self._process_agent, agent, inbox, timestep, current_price, portal)
                        future_to_agent[future] = agent

                    with tqdm(total=len(agents), desc="  Agents", position=1, leave=False, disable=len(agents) <= 1) as pbar_agents:
                        for future in concurrent.futures.as_completed(future_to_agent):
                            agent = future_to_agent[future]
                            try:
                                agent_result = future.result()
                                belief_snapshot[agent.agent_id] = agent_result['belief']
                                if agent_result['order']:
                                    orders.append(agent_result['order'])

                                # Log belief update for flow visualization
                                if flow_logger:
                                    prev_belief = prev_beliefs.get(agent.agent_id, 0.5)
                                    flow_logger.log_belief_update(
                                        timestep=timestep,
                                        agent_id=agent.agent_id,
                                        belief_before=prev_belief,
                                        belief_after=agent_result['belief'],
                                        market_price=current_price,
                                    )

                                    # Log cross-posts if any
                                    for post in agent_result.get('posts', []):
                                        flow_logger.log_crosspost(
                                            timestep=timestep,
                                            agent_id=agent.agent_id,
                                            target_channel=post.get('target_node', 'unknown'),
                                            original_event_id=post.get('event_id'),
                                            content=post.get('content', ''),
                                            transformation='forwarded',
                                        )

                                agent_time = agent_result['time']
                                pbar_agents.set_postfix(time=f"{agent_time:.1f}s", belief=f"{agent_result['belief']:.3f}")
                                pbar_agents.update(1)
                            except Exception as e:
                                print(f"Agent {agent.agent_id} failed: {e}")

                if orders:
                    market.submit_orders(orders, timestep)

                    # Log trades for flow visualization
                    if flow_logger:
                        for order in orders:
                            flow_logger.log_trade(
                                timestep=timestep,
                                agent_id=order.agent_id,
                                side=order.side,
                                shares=order.size,
                                price=order.limit_price,
                                confidence=order.confidence,
                            )

                current_price = market.current_price()
                snapshot = market.snapshot()
                result.prices.append(current_price)
                result.belief_history.append(belief_snapshot)
                result.market_snapshots.append(snapshot)

                # Capture per-agent PnL if adapter supports it
                if hasattr(market, "get_agent_position"):
                    pnl_snapshot: Dict[str, float] = {}
                    for agent in agents:
                        try:
                            pos = market.get_agent_position(agent.agent_id)
                            pnl_snapshot[agent.agent_id] = float(pos.get("pnl", 0.0))
                        except Exception:
                            pnl_snapshot[agent.agent_id] = 0.0
                    result.agent_pnl_history.append(pnl_snapshot)

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

                timestep_time = time.time() - timestep_start
                pbar_timestep.write(f"[Timestep {timestep}] Complete in {timestep_time:.1f}s - {len(orders)} orders, price: {current_price:.4f}")
                pbar_timestep.set_postfix(price=f"{current_price:.4f}", orders=len(orders))
                pbar_timestep.update(1)
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

        # Save flow logs for animation
        if flow_logger:
            flow_path = flow_logger.save_to_json()
            result.log_files["flow"] = flow_path

        # Capture trades if adapter supports it
        if hasattr(market, "get_trades"):
            try:
                result.trade_log = list(market.get_trades())
            except Exception:
                result.trade_log = []

        return result

    def _process_agent(self, agent, inbox, timestep, current_price, portal):
        agent_start = time.time()

        if inbox:
            agent.ingest(inbox)

        belief = agent.update_belief(timestep, current_price)
        maybe_order = agent.generate_order(belief, current_price)

        # Optional: agent-generated posts (cross-posting)
        posts = []
        generate_posts = getattr(agent, "generate_posts", None)
        if callable(generate_posts):
            posts = generate_posts(timestep)
            for post in posts:
                portal.ingest_agent_feedback(agent.agent_id, post)

        return {
            'belief': belief,
            'order': maybe_order,
            'posts': posts,  # Include posts for flow logging
            'time': time.time() - agent_start
        }
