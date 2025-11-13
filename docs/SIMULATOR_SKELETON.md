# Simulator Skeleton Guide

This document explains the interfaces and flow for the high-level simulation
module so teammates can connect their agents, portals, data streams, and
evaluation logic without touching the orchestrator.

## Repository Layout

```
Prediction-Market-Agents-Simulations/
├── data/
│   ├── raw/         # drop raw AgentSociety or external datasets here
│   ├── processed/   # intermediate features, cached embeddings, etc.
│   └── external/    # third-party references (e.g., Kalshi, Twitter snapshots)
├── docs/
│   └── SIMULATOR_SKELETON.md
├── src/
│   └── prediction_market_sim/
│       ├── agents/          # teammate-owned agent logic
│       ├── data_sources/    # synthetic data + portal network modules
│       ├── evaluation/      # metric calculators
│       ├── market/          # custom market helpers/adapters
│       ├── simulation/      # orchestrator (engine, interfaces)
│       └── utils/           # shared helpers, logging, config parsing
```

## Architecture Overview

```
MessageStream -> PortalNetwork -> Agent(s) -> MarketAdapter -> Evaluators
                               \-> SimulationEngine loop
```

1. **MessageStream** (`src/prediction_market_sim/simulation/interfaces.py`)
   - Generates messages per timestep (synthetic data, AgentSociety dataset,
     API feeds, etc.).
   - Must implement `bootstrap(seed)`, `next_batch()`, and `finished`.
2. **PortalNetwork**
   - Routes messages to agent inboxes according to your network graph.
   - Optional `ingest_agent_feedback` hook lets agents push summaries back.
3. **Agent**
   - Owns belief updates and betting policy (Kelly vs. LLM-sized orders).
   - Exposes `ingest`, `update_belief`, `generate_order`.
4. **MarketAdapter**
   - Provides real price discovery. We ship
     `ExternalOrderBookAdapter` (wraps the [`orderbook`](https://pypi.org/project/orderbook/) PyPI lib).
   - Reimplement if you prefer LMSR or a custom market maker.
5. **Evaluator**
   - Computes metrics (Brier, ECE, price convergence, wealth, etc.) each tick
     and on `finalize()`.

## Simulation Engine

- Defined in `src/prediction_market_sim/simulation/engine.py`.
- Accepts **factories** for all module types plus a `SimulationRuntimeConfig`.
- Loop per timestep:
  1. Pull messages from `MessageStream`.
  2. Route via `PortalNetwork` to agent inboxes.
  3. Agents ingest messages, update beliefs, emit orders.
  4. Market adapter processes orders and returns new price snapshot.
  5. Evaluators observe prices/beliefs for metrics.
- `run_once` executes a single scenario; `run_many` helps with batch experiments
  (e.g., Kelly vs. LLM, symmetric vs. asymmetric info splits).

## Integration Checklist

1. **Provide factories** to `SimulationEngine`:
   ```python
   engine = SimulationEngine(
       stream_factory=lambda: MyTextStream(...),
       portal_factory=lambda: MyPortalNetwork(...),
       agent_factories=[lambda: KellyAgent(...), lambda: LLMAgent(...)],
       market_factory=lambda: ExternalOrderBookAdapter(initial_price=0.5),
       evaluator_factories=[lambda: BrierEvaluator(...), lambda: PriceConvergenceEvaluator(...)],
       runtime_config=SimulationRuntimeConfig(max_timesteps=500),
   )
   ```
2. **Ensure IDs align**:
   - `MessageStream` and `PortalNetwork` must agree on `source_id` keys.
   - `PortalNetwork.route` should return agent IDs matching `Agent.agent_id`.
3. **Stub dependencies** while your module is under construction. Returning
   `None` from `generate_order` skips market submission for that tick.
4. **Install optional deps** if you use our adapter:
   ```
   pip install orderbook
   ```
5. **Logging**: `SimulationResult` captures prices, beliefs, and market
   snapshots. Extend or serialize as needed once metric pipelines are ready.

By keeping the simulator thin, each teammate can iterate independently on data
generation, agent reasoning, market mechanics, or evaluation logic while
maintaining a common contract.
