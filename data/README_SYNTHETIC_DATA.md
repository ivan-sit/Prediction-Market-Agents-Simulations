# Synthetic Dataset Guide for Prediction Market Simulation

## Overview

This guide explains how to create synthetic datasets for the prediction market simulation. The simulation uses event-based data that unfolds over time, with agents reacting to information from different sources and placing trades.

## Standard Portal Names

Your synthetic datasets **must use only these portal IDs** in the `source_nodes` field:

| Portal ID | Reliability | Description |
|-----------|-------------|-------------|
| `twitter` | 0.60 | Social media - fast but noisy, lower reliability |
| `news_feed` | 0.80 | News outlets - moderate reliability, professional sources |
| `expert_analysis` | 0.90 | Expert opinions - high reliability, deep analysis |
| `reddit` | 0.65 | Community discussions - mixed reliability |
| `discord` | 0.65 | Real-time chat - community sentiment |

**Important:** These portals are pre-configured in the simulation. Events referencing other portal names will be silently dropped!

## Event Data Format

Your JSON file must follow this exact structure:

```json
{
  "events": [
    {
      "event_id": "evt_001",
      "initial_time": 0,
      "source_nodes": ["twitter", "reddit"],
      "tagline": "Short headline (1 sentence)",
      "description": "Detailed description of the event and its implications..."
    },
    {
      "event_id": "evt_002",
      "initial_time": 5,
      "source_nodes": ["news_feed", "expert_analysis"],
      "tagline": "Another event headline",
      "description": "More detailed information..."
    }
  ]
}
```

### Field Descriptions

- **event_id** (required, string): Unique identifier for the event (e.g., "evt_001", "evt_002")
- **initial_time** (required, integer): Timestep when this event occurs (0, 1, 2, ...)
- **source_nodes** (required, array): List of portal IDs where event appears (use only standard names above!)
- **tagline** (required, string): Brief headline/summary of the event
- **description** (required, string): Detailed description that agents will analyze

### Important Rules

1. **Events must be sorted by initial_time** (ascending order)
2. **Event IDs must be unique** across the dataset
3. **source_nodes must contain only standard portal names**
4. **initial_time values should be non-negative integers**
5. **No duplicate events** (same event_id appearing twice)

## Best Practices for Synthetic Data

### 1. Narrative Structure

Create a coherent story that unfolds over time:

- **Early events (t=0-10):** Weak signals, rumors, vague information
- **Mid events (t=10-25):** Stronger signals, mixed evidence, analysis
- **Late events (t=25+):** Definitive information, confirmations, resolutions

### 2. Source Distribution

Match information quality to source type:

- **twitter, reddit:** Early rumors, social sentiment, unverified claims
- **news_feed:** Official announcements, verified reports, interviews
- **expert_analysis:** Deep analysis, forecasts, professional opinions

### 3. Signal Progression

Show realistic information evolution:

```
Timestep 0:  twitter → "Rumor: Company X developing new product"
Timestep 5:  news_feed → "Analyst notes unusual patent filings from Company X"
Timestep 10: expert_analysis → "Expert: Patent filings suggest quantum computing focus"
Timestep 15: news_feed → "CONFIRMED: Company X announces quantum breakthrough"
```

### 4. Timing Patterns

- **Irregular intervals:** Don't post events every timestep (0, 3, 7, 10, 15...)
- **Information cascades:** Related events can cluster (multiple events at t=10)
- **Quiet periods:** Allow gaps where no new information arrives
- **Realistic pacing:** Major events need time for market to digest

### 5. Conflicting Information

Include some contrary signals to test agent reasoning:

```json
{
  "event_id": "evt_005",
  "initial_time": 12,
  "source_nodes": ["twitter"],
  "tagline": "Skeptics question Company X claims",
  "description": "Technical experts on social media express doubt about feasibility..."
}
```

## Example Dataset Themes

### Election Prediction Market

```json
{
  "events": [
    {"event_id": "evt_001", "initial_time": 0, "source_nodes": ["twitter"],
     "tagline": "Early polls show tight race", "description": "..."},
    {"event_id": "evt_002", "initial_time": 5, "source_nodes": ["news_feed"],
     "tagline": "Debate performance favors Candidate A", "description": "..."},
    {"event_id": "evt_003", "initial_time": 10, "source_nodes": ["expert_analysis"],
     "tagline": "Political scientist predicts Candidate A victory", "description": "..."}
  ]
}
```

### Product Launch Market

```json
{
  "events": [
    {"event_id": "evt_001", "initial_time": 0, "source_nodes": ["reddit"],
     "tagline": "Leak suggests new iPhone features", "description": "..."},
    {"event_id": "evt_002", "initial_time": 8, "source_nodes": ["news_feed"],
     "tagline": "Apple sends press invites for Sept event", "description": "..."},
    {"event_id": "evt_003", "initial_time": 15, "source_nodes": ["expert_analysis", "news_feed"],
     "tagline": "Official announcement confirms new features", "description": "..."}
  ]
}
```

### Economic Indicator Market

```json
{
  "events": [
    {"event_id": "evt_001", "initial_time": 0, "source_nodes": ["twitter"],
     "tagline": "Economists expect strong jobs report", "description": "..."},
    {"event_id": "evt_002", "initial_time": 10, "source_nodes": ["news_feed"],
     "tagline": "ADP private payrolls beat expectations", "description": "..."},
    {"event_id": "evt_003", "initial_time": 20, "source_nodes": ["expert_analysis", "news_feed"],
     "tagline": "Official BLS report: 250K jobs added", "description": "..."}
  ]
}
```

## Running the Simulation

Once you've created your synthetic dataset:

```bash
# Basic usage
python examples/run_with_synthetic_data.py --events data/my_events.json

# Full options
python examples/run_with_synthetic_data.py \
    --events data/my_events.json \
    --market lmsr \
    --agents 3 \
    --timesteps 50 \
    --liquidity 100.0 \
    --seed 42
```

### Parameters

- `--events`: Path to your JSON event file (required)
- `--market`: Market type - 'lmsr' (simple) or 'orderbook' (realistic)
- `--agents`: Number of agents (1-5, default: 3)
- `--timesteps`: Maximum simulation timesteps (default: 50)
- `--liquidity`: Market liquidity parameter (default: 100.0)
- `--seed`: Random seed for reproducibility (default: 42)

## Agent Subscriptions

The simulation creates agents with different portal subscriptions:

| Agent | Personality | Subscribed Portals |
|-------|-------------|-------------------|
| conservative_trader | Risk-averse, values quality | news_feed, expert_analysis |
| aggressive_trader | Bold, follows sentiment | twitter, reddit, news_feed |
| contrarian_trader | Bets against crowd | twitter, expert_analysis |
| well_informed_trader | Monitors everything | twitter, news_feed, expert_analysis, reddit |
| social_trader | Follows community | twitter, reddit, discord |

Agents only see events from their subscribed portals!

## Simulation Behavior

### Each Timestep:

1. **Event posting:** Events at current timestep are posted to their source_nodes
2. **Agent reading:** Agents read new events from their subscribed portals
3. **Agent reasoning:** LLM agents analyze events and update beliefs
4. **Order generation:** Agents may place BUY/SELL orders based on analysis
5. **Market execution:** Orders are processed, price updates
6. **Cross-posting:** Agents may post their analysis to other portals (information cascade!)

### Key Features:

- **Agents have memory:** Past events influence future decisions
- **Agent personalities matter:** Conservative vs aggressive trading styles
- **Information cascades:** Agents can amplify signals by cross-posting
- **Differential information:** Agents see different events based on subscriptions

## Output Files

After simulation completes, check `artifacts/` directory:

- `{run_name}_run1_market.csv/json`: Price evolution, volume, trades
- `{run_name}_run1_beliefs.csv/json`: Agent beliefs over time
- `{run_name}_run1_sources.csv/json`: All events that occurred

## Validation Checklist

Before running your synthetic dataset:

- [ ] JSON is valid (use JSONLint or similar)
- [ ] All events have required fields (event_id, initial_time, source_nodes, tagline, description)
- [ ] Events are sorted by initial_time
- [ ] All source_nodes use only standard portal names
- [ ] Event IDs are unique
- [ ] initial_time values are non-negative integers
- [ ] Narrative is coherent and realistic
- [ ] Signal strength progresses logically
- [ ] Dataset has 10-30 events for good simulation

## Troubleshooting

**"Events file not found"**
- Check file path is correct
- Use relative path from project root

**"No events processed"**
- Verify JSON structure matches exactly
- Check events array is not empty
- Ensure initial_time values are within timesteps range

**"Agents not trading"**
- Events may not be informative enough
- Check agent subscriptions match event source_nodes
- Verify LLM (Ollama) is running: `ollama serve`

**"Prices not moving"**
- Events may be too weak/ambiguous
- Try stronger signal language in descriptions
- Increase number of agents for more activity

## Example: Creating an Election Dataset

See `data/sample_election_events.json` for a complete example showing:
- Progressive information revelation
- Multiple source types
- Conflicting signals
- Realistic timing
- Strong narrative arc

Study this example when creating your own datasets!
