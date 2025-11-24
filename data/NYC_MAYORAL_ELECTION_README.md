# NYC Mayoral Election 2025 - Event Data

This dataset contains 25 real events from the 2025 New York City mayoral election, formatted for the Prediction Market Agents Simulation framework.

## Dataset Information

- **File**: `data/nyc_mayoral_election_2025.json`
- **Events**: 25 events spanning the entire election cycle
- **Timeline**: Timesteps 0-65 (simulating ~6 months of the campaign)
- **Winner**: Zohran Mamdani (Democratic) - 50.4%
- **Election Date**: November 4, 2025

## Key Candidates

1. **Zohran Mamdani** (Democratic nominee) - Progressive/Democratic Socialist
   - Campaign focus: Affordability, rent freeze, free bus service, universal child care
   - Won with 50.4% of the vote
   - First Muslim and South Asian NYC mayor

2. **Andrew Cuomo** (Independent) - Centrist
   - Former NY Governor attempting comeback
   - Platform: Crime reduction, combating antisemitism
   - Received endorsements from Trump, Musk, and Adams

3. **Curtis Sliwa** (Republican)
   - Guardian Angels founder
   - Tough-on-crime platform

## Event Timeline Highlights

| Timestep | Date | Key Event |
|----------|------|-----------|
| 0 | Sept 2024 | Adams indicted on federal charges |
| 5 | March 2025 | Cuomo announces mayoral campaign |
| 15 | June 24, 2025 | Mamdani wins Democratic primary (major upset) |
| 22 | Sept 10, 2025 | Poll shows Mamdani up 22 points |
| 35 | Oct 16, 2025 | First general election debate |
| 40 | Oct 22, 2025 | Final debate |
| 55 | Nov 3, 2025 | Trump & Musk endorse Cuomo |
| 60 | Nov 4, 2025 | Election Day - Mamdani wins |

## Information Sources Used

Events are distributed across five information portals with varying reliability:

- **news_feed** (reliability: 0.80) - Professional news outlets, major announcements
- **expert_analysis** (reliability: 0.90) - Polling data, political analysis
- **twitter** (reliability: 0.60) - Social media trends, viral moments
- **reddit** (reliability: 0.65) - Community discussions, grassroots movements
- **discord** (reliability: 0.65) - Real-time political discourse

## Running the Simulation

### Basic Usage

```bash
python examples/run_with_synthetic_data.py \
    --events data/nyc_mayoral_election_2025.json \
    --market lmsr \
    --agents 5 \
    --timesteps 70 \
    --liquidity 100.0 \
    --seed 42
```

### With Order Book Market (More Realistic)

```bash
python examples/run_with_synthetic_data.py \
    --events data/nyc_mayoral_election_2025.json \
    --market orderbook \
    --agents 10 \
    --timesteps 70 \
    --seed 42
```

### Custom Configuration

```bash
python examples/run_with_synthetic_data.py \
    --events data/nyc_mayoral_election_2025.json \
    --market lmsr \
    --agents 8 \
    --timesteps 70 \
    --liquidity 150.0 \
    --seed 123 \
    --run-name nyc_election_sim
```

## Expected Simulation Behavior

The simulation will generate agents that:

1. **Subscribe to different information sources** based on their personalities
2. **Receive events at specified timesteps** (0-65)
3. **Update beliefs** about Mamdani's chances of winning
4. **Generate trades** (BUY/SELL) based on their beliefs vs. market price
5. **Compete in the prediction market** to profit from information advantages

### Key Trading Moments

Agents should exhibit interesting trading behavior around:

- **Timestep 15**: Primary upset win for Mamdani (major belief update)
- **Timestep 22**: Poll showing 22-point lead (bullish signal)
- **Timestep 28**: Adams endorses Cuomo (bearish for Mamdani)
- **Timestep 52**: Race tightens in final polls (increased volatility)
- **Timestep 55**: Trump/Musk endorsement (uncertain impact)
- **Timestep 60**: Election results (resolution)

## Output Files

After running the simulation, you'll find in `simulation_logs/`:

1. **`{run_name}_run1_market.json`** - Price evolution, volume, trades
2. **`{run_name}_run1_beliefs.json`** - Agent belief trajectories
3. **`{run_name}_run1_sources.json`** - Event logs with source reliability

## Analysis Questions

This dataset is ideal for exploring:

1. **Information efficiency**: Do agents with access to expert_analysis outperform those relying on social media?
2. **Belief updating**: How quickly do agents incorporate major events (primary win, endorsements)?
3. **Polarization**: Do agents with different information sources develop divergent beliefs?
4. **Contrarian trading**: Do some agents profit from betting against consensus after Trump endorsement?
5. **Market accuracy**: Does the final market price converge to ~50% (Mamdani's actual vote share)?

## Data Source

All events are based on real-world coverage of the 2025 NYC mayoral election from:
- NBC News, CBS News, ABC News, CNN
- The New York Times, Washington Post, PBS
- Quinnipiac University, Marist Poll, AtlasIntel (polling)
- Wikipedia's comprehensive timeline

The events have been adapted and sequenced for simulation purposes while maintaining factual accuracy.

## License

This dataset is provided for educational and research purposes. All event descriptions are based on publicly available news reporting.
