"""Test LMSR Pricing and Generate Price Graph - NO HANGING"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - PREVENTS HANGING

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import random

# Import simulation components
from src.prediction_market_sim.simulation.engine import SimulationEngine, SimulationRuntimeConfig
from src.prediction_market_sim.simulation.interfaces import (
    Agent, MarketOrder, MessageStream, PortalNetwork, Evaluator
)
from src.prediction_market_sim.market import LMSRMarketAdapter

# Create output directory
output_dir = Path("pricing_test")
output_dir.mkdir(exist_ok=True)
print(f"[DIR] Created output directory: {output_dir}/")

# ===========================
# STUB COMPONENTS
# ===========================

class RandomTraderAgent(Agent):
    """Agent that trades randomly to test pricing"""
    def __init__(self, agent_id: str, initial_belief: float = 0.5):
        self._id = agent_id
        self._belief = initial_belief
    
    @property
    def agent_id(self) -> str:
        return self._id
    
    def receive_message(self, message: dict) -> None:
        # Random belief update based on sentiment
        sentiment = message.get("sentiment", 0.0)
        self._belief += sentiment * random.uniform(0.05, 0.15)
        self._belief = max(0.1, min(0.9, self._belief))  # Keep in [0.1, 0.9]
    
    def get_belief(self, outcome: str) -> float:
        return self._belief if outcome == "YES" else 1.0 - self._belief
    
    def generate_orders(self) -> list[MarketOrder]:
        # Trade based on belief vs current price
        # If we don't know price, trade randomly
        if random.random() < 0.3:  # 30% chance to trade
            side = "BUY" if self._belief > 0.5 else "SELL"
            quantity = random.uniform(5, 20)
            return [MarketOrder(
                agent_id=self._id,
                outcome="YES",
                side=side,
                quantity=quantity,
                order_type="market"
            )]
        return []

class SimpleMessageStream(MessageStream):
    """Generates random news events with sentiment"""
    def __init__(self, num_ticks: int):
        self.num_ticks = num_ticks
        self.current_tick = 0
    
    def get_messages(self, timestep: int) -> list[dict]:
        if timestep < self.num_ticks:
            # Random sentiment: positive or negative news
            sentiment = random.choice([-0.2, -0.1, 0.0, 0.1, 0.2])
            return [{
                "source": "news",
                "content": f"Market update at t={timestep}",
                "sentiment": sentiment,
                "timestamp": timestep
            }]
        return []

class SimplePortal(PortalNetwork):
    def route_messages(self, messages: list[dict], agents: list[Agent]) -> None:
        for agent in agents:
            for msg in messages:
                agent.receive_message(msg)

class SimpleEvaluator(Evaluator):
    def evaluate(self, timestep: int, market_state: dict, agents: list[Agent]) -> dict:
        return {"timestep": timestep}

# ===========================
# RUN SIMULATION
# ===========================

print("\n[RUN] Running LMSR pricing simulation...")
print("="*80)

# Create market
market = LMSRMarketAdapter(
    market_id="pricing_test",
    outcomes=["YES", "NO"],
    liquidity_param=100.0,
    initial_beliefs={"YES": 0.5, "NO": 0.5}
)

# Create agents
agents = [
    RandomTraderAgent("trader_1", 0.6),
    RandomTraderAgent("trader_2", 0.4),
    RandomTraderAgent("trader_3", 0.7),
    RandomTraderAgent("trader_4", 0.3),
    RandomTraderAgent("trader_5", 0.5),
]

# Create simulation
num_ticks = 50
engine = SimulationEngine(
    agents=agents,
    message_stream=SimpleMessageStream(num_ticks),
    portal_network=SimplePortal(),
    market_adapter=market,
    evaluator=SimpleEvaluator(),
    config=SimulationRuntimeConfig(
        max_ticks=num_ticks,
        enable_logging=False  # We'll track prices manually
    )
)

# Track price over time
price_history = []
timesteps = []

print(f"Simulating {num_ticks} timesteps...\n")

for tick in range(num_ticks):
    # Record price before tick
    state = market.get_market_state()
    price = state["price"]
    timesteps.append(tick)
    price_history.append(price)
    
    if tick % 10 == 0:
        print(f"t={tick:3d}: Price = {price:.4f} ({price*100:.2f}%)")
    
    # Run one tick
    engine._run_tick(tick)

# Final state
final_state = market.get_market_state()
final_price = final_state["price"]
timesteps.append(num_ticks)
price_history.append(final_price)

print(f"t={num_ticks:3d}: Price = {final_price:.4f} ({final_price*100:.2f}%)")
print("\n" + "="*80)
print("[DONE] Simulation complete!")
print(f"   Initial Price: {price_history[0]:.4f}")
print(f"   Final Price:   {price_history[-1]:.4f}")
print(f"   Change:        {price_history[-1] - price_history[0]:+.4f}")
print(f"   Total Volume:  {final_state['total_volume']:.2f} shares")
print(f"   Total Trades:  {final_state['num_trades']}")

# ===========================
# CREATE PLOT
# ===========================

print("\n[PLOT] Creating price graph...")

fig, ax = plt.subplots(figsize=(12, 6))

# Plot price line
ax.plot(timesteps, price_history, 'b-', linewidth=2, label='Market Price')

# Add horizontal reference lines
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% (Initial)')
ax.axhline(y=price_history[-1], color='red', linestyle=':', alpha=0.7, label=f'Final: {price_history[-1]:.2%}')

# Styling
ax.set_xlabel('Timestep', fontsize=12, fontweight='bold')
ax.set_ylabel('Price (Probability)', fontsize=12, fontweight='bold')
ax.set_title('LMSR Market Price Evolution Over Time', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='best')

# Format y-axis as percentage
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

# Save plot
plot_filename = output_dir / "price_evolution.png"
plt.tight_layout()
plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
plt.close()

print(f"[SAVED] Saved plot: {plot_filename}")

# ===========================
# CREATE MARKDOWN REPORT
# ===========================

print("\n[DOC] Creating markdown report...")

md_content = f"""# LMSR Pricing Test Results

## Test Configuration
- **Market Type:** LMSR (Logarithmic Market Scoring Rule)
- **Liquidity Parameter (b):** 100.0
- **Number of Agents:** {len(agents)}
- **Simulation Length:** {num_ticks} timesteps

## Pricing Formula

The LMSR uses the following formula to calculate prices:

```
Price_YES = e^(q_YES / b) / (e^(q_YES / b) + e^(q_NO / b))
```

Where:
- `q_YES`, `q_NO` = cumulative shares purchased
- `b` = liquidity parameter (controls price sensitivity)

## Results Summary

| Metric | Value |
|--------|-------|
| Initial Price | {price_history[0]:.4f} ({price_history[0]*100:.2f}%) |
| Final Price | {price_history[-1]:.4f} ({price_history[-1]*100:.2f}%) |
| Price Change | {price_history[-1] - price_history[0]:+.4f} ({(price_history[-1] - price_history[0])*100:+.2f}%) |
| Min Price | {min(price_history):.4f} ({min(price_history)*100:.2f}%) |
| Max Price | {max(price_history):.4f} ({max(price_history)*100:.2f}%) |
| Total Volume | {final_state['total_volume']:.2f} shares |
| Total Trades | {final_state['num_trades']} |

## Price Evolution Graph

![Price Evolution Over Time](price_evolution.png)

## Key Observations

1. **Price Dynamics:** The market price evolved from {price_history[0]:.2%} to {price_history[-1]:.2%}
2. **Volatility:** Price ranged from {min(price_history):.2%} to {max(price_history):.2%}
3. **Trading Activity:** {final_state['num_trades']} trades executed over {num_ticks} timesteps
4. **Liquidity:** Market maintained continuous liquidity via LMSR

## How LMSR Pricing Works

Unlike order-book markets (like Kalshi), LMSR uses an **automated market maker** that:

- **Always provides liquidity** - agents can trade at any time
- **Prices adjust algorithmically** - based on supply/demand via formula
- **No order matching needed** - instant execution at calculated price
- **Prices reflect probabilities** - converge to market consensus

This makes LMSR ideal for prediction markets where continuous trading and price discovery are essential.

---

*Generated by LMSR Pricing Test*
"""

md_filename = output_dir / "README.md"
with open(md_filename, 'w') as f:
    f.write(md_content)

print(f"[SAVED] Saved report: {md_filename}")

# ===========================
# SAVE RAW DATA
# ===========================

print("\n[DATA] Saving raw data...")

df = pd.DataFrame({
    'timestep': timesteps,
    'price': price_history,
    'price_pct': [p * 100 for p in price_history]
})
csv_filename = output_dir / "price_data.csv"
df.to_csv(csv_filename, index=False)

print(f"[SAVED] Saved data: {csv_filename}")

# ===========================
# SUMMARY
# ===========================

print("\n" + "="*80)
print("TEST COMPLETE!")
print("="*80)
print(f"\nAll files saved in: {output_dir}/")
print(f"   • README.md          - Markdown report with graph")
print(f"   • price_evolution.png - Price chart")
print(f"   • price_data.csv      - Raw price data")
print(f"\nOpen {output_dir}/README.md to view the full report!")
print("="*80)

