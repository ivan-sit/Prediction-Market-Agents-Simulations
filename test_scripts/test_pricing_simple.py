"""Simple LMSR Pricing Test - Minimal Dependencies"""
from pathlib import Path
from src.prediction_market_sim.market import LMSRMarket

# Create output directory
output_dir = Path("pricing_test")
output_dir.mkdir(exist_ok=True)

print("="*80)
print("LMSR PRICING TEST")
print("="*80)

# Create market
market = LMSRMarket(
    liquidity_param=100.0,
    initial_shares={"YES": 0.0, "NO": 0.0}
)

print("\nTesting LMSR Pricing Formula")
print("-"*80)

# Track prices
timesteps = []
prices = []
volumes = []

# Initial state
initial_price = market.get_price("YES")
print(f"Initial Price: {initial_price:.4f} ({initial_price*100:.2f}%)")

timesteps.append(0)
prices.append(initial_price)
volumes.append(0.0)

# Simulate trades
print("\nSimulating 30 trades...")
trades = [
    ("BUY", 10),   # Buy 10 YES shares
    ("BUY", 15),   # Buy 15 more
    ("SELL", 5),   # Sell 5
    ("BUY", 20),   # Buy 20
    ("SELL", 10),  # Sell 10
    ("BUY", 10),   # Buy 10
    ("BUY", 5),    # Buy 5
    ("SELL", 15),  # Sell 15
    ("BUY", 25),   # Buy 25
    ("SELL", 8),   # Sell 8
] * 3  # Repeat 3 times = 30 trades

total_volume = 0
for i, (side, qty) in enumerate(trades, 1):
    # In LMSR, selling is just buying negative shares
    shares = qty if side == "BUY" else -qty
    
    market.buy_shares(
        agent_id=f"agent_{i%5}",
        outcome="YES",
        num_shares=shares,
        timestamp=i
    )
    
    total_volume += qty
    price = market.get_price("YES")
    
    timesteps.append(i)
    prices.append(price)
    volumes.append(total_volume)
    
    if i % 5 == 0:
        print(f"  Trade {i:2d}: {side:4s} {qty:2d} shares → Price: {price:.4f} ({price*100:.2f}%)")

final_price = market.get_price("YES")

print("\n" + "-"*80)
print(f"Final Price:   {final_price:.4f} ({final_price*100:.2f}%)")
print(f"Price Change:  {final_price - initial_price:+.4f} ({(final_price - initial_price)*100:+.2f}%)")
print(f"Total Volume:  {total_volume:.2f} shares")

# Save data to CSV
csv_data = "timestep,price,price_pct,volume\n"
for t, p, v in zip(timesteps, prices, volumes):
    csv_data += f"{t},{p:.6f},{p*100:.4f},{v:.2f}\n"

csv_file = output_dir / "price_data.csv"
with open(csv_file, 'w') as f:
    f.write(csv_data)

print(f"\n[SAVED] {csv_file}")

# Create simple markdown
md = f"""# LMSR Pricing Test Results

## Summary

| Metric | Value |
|--------|-------|
| Initial Price | {initial_price:.4f} ({initial_price*100:.2f}%) |
| Final Price | {final_price:.4f} ({final_price*100:.2f}%) |
| Price Change | {final_price - initial_price:+.4f} ({(final_price - initial_price)*100:+.2f}%) |
| Min Price | {min(prices):.4f} ({min(prices)*100:.2f}%) |
| Max Price | {max(prices):.4f} ({max(prices)*100:.2f}%) |
| Total Trades | {len(trades)} |
| Total Volume | {total_volume:.2f} shares |

## LMSR Pricing Formula

```
Price_YES = e^(q_YES / b) / (e^(q_YES / b) + e^(q_NO / b))
```

Where:
- `q_YES`, `q_NO` = cumulative shares purchased  
- `b` = liquidity parameter = 100.0

## Key Points

**Automated Market Maker** - No order matching needed
**Always Liquid** - Can trade at any time
**Formula-Based Pricing** - Prices adjust algorithmically
**Probability Interpretation** - Price = probability of YES outcome  

## Price Data

See `price_data.csv` for full timestep data.

---

*Test completed successfully*
"""

md_file = output_dir / "README.md"
with open(md_file, 'w') as f:
    f.write(md)

print(f"[SAVED] {md_file}")

print("\n" + "="*80)
print("[DONE] TEST COMPLETE")
print("="*80)
print(f"\nFiles in {output_dir}/:")
print(f"  • README.md       - Results summary")
print(f"  • price_data.csv  - Raw price data")
print("\nNow generating graph...")

