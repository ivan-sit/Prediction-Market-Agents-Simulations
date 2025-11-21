"""Simple text-based analysis - NO GRAPHICS, NO HANGING"""
import pandas as pd
from pathlib import Path

run_id = "lmsr_demo_run1"
log_dir = Path("simulation_logs")

print("="*80)
print(f"ANALYSIS: {run_id}")
print("="*80)

# Load market data
market_df = pd.read_csv(log_dir / f"{run_id}_market.csv")
print(f"\n📊 MARKET DATA ({len(market_df)} timesteps)")
print("-"*80)
print(f"Initial Price: {market_df['price'].iloc[0]:.4f} (50.00%)")
print(f"Final Price:   {market_df['price'].iloc[-1]:.4f} ({market_df['price'].iloc[-1]*100:.2f}%)")
print(f"Price Change:  {market_df['price'].iloc[-1] - market_df['price'].iloc[0]:+.4f}")
print(f"\nPrice Range:   {market_df['price'].min():.4f} - {market_df['price'].max():.4f}")
print(f"Average Price: {market_df['price'].mean():.4f}")

print(f"\n💰 TRADING VOLUME")
print("-"*80)
print(f"Total Volume:  {market_df['total_volume'].iloc[-1]:.2f} shares")
print(f"Num Trades:    {market_df['num_trades'].iloc[-1]}")
print(f"Avg per Tick:  {market_df['tick_volume'].mean():.2f} shares")

# Load beliefs
beliefs_df = pd.read_csv(log_dir / f"{run_id}_beliefs.csv")
print(f"\n👥 AGENT BELIEFS ({len(beliefs_df['agent_id'].unique())} agents)")
print("-"*80)

for agent in beliefs_df['agent_id'].unique():
    agent_data = beliefs_df[beliefs_df['agent_id'] == agent]
    initial = agent_data['belief'].iloc[0]
    final = agent_data['belief'].iloc[-1]
    change = final - initial
    print(f"{agent:20s}: {initial:.4f} → {final:.4f} (change: {change:+.4f})")

# Load sources
sources_df = pd.read_csv(log_dir / f"{run_id}_sources.csv")
print(f"\n📰 INFORMATION EVENTS ({len(sources_df)} messages)")
print("-"*80)
positive = len(sources_df[sources_df['sentiment'] > 0.1])
negative = len(sources_df[sources_df['sentiment'] < -0.1])
neutral = len(sources_df) - positive - negative
print(f"Positive News: {positive} ({positive/len(sources_df)*100:.1f}%)")
print(f"Negative News: {negative} ({negative/len(sources_df)*100:.1f}%)")
print(f"Neutral:       {neutral} ({neutral/len(sources_df)*100:.1f}%)")

print(f"\n📈 KEY TIMESTEPS (Price Movements)")
print("-"*80)
# Find biggest price changes
market_df['price_change'] = market_df['price'].diff().abs()
top_moves = market_df.nlargest(5, 'price_change')[['timestep', 'price', 'price_change', 'tick_volume']]
print(top_moves.to_string(index=False))

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE")
print("="*80)
print(f"\nData files in: {log_dir}/")
print(f"  • {run_id}_market.csv")
print(f"  • {run_id}_beliefs.csv")
print(f"  • {run_id}_sources.csv")


