"""Quick visualization - no hanging!"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Load the data
run_id = "lmsr_demo_run1"
market_df = pd.read_csv(f"simulation_logs/{run_id}_market.csv")
beliefs_df = pd.read_csv(f"simulation_logs/{run_id}_beliefs.csv")
sources_df = pd.read_csv(f"simulation_logs/{run_id}_sources.csv")

print(f"[OK] Loaded {len(market_df)} market records")
print(f"[OK] Loaded {len(beliefs_df)} belief records")
print(f"[OK] Loaded {len(sources_df)} source messages")

# Create output directory
viz_dir = Path("visualizations")
viz_dir.mkdir(exist_ok=True)

# Plot 1: Price Evolution
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(market_df['timestep'], market_df['price'], linewidth=2, color='#2E86AB')
ax.fill_between(market_df['timestep'], 0, market_df['price'], alpha=0.3, color='#2E86AB')
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% (neutral)')
ax.set_xlabel('Timestep', fontsize=12)
ax.set_ylabel('Market Price (Probability)', fontsize=12)
ax.set_title('Prediction Market Price Evolution', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(viz_dir / f"{run_id}_price.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"[SAVED] {viz_dir}/{run_id}_price.png")

# Plot 2: Trading Volume
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Volume
ax1.bar(market_df['timestep'], market_df['tick_volume'], color='#A23B72', alpha=0.7)
ax1.set_ylabel('Volume per Tick', fontsize=12)
ax1.set_title('Trading Activity', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Net Flow (buy pressure)
colors = ['green' if x > 0 else 'red' for x in market_df['net_flow']]
ax2.bar(market_df['timestep'], market_df['net_flow'], color=colors, alpha=0.7)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.set_xlabel('Timestep', fontsize=12)
ax2.set_ylabel('Net Flow (Buy - Sell)', fontsize=12)
ax2.set_title('Buy/Sell Pressure', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(viz_dir / f"{run_id}_volume.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"[SAVED] {viz_dir}/{run_id}_volume.png")

# Plot 3: Agent Beliefs vs Market Price
fig, ax = plt.subplots(figsize=(12, 6))

# Plot market price
ax.plot(market_df['timestep'], market_df['price'], 
        linewidth=3, color='black', label='Market Price', zorder=10)

# Plot each agent's beliefs
agents = beliefs_df['agent_id'].unique()
colors = ['#E63946', '#F77F00', '#06AED5', '#86BBD8']
for i, agent in enumerate(agents):
    agent_data = beliefs_df[beliefs_df['agent_id'] == agent]
    ax.plot(agent_data['timestep'], agent_data['belief'], 
            linewidth=1.5, alpha=0.7, label=agent, color=colors[i % len(colors)])

ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Timestep', fontsize=12)
ax.set_ylabel('Probability Belief', fontsize=12)
ax.set_title('Agent Beliefs vs Market Price', fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(viz_dir / f"{run_id}_beliefs.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"[SAVED] {viz_dir}/{run_id}_beliefs.png")

# Plot 4: Information Sources (Events)
fig, ax = plt.subplots(figsize=(12, 6))

# Color code by sentiment
colors = []
for sentiment in sources_df['sentiment']:
    if sentiment > 0.1:
        colors.append('green')
    elif sentiment < -0.1:
        colors.append('red')
    else:
        colors.append('gray')

ax.scatter(sources_df['timestep'], sources_df['sentiment'], 
          c=colors, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Timestep', fontsize=12)
ax.set_ylabel('Sentiment Signal', fontsize=12)
ax.set_title('Information Events (News/Signals)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', label='Positive News'),
    Patch(facecolor='gray', label='Neutral'),
    Patch(facecolor='red', label='Negative News')
]
ax.legend(handles=legend_elements, loc='best')

plt.tight_layout()
plt.savefig(viz_dir / f"{run_id}_sources.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"[SAVED] {viz_dir}/{run_id}_sources.png")

print("\n" + "="*80)
print("[DONE] ALL VISUALIZATIONS COMPLETE!")
print("="*80)
print(f"\nView results in: {viz_dir}/")
print(f"  • {run_id}_price.png - Price evolution over time")
print(f"  • {run_id}_volume.png - Trading volume and buy/sell pressure")
print(f"  • {run_id}_beliefs.png - Agent beliefs vs market price")
print(f"  • {run_id}_sources.png - Information events timeline")


