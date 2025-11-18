"""
Full Integration Demo - Connects All Modules

This example demonstrates the complete simulation with:
1. Event data from your colleague's data module
2. Source nodes (portals) from data_sources module  
3. LLM-based agents from agents module
4. Market (LMSR or OrderBook) from market module
5. Simulation orchestration and logging

Run with: python examples/full_integration_demo.py
"""

from pathlib import Path

from prediction_market_sim.simulation.engine import (
    SimulationEngine,
    SimulationRuntimeConfig
)
from prediction_market_sim.data_sources import (
    SourceNode,
    create_event_stream,
    create_portal_network
)
from prediction_market_sim.agents import create_prediction_agent
from prediction_market_sim.market import LMSRMarketAdapter
from prediction_market_sim.simulation.interfaces import Evaluator
from typing import Mapping


class SimpleEvaluator:
    """Basic evaluator to track simulation metrics."""
    
    def __init__(self):
        self.tick_count = 0
        self.total_price_change = 0.0
        self.last_price = 0.5
        
    def on_tick(
        self,
        *,
        timestep: int,
        price: float,
        agent_beliefs: Mapping[str, float],
        market_snapshot: Mapping[str, object]
    ) -> None:
        self.tick_count += 1
        self.total_price_change += abs(price - self.last_price)
        self.last_price = price
        
    def finalize(self) -> Mapping[str, float]:
        return {
            'total_timesteps': float(self.tick_count),
            'avg_price_volatility': self.total_price_change / max(1, self.tick_count),
            'final_price': self.last_price
        }


def main():
    print("=" * 60)
    print("🚀 FULL PREDICTION MARKET SIMULATION - INTEGRATED")
    print("=" * 60)
    print()
    
    # Step 1: Configure paths
    project_root = Path(__file__).parent.parent
    events_db = project_root / "artifacts" / "sample_events.json"
    logs_dir = project_root / "simulation_logs" / "full_integration"
    
    print(f"📂 Events Database: {events_db}")
    print(f"📂 Logs Directory: {logs_dir}")
    print()
    
    # Step 2: Create source nodes (information portals)
    print("📡 Setting up information portals...")
    portal_network = create_portal_network([
        {'node_id': 'twitter', 'reliability': 0.6},
        {'node_id': 'news_feed', 'reliability': 0.8},
        {'node_id': 'expert_analysis', 'reliability': 0.9}
    ])
    
    # Step 3: Create agents with personalities
    print("🤖 Creating agents...")
    
    agent_configs = [
        {
            'agent_id': 'alice',
            'personality': 'Conservative trader who values expert analysis and avoids risk',
            'initial_cash': 10000.0
        },
        {
            'agent_id': 'bob',
            'personality': 'Aggressive momentum trader who follows social media sentiment',
            'initial_cash': 10000.0
        },
        {
            'agent_id': 'charlie',
            'personality': 'Contrarian value investor who bets against public opinion',
            'initial_cash': 10000.0
        }
    ]
    
    agent_factories = []
    for config in agent_configs:
        def make_agent(cfg=config):  # Closure to capture config
            agent = create_prediction_agent(
                agent_id=cfg['agent_id'],
                personality=cfg['personality'],
                initial_cash=cfg['initial_cash']
            )
            return agent
        agent_factories.append(make_agent)
    
    # Subscribe agents to different portals based on their personality
    portal_network.subscribe_agent('alice', ['news_feed', 'expert_analysis'])
    portal_network.subscribe_agent('bob', ['twitter', 'news_feed'])
    portal_network.subscribe_agent('charlie', ['twitter', 'expert_analysis'])
    
    print(f"   ✅ Alice (conservative) - subscribes to: news_feed, expert_analysis")
    print(f"   ✅ Bob (aggressive) - subscribes to: twitter, news_feed")
    print(f"   ✅ Charlie (contrarian) - subscribes to: twitter, expert_analysis")
    print()
    
    # Step 4: Create market (LMSR for now)
    print("💹 Setting up market...")
    print("   Market Type: LMSR (Automated Market Maker)")
    print("   Liquidity Parameter: 100.0")
    print()
    
    def market_factory():
        return LMSRMarketAdapter(
            market_id="home_team_wins",
            outcomes=["YES", "NO"],
            liquidity_param=100.0
        )
    
    # Step 5: Configure simulation
    runtime_config = SimulationRuntimeConfig(
        max_timesteps=20,
        log_dir=logs_dir,
        run_name="full_integration_v1",
        log_every=1,
        enable_logging=True,
        save_logs_as_csv=True,
        save_logs_as_json=True
    )
    
    # Step 6: Build the engine
    print("🔧 Building simulation engine...")
    engine = SimulationEngine(
        stream_factory=lambda: create_event_stream(str(events_db)),
        portal_factory=lambda: portal_network,
        agent_factories=agent_factories,
        market_factory=market_factory,
        evaluator_factories=[lambda: SimpleEvaluator()],
        runtime_config=runtime_config
    )
    print("   ✅ Engine ready!")
    print()
    
    # Step 7: Run simulation
    print("▶️  Running simulation...")
    print("-" * 60)
    
    try:
        result = engine.run_once(run_id=1, seed=42)
        
        print()
        print("-" * 60)
        print("✅ Simulation Complete!")
        print()
        
        # Display results
        print("📊 RESULTS SUMMARY")
        print("=" * 60)
        print(f"Total Timesteps: {len(result.prices)}")
        print(f"Starting Price: {result.prices[0]:.4f}" if result.prices else "N/A")
        print(f"Final Price: {result.prices[-1]:.4f}" if result.prices else "N/A")
        print(f"Price Range: [{min(result.prices):.4f}, {max(result.prices):.4f}]" if result.prices else "N/A")
        print()
        
        print("📈 Market Price Evolution:")
        for i, price in enumerate(result.prices[:10], 1):  # Show first 10
            bar = "█" * int(price * 40)
            print(f"   T{i:02d}: {bar} {price:.3f}")
        if len(result.prices) > 10:
            print(f"   ... ({len(result.prices) - 10} more timesteps)")
        print()
        
        print("📁 Log Files:")
        if result.log_files:
            for log_type, path in result.log_files.items():
                print(f"   {log_type}: {path}")
        print()
        
        print("📊 Summary Statistics:")
        if result.summary_stats:
            for key, value in result.summary_stats.items():
                print(f"   {key}: {value}")
        print()
        
        print("🎯 Evaluator Metrics:")
        for eval_name, metrics in result.evaluator_metrics.items():
            print(f"   {eval_name}:")
            for metric, value in metrics.items():
                print(f"      {metric}: {value:.4f}")
        print()
        
        print("=" * 60)
        print("🎉 Integration successful! All modules working together.")
        print("=" * 60)
        
    except Exception as e:
        print()
        print(f"❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("💡 Troubleshooting:")
        print("   1. Make sure Ollama is running: ollama serve")
        print("   2. Check that sample_events.json exists")
        print("   3. Verify all dependencies are installed")


if __name__ == "__main__":
    main()

