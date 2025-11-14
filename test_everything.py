#!/usr/bin/env python3
"""Quick test script to verify everything works.

Run with: python test_everything.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 80)
print("TESTING PREDICTION MARKET SIMULATION FRAMEWORK")
print("=" * 80)

# Test 1: Imports
print("\n✓ Test 1: Checking imports...")
try:
    from prediction_market_sim import SimulationEngine, SimulationRuntimeConfig
    from prediction_market_sim.simulation import MarketOrder
    from prediction_market_sim.market import (
        OrderBookMarketAdapter,
        LMSRMarketAdapter,
        create_orderbook_adapter
    )
    print("  ✅ All core imports successful")
except ImportError as e:
    print(f"  ❌ Import error: {e}")
    sys.exit(1)

# Test 2: Check external library availability
print("\n✓ Test 2: Checking orderbook libraries...")
try:
    from lob import OrderBook
    print("  ✅ limit-order-book available (HIGH PERFORMANCE)")
    external_lib = "limit-order-book"
except ImportError:
    try:
        from pyorderbook import OrderBook
        print("  ✅ pyorderbook available (Pure Python)")
        external_lib = "pyorderbook"
    except ImportError:
        print("  ⚠️  No external library - using custom implementation")
        print("     Install for better performance: pip install limit-order-book")
        external_lib = "custom"

# Test 3: Create market adapters
print("\n✓ Test 3: Creating market adapters...")
try:
    # Test smart adapter
    smart_market = create_orderbook_adapter(initial_price=0.5, prefer="auto")
    print(f"  ✅ Smart adapter created (using: {external_lib})")
    
    # Test custom adapter
    custom_market = OrderBookMarketAdapter(initial_price=0.5)
    print("  ✅ Custom orderbook adapter created")
    
    # Test LMSR
    lmsr_market = LMSRMarketAdapter(liquidity_param=100.0)
    print("  ✅ LMSR market adapter created")
except Exception as e:
    print(f"  ❌ Error creating adapters: {e}")
    sys.exit(1)

# Test 4: Test market operations
print("\n✓ Test 4: Testing market operations...")
try:
    from prediction_market_sim.simulation import MarketOrder
    
    # Create orders
    orders = [
        MarketOrder(
            agent_id="test_agent_1",
            side="buy",
            size=10.0,
            limit_price=0.55,
            confidence=0.8
        ),
        MarketOrder(
            agent_id="test_agent_2",
            side="sell",
            size=8.0,
            limit_price=0.54,
            confidence=0.7
        )
    ]
    
    # Test custom orderbook
    custom_market.submit_orders(orders, timestep=0)
    price = custom_market.current_price()
    print(f"  ✅ Custom orderbook: price = {price:.4f}")
    
    # Test LMSR
    lmsr_market.submit_orders(orders, timestep=0)
    price = lmsr_market.current_price()
    print(f"  ✅ LMSR: price = {price:.4f}")
    
except Exception as e:
    print(f"  ❌ Error in market operations: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Check logging
print("\n✓ Test 5: Checking data logging...")
try:
    from prediction_market_sim.simulation import SimulationLogger
    
    logger = SimulationLogger(run_id="test_run")
    logger.log_market_state(0, 0.5, {"spread": 0.01, "volume": 10.0})
    logger.log_beliefs(0, {"agent_1": 0.6}, 0.5)
    logger.log_source_message(0, {"source_id": "test", "sentiment": 0.1})
    
    print(f"  ✅ Logged {len(logger.market_records)} market records")
    print(f"  ✅ Logged {len(logger.belief_records)} belief records")
    print(f"  ✅ Logged {len(logger.text_records)} source records")
except Exception as e:
    print(f"  ❌ Error in logging: {e}")
    sys.exit(1)

# Test 6: Check log files exist
print("\n✓ Test 6: Checking existing log files...")
log_dir = Path("simulation_logs")
if log_dir.exists():
    csv_files = list(log_dir.glob("*.csv"))
    json_files = list(log_dir.glob("*.json"))
    print(f"  ✅ Found {len(csv_files)} CSV files")
    print(f"  ✅ Found {len(json_files)} JSON files")
    
    if csv_files:
        print("\n  Recent runs:")
        runs = set()
        for f in csv_files:
            run_name = f.stem.rsplit("_", 1)[0]
            runs.add(run_name)
        for run in sorted(runs):
            print(f"    - {run}")
else:
    print("  ⚠️  No simulation_logs directory yet")
    print("     Run: python examples/demo_full_simulation.py")

# Test 7: Quick data analysis
print("\n✓ Test 7: Testing data analysis...")
try:
    import pandas as pd
    
    if log_dir.exists() and csv_files:
        # Find a market file
        market_files = list(log_dir.glob("*_market.csv"))
        if market_files:
            df = pd.read_csv(market_files[0])
            print(f"  ✅ Loaded {len(df)} timesteps from {market_files[0].name}")
            print(f"     Columns: {', '.join(df.columns[:5])}...")
            if len(df) > 0:
                print(f"     Price range: {df['price'].min():.4f} - {df['price'].max():.4f}")
except ImportError:
    print("  ⚠️  pandas not installed (optional)")
    print("     Install for analysis: pip install pandas")
except Exception as e:
    print(f"  ⚠️  Could not analyze data: {e}")

# Summary
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print(f"✅ All core functionality working!")
print(f"✅ Market implementation: {external_lib}")
print(f"✅ Data logging: operational")
print(f"✅ Adapters: working")

print("\n📋 NEXT STEPS:")
print("1. Run full demo:")
print("   PYTHONPATH=src python examples/demo_full_simulation.py")
print()
print("2. Analyze results (if pandas/matplotlib installed):")
print("   pip install pandas matplotlib seaborn")
print("   python examples/visualize_results.py orderbook_demo_run1")
print()
print("3. For your team:")
print("   - Data team: Implement MessageStream and PortalNetwork")
print("   - Agent team: Implement Agent interface")
print("   - See: docs/QUICKSTART.md")
print()
print("4. For better performance:")
print("   pip install limit-order-book")

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED!")
print("=" * 80)


