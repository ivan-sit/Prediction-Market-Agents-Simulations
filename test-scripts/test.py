from typing import Dict, Any, List
import sys
import os

# Import the production PredictionMarketAgent and OllamaLLM
from prediction_market_agent import PredictionMarketAgent
from ollama_llm import OllamaLLM


class MockPortalNetwork:
    """Mock portal network for testing cross-posting"""
    def __init__(self):
        self.posted_messages = []

    def ingest_agent_feedback(self, target_source: str, payload: Dict[str, Any]):
        self.posted_messages.append({
            'target': target_source,
            'payload': payload
        })


def main():
    """Test the production PredictionMarketAgent with cross-posting functionality"""
    print("="*70)
    print("PREDICTION MARKET AGENT - CROSS-POSTING TEST")
    print("="*70)

    print("\n1. Creating Ollama LLM...")
    llm = OllamaLLM(model="llama3.1:8b")
    print("✓ LLM created")

    print("\n2. Creating Mock Portal Network...")
    portal = MockPortalNetwork()
    print("✓ Portal created")

    print("\n3. Creating PredictionMarketAgent with HIGH cross-post probability (50%)...")
    agent = PredictionMarketAgent(
        llm=llm,
        personality_prompt="Conservative trader, risk-averse",
        initial_bankroll=10000.0,
        subscribed_sources=["twitter"],
        cross_post_probability=0.5
    )
    agent.set_portal_network(portal)
    print(f"✓ Agent created with ${agent.get_bankroll():.2f} bankroll")

    events = [
        {"event_id": "EVENT_001", "description": "Will Bitcoin reach $100,000 by end of 2025?", "source": "twitter"},
        {"event_id": "EVENT_002", "description": "Will Apple release AR glasses in 2025?", "source": "reddit"},
        {"event_id": "EVENT_003", "description": "Will the US economy enter recession in 2025?", "source": "news"},
        {"event_id": "EVENT_004", "description": "Will SpaceX land on Mars by 2027?", "source": "discord"},
        {"event_id": "EVENT_005", "description": "Will Tesla stock double in 2025?", "source": "telegram"},
    ]

    print(f"\n4. Processing {len(events)} events...\n")

    results = []
    for event in events:
        print(f"Event {event['event_id']} from {event['source']}: {event['description']}")

        agent.insert_event(event)
        result = agent.workflow()
        results.append(result)

        print(f"  → Decision: {result['decision']}")
        print(f"  → Amount: ${result['amount']:.2f}")
        print(f"  → Confidence: {result['confidence']:.2f}")
        print(f"  → Bankroll: ${result['bankroll']:.2f}\n")

    print("="*70)
    print("CROSS-POSTING SUMMARY")
    print("="*70)

    cross_posts = agent.get_cross_post_history()
    print(f"\nTotal cross-posts: {len(cross_posts)}")

    if cross_posts:
        print("\nCross-post details:")
        for i, cp in enumerate(cross_posts, 1):
            print(f"\n{i}. Event: {cp['event_id']}")
            print(f"   Route: {cp['from_source']} → {cp['to_source']}")
            print(f"   Signal: {cp['payload']['signal']}")
            print(f"   Confidence: {cp['confidence']:.2f}")

    print(f"\n\nPortal network received {len(portal.posted_messages)} messages:")
    for i, msg in enumerate(portal.posted_messages, 1):
        print(f"{i}. Target: {msg['target']}, Signal: {msg['payload']['signal']}, Event: {msg['payload']['event_id']}")

    print("\n" + "="*70)
    print("AGENT PERFORMANCE")
    print("="*70)

    summary = agent.get_performance_summary()
    events_that_cross_posted = len(set(cp['event_id'] for cp in cross_posts))
    print(f"\nInitial Bankroll:  ${summary['initial_bankroll']:.2f}")
    print(f"Final Bankroll:    ${summary['current_bankroll']:.2f}")
    print(f"Profit/Loss:       ${summary['profit_loss']:.2f} ({summary['profit_loss_pct']:.1f}%)")
    print(f"Total Trades:      {summary['total_trades']}")
    print(f"Events Cross-posted: {events_that_cross_posted} ({(events_that_cross_posted/summary['total_trades']*100):.1f}%)")
    print(f"Total Cross-post Messages: {summary['cross_posts']}")

    print("\n✅ TEST COMPLETE")


if __name__ == "__main__":
    main()
