"""
Convert AgentSociety Yelp data into prediction market events.
Market: "What percentage of restaurants will receive 4+ stars?"

Usage:
    python data_processing/process_agentsociety_yelp.py \
        --data-dir /path/to/AgentSocietyChallenge/example/track1/yelp \
        --max-events 30 \
        --output data/yelp_agentsociety_market.json
"""
import json
import argparse
from pathlib import Path


def load_agentsociety_data(data_dir: Path):
    """Load and pair task/groundtruth files"""
    tasks_dir = data_dir / "tasks"
    gt_dir = data_dir / "groundtruth"

    if not tasks_dir.exists():
        raise FileNotFoundError(f"Tasks directory not found: {tasks_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"Groundtruth directory not found: {gt_dir}")

    pairs = []
    for task_file in sorted(tasks_dir.glob("task_*.json")):
        task_num = task_file.stem.split("_")[1]
        gt_file = gt_dir / f"groundtruth_{task_num}.json"
        if gt_file.exists():
            with open(task_file) as f:
                task = json.load(f)
            with open(gt_file) as f:
                gt = json.load(f)
            pairs.append((task, gt, int(task_num)))

    return sorted(pairs, key=lambda x: x[2])


def classify_sentiment(text: str) -> str:
    """Basic sentiment classification from review text"""
    positive_words = ['great', 'excellent', 'amazing', 'love', 'best', 'delicious',
                      'fantastic', 'wonderful', 'perfect', 'awesome', 'friendly',
                      'recommend', 'happy', 'impressed', 'outstanding']
    negative_words = ['terrible', 'awful', 'worst', 'horrible', 'bad', 'disappointing',
                      'poor', 'never', 'rude', 'slow', 'dirty', 'overpriced',
                      'waste', 'avoid', 'mediocre', 'underwhelming']

    text_lower = text.lower()
    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)

    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    return "mixed"


def determine_sources(text: str, stars: float) -> list:
    """Determine which channels this review appears on"""
    sources = ["news_feed"]

    # Extreme ratings go viral on social media
    if stars <= 1.5 or stars >= 4.5:
        sources.extend(["twitter", "reddit"])

    # Long detailed reviews get expert attention
    if len(text) > 400:
        sources.append("expert_analysis")

    # Controversial mixed reviews spark discussion
    if stars == 3.0 and len(text) > 300:
        sources.append("discord")

    return sources


def create_market_events(pairs: list, max_events: int = 30):
    """Convert pairs to prediction market events"""
    events = []

    for i, (task, gt, task_num) in enumerate(pairs[:max_events]):
        review_text = gt["review"]
        stars = gt["stars"]
        sentiment = classify_sentiment(review_text)
        sources = determine_sources(review_text, stars)

        # Truncate review for event description
        text_preview = review_text[:400] + "..." if len(review_text) > 400 else review_text

        event = {
            "event_id": f"yelp_{i:03d}",
            "initial_time": i,
            "source_nodes": sources,
            "tagline": f"New restaurant review posted ({sentiment} sentiment)",
            "description": f"A customer left the following review:\n\n\"{text_preview}\"",
            "metadata": {
                "user_id": task["user_id"],
                "item_id": task["item_id"],
                "actual_stars": stars,  # Hidden from agents during sim
                "is_success": stars >= 4.0,
                "review_length": len(review_text),
                "sentiment": sentiment
            }
        }
        events.append(event)

    return events


def calculate_ground_truth(pairs: list, threshold: float = 4.0):
    """Calculate actual success rate"""
    total = len(pairs)
    successes = sum(1 for _, gt, _ in pairs if gt["stars"] >= threshold)
    return successes / total if total > 0 else 0.0


def print_stats(pairs: list, events: list, ground_truth: float):
    """Print summary statistics"""
    print("\n" + "=" * 60)
    print("AGENTSOCIETY YELP DATA PROCESSING")
    print("=" * 60)

    # Star distribution
    stars_list = [gt["stars"] for _, gt, _ in pairs[:len(events)]]
    from collections import Counter
    star_counts = Counter(stars_list)

    print(f"\nProcessed {len(events)} reviews")
    print(f"\nStar Distribution:")
    for stars in sorted(star_counts.keys()):
        count = star_counts[stars]
        pct = count / len(events) * 100
        bar = "█" * int(pct / 2)
        print(f"  {stars} stars: {count:3d} ({pct:5.1f}%) {bar}")

    # Sentiment distribution
    sentiments = Counter(e["metadata"]["sentiment"] for e in events)
    print(f"\nSentiment Distribution:")
    for sent, count in sentiments.most_common():
        pct = count / len(events) * 100
        print(f"  {sent}: {count} ({pct:.1f}%)")

    print(f"\n{'=' * 60}")
    print(f"Ground Truth Success Rate (4+ stars): {ground_truth:.1%}")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert AgentSociety Yelp data into prediction market events"
    )
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to AgentSociety track1/yelp directory")
    parser.add_argument("--max-events", type=int, default=30,
                        help="Maximum number of events to generate")
    parser.add_argument("--output", type=str, default="data/yelp_agentsociety_market.json",
                        help="Output JSON file path")
    parser.add_argument("--threshold", type=float, default=4.0,
                        help="Star threshold for 'success' (default: 4.0)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load data
    print(f"Loading data from: {data_dir}")
    pairs = load_agentsociety_data(data_dir)
    print(f"Found {len(pairs)} task/groundtruth pairs")

    # Create events
    events = create_market_events(pairs, args.max_events)

    # Calculate ground truth for the events we're using
    ground_truth = calculate_ground_truth(pairs[:args.max_events], args.threshold)

    # Print statistics
    print_stats(pairs, events, ground_truth)

    # Create output structure
    output = {
        "market_type": "restaurant_success_rate",
        "market_question": f"What percentage of reviewed restaurants will receive {args.threshold}+ star ratings?",
        "ground_truth": {
            "success_rate": ground_truth,
            "threshold": args.threshold,
            "total_reviews": len(events),
            "successful_reviews": sum(1 for e in events if e["metadata"]["is_success"])
        },
        "events": events
    }

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {len(events)} events to: {output_path}")
    print(f"\nTo run the simulation:")
    print(f"  python examples/run_with_synthetic_data.py --events {args.output}")
    print(f"\nTo evaluate results:")
    print(f"  python examples/evaluate_simulation.py --run-name prediction_sim --actual-outcome {ground_truth:.3f} --plot")


if __name__ == "__main__":
    main()
