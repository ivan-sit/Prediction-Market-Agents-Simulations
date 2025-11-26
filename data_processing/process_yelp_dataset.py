"""
Process Yelp dataset into prediction market events format.

This script converts Yelp review data into time-ordered market events
for a specific restaurant prediction market.

Usage:
    python data_processing/process_yelp_dataset.py \
        --yelp-dir /path/to/yelp_dataset \
        --business-id <business_id> \
        --output data/yelp_restaurant_market.json
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import sys

def load_yelp_business(yelp_dir: Path, business_id: str) -> Dict[str, Any]:
    """Load specific business information from Yelp dataset."""
    business_file = yelp_dir / "yelp_academic_dataset_business.json"

    print(f"[INFO] Loading business data from {business_file}")

    if not business_file.exists():
        raise FileNotFoundError(f"Business file not found: {business_file}")

    # Yelp dataset is line-delimited JSON
    with open(business_file, 'r', encoding='utf-8') as f:
        for line in f:
            business = json.loads(line.strip())
            if business['business_id'] == business_id:
                print(f"[OK] Found business: {business['name']}")
                print(f"      Stars: {business['stars']}")
                print(f"      Review Count: {business['review_count']}")
                print(f"      Categories: {business.get('categories', 'N/A')}")
                return business

    raise ValueError(f"Business ID {business_id} not found in dataset")


def load_yelp_reviews(yelp_dir: Path, business_id: str) -> List[Dict[str, Any]]:
    """Load all reviews for a specific business, sorted by date."""
    review_file = yelp_dir / "yelp_academic_dataset_review.json"

    print(f"[INFO] Loading reviews from {review_file}")

    if not review_file.exists():
        raise FileNotFoundError(f"Review file not found: {review_file}")

    reviews = []
    line_count = 0

    # Yelp dataset is line-delimited JSON (can be very large)
    with open(review_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_count += 1
            if line_count % 100000 == 0:
                print(f"      Processed {line_count} reviews, found {len(reviews)} for this business...")

            review = json.loads(line.strip())
            if review['business_id'] == business_id:
                reviews.append(review)

    print(f"[OK] Found {len(reviews)} reviews for business {business_id}")

    # Sort by date
    reviews.sort(key=lambda r: r['date'])

    return reviews


def load_yelp_user(yelp_dir: Path, user_id: str, user_cache: Dict[str, Dict]) -> Dict[str, Any]:
    """Load user information (cached for efficiency)."""
    if user_id in user_cache:
        return user_cache[user_id]

    user_file = yelp_dir / "yelp_academic_dataset_user.json"

    with open(user_file, 'r', encoding='utf-8') as f:
        for line in f:
            user = json.loads(line.strip())
            if user['user_id'] == user_id:
                user_cache[user_id] = user
                return user

    # Return minimal user if not found
    return {'user_id': user_id, 'name': 'Unknown', 'elite': []}


def classify_review_sentiment(stars: int, text: str) -> str:
    """Classify review sentiment based on stars and text."""
    if stars >= 4:
        return "positive"
    elif stars <= 2:
        return "negative"
    else:
        return "mixed"


def determine_source_nodes(stars: int, is_elite: bool, useful_votes: int) -> List[str]:
    """Determine which information channels this review would appear on."""
    sources = []

    # All reviews appear on news feed
    sources.append("news_feed")

    # Elite reviews and highly useful reviews appear on expert analysis
    if is_elite or useful_votes >= 10:
        sources.append("expert_analysis")

    # Very positive or very negative reviews appear on social media
    if stars == 5 or stars == 1:
        sources.append("twitter")
        sources.append("reddit")

    # Detailed reviews (long text) appear on expert channels
    if len(text) > 500:
        sources.append("expert_analysis")

    return sources


def convert_review_to_event(review: Dict[str, Any],
                            user_info: Dict[str, Any],
                            event_index: int,
                            business_name: str) -> Dict[str, Any]:
    """Convert a Yelp review into a prediction market event."""

    stars = review['stars']
    text = review['text']
    useful = review.get('useful', 0)
    funny = review.get('funny', 0)
    cool = review.get('cool', 0)
    date = review['date']

    # Check if user is elite
    user_elite = user_info.get('elite', [])
    is_elite = len(user_elite) > 0
    user_name = user_info.get('name', 'Anonymous')
    user_review_count = user_info.get('review_count', 0)

    # Determine source nodes
    sources = determine_source_nodes(stars, is_elite, useful)

    # Create tagline
    sentiment = classify_review_sentiment(stars, text)
    elite_tag = " [ELITE REVIEWER]" if is_elite else ""

    if stars == 5:
        tagline = f"{user_name}{elite_tag} gives {business_name} glowing 5-star review"
    elif stars == 4:
        tagline = f"{user_name}{elite_tag} posts positive 4-star review of {business_name}"
    elif stars == 3:
        tagline = f"{user_name} leaves mixed 3-star review of {business_name}"
    elif stars == 2:
        tagline = f"{user_name} posts disappointing 2-star review of {business_name}"
    else:  # stars == 1
        tagline = f"{user_name}{elite_tag} slams {business_name} with scathing 1-star review"

    # Create detailed description
    # Truncate review text to first 300 chars for event description
    text_preview = text[:300] + "..." if len(text) > 300 else text

    description = f"Posted on {date}. "

    if is_elite:
        description += f"Elite Yelp user {user_name} (with {user_review_count} total reviews) "
    else:
        description += f"User {user_name} "

    description += f"rated {business_name} {stars} out of 5 stars. "

    if useful > 0:
        description += f"Review received {useful} useful votes"
        if funny > 0 or cool > 0:
            description += f", {funny} funny votes, {cool} cool votes. "
        else:
            description += ". "

    description += f"\n\nReview excerpt: \"{text_preview}\""

    # Create event
    event = {
        "event_id": f"yelp_{event_index:03d}",
        "initial_time": event_index,
        "source_nodes": sources,
        "tagline": tagline,
        "description": description,
        "metadata": {
            "stars": stars,
            "useful_votes": useful,
            "funny_votes": funny,
            "cool_votes": cool,
            "is_elite_reviewer": is_elite,
            "reviewer_total_reviews": user_review_count,
            "review_date": date,
            "sentiment": sentiment,
            "review_length": len(text)
        }
    }

    return event


def calculate_outcome(reviews: List[Dict[str, Any]], business: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate the ground truth outcome for the prediction market."""

    # Calculate average rating from reviews
    total_stars = sum(r['stars'] for r in reviews)
    avg_stars = total_stars / len(reviews) if reviews else 0

    # Use business final rating as ground truth
    final_stars = business['stars']
    final_review_count = business['review_count']

    # Calculate rating trajectory
    if len(reviews) >= 10:
        first_10_avg = sum(r['stars'] for r in reviews[:10]) / 10
        last_10_avg = sum(r['stars'] for r in reviews[-10:]) / 10
        trend = "improving" if last_10_avg > first_10_avg else "declining"
    else:
        trend = "stable"

    outcome = {
        "business_id": business['business_id'],
        "business_name": business['name'],
        "final_average_stars": final_stars,
        "total_reviews": final_review_count,
        "rating_trend": trend,
        "outcome_4_5_plus": final_stars >= 4.5,  # Binary market: Will it be 4.5+ stars?
        "categories": business.get('categories', ''),
        "city": business.get('city', ''),
        "state": business.get('state', '')
    }

    return outcome


def process_yelp_to_market_events(yelp_dir: Path,
                                  business_id: str,
                                  max_reviews: int = None) -> Dict[str, Any]:
    """
    Main processing function to convert Yelp data to prediction market format.

    Args:
        yelp_dir: Directory containing Yelp dataset files
        business_id: Yelp business_id to process
        max_reviews: Optional limit on number of reviews to process

    Returns:
        Dictionary with events and ground truth
    """

    # Load business info
    business = load_yelp_business(yelp_dir, business_id)

    # Load reviews
    reviews = load_yelp_reviews(yelp_dir, business_id)

    if len(reviews) == 0:
        raise ValueError(f"No reviews found for business {business_id}")

    # Optionally limit number of reviews
    if max_reviews and len(reviews) > max_reviews:
        print(f"[INFO] Limiting to first {max_reviews} reviews (out of {len(reviews)})")
        reviews = reviews[:max_reviews]

    # Load user info for each review (cached)
    print(f"[INFO] Loading user information for {len(reviews)} reviewers...")
    user_cache = {}

    # Convert reviews to events
    print(f"[INFO] Converting reviews to market events...")
    events = []

    for i, review in enumerate(reviews):
        if i % 50 == 0 and i > 0:
            print(f"      Processed {i}/{len(reviews)} reviews...")

        user_info = load_yelp_user(yelp_dir, review['user_id'], user_cache)
        event = convert_review_to_event(review, user_info, i, business['name'])
        events.append(event)

    print(f"[OK] Created {len(events)} market events")

    # Calculate outcome
    outcome = calculate_outcome(reviews, business)

    # Create final output
    market_data = {
        "market_type": "restaurant_success",
        "market_question": f"Will {business['name']} maintain a rating of 4.5+ stars?",
        "ground_truth": outcome,
        "events": events,
        "metadata": {
            "data_source": "Yelp Academic Dataset",
            "business_id": business_id,
            "business_name": business['name'],
            "total_events": len(events),
            "date_range": f"{reviews[0]['date']} to {reviews[-1]['date']}"
        }
    }

    return market_data


def find_interesting_restaurants(yelp_dir: Path, min_reviews: int = 100, limit: int = 10):
    """
    Helper function to find interesting restaurants with many reviews.
    Use this to discover good business_ids for prediction markets.
    """
    business_file = yelp_dir / "yelp_academic_dataset_business.json"

    print(f"[INFO] Searching for restaurants with {min_reviews}+ reviews...")

    candidates = []

    with open(business_file, 'r', encoding='utf-8') as f:
        for line in f:
            business = json.loads(line.strip())

            # Filter for restaurants only
            categories = business.get('categories', '')
            if not categories or 'Restaurant' not in categories:
                continue

            review_count = business.get('review_count', 0)
            if review_count >= min_reviews:
                candidates.append({
                    'business_id': business['business_id'],
                    'name': business['name'],
                    'stars': business['stars'],
                    'review_count': review_count,
                    'city': business.get('city', ''),
                    'state': business.get('state', ''),
                    'categories': categories
                })

    # Sort by review count
    candidates.sort(key=lambda x: x['review_count'], reverse=True)

    print(f"\n[OK] Found {len(candidates)} restaurants with {min_reviews}+ reviews")
    print(f"\nTop {limit} candidates for prediction markets:\n")

    for i, biz in enumerate(candidates[:limit], 1):
        print(f"{i}. {biz['name']}")
        print(f"   ID: {biz['business_id']}")
        print(f"   Location: {biz['city']}, {biz['state']}")
        print(f"   Stars: {biz['stars']}")
        print(f"   Reviews: {biz['review_count']}")
        print(f"   Categories: {biz['categories'][:80]}...")
        print()

    return candidates[:limit]


def main():
    parser = argparse.ArgumentParser(
        description='Convert Yelp dataset to prediction market events'
    )
    parser.add_argument('--yelp-dir', type=str, required=True,
                        help='Directory containing Yelp dataset files')
    parser.add_argument('--business-id', type=str,
                        help='Specific business ID to process')
    parser.add_argument('--find-restaurants', action='store_true',
                        help='Find interesting restaurants with many reviews')
    parser.add_argument('--min-reviews', type=int, default=100,
                        help='Minimum number of reviews (for --find-restaurants)')
    parser.add_argument('--max-reviews', type=int,
                        help='Maximum number of reviews to process')
    parser.add_argument('--output', type=str, default='data/yelp_restaurant_market.json',
                        help='Output JSON file')

    args = parser.parse_args()

    yelp_dir = Path(args.yelp_dir)

    if not yelp_dir.exists():
        print(f"[ERROR] Yelp directory not found: {yelp_dir}")
        sys.exit(1)

    # Mode 1: Find interesting restaurants
    if args.find_restaurants:
        find_interesting_restaurants(yelp_dir, args.min_reviews)
        return

    # Mode 2: Process specific restaurant
    if not args.business_id:
        print("[ERROR] Must provide --business-id or use --find-restaurants")
        sys.exit(1)

    print("=" * 70)
    print("YELP TO PREDICTION MARKET CONVERTER")
    print("=" * 70)

    try:
        # Process Yelp data
        market_data = process_yelp_to_market_events(
            yelp_dir,
            args.business_id,
            max_reviews=args.max_reviews
        )

        # Save output
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(market_data, f, indent=2, ensure_ascii=False)

        print(f"\n[OK] Saved prediction market data to {output_path}")
        print(f"\nMarket Question: {market_data['market_question']}")
        print(f"Ground Truth: {market_data['ground_truth']['outcome_4_5_plus']}")
        print(f"Total Events: {len(market_data['events'])}")

    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
