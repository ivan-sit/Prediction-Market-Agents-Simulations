"""
Data Module for Prediction Market Simulation

This module manages the central event database stored as JSON.
Events are stored in chronological order by their initial timestamp.
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path

class Event:
    """
Represents a single event in the prediction market.


Attributes:
    event_id: Unique identifier for the event
    initial_time: Timestamp when event first hits a data source
    source_nodes: List of portal IDs where event gets posted (chronologically ordered)
    tagline: Brief description/headline of the event
    description: Detailed description of the event
"""

    def __init__(self, event_id: str, initial_time: int, source_nodes: List[str],
                 tagline: str, description: str):
        self.event_id = event_id
        self.initial_time = initial_time
        self.source_nodes = source_nodes  # Already sorted by posting order
        self.tagline = tagline
        self.description = description

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary format for JSON serialization."""
        return {
            "event_id": self.event_id,
            "initial_time": self.initial_time,
            "source_nodes": self.source_nodes,
            "tagline": self.tagline,
            "description": self.description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create an Event object from a dictionary."""
        return cls(
            event_id=data["event_id"],
            initial_time=data["initial_time"],
            source_nodes=data["source_nodes"],
            tagline=data["tagline"],
            description=data["description"]
        )

    def __repr__(self) -> str:
        return f"Event(id={self.event_id}, time={self.initial_time}, tagline='{self.tagline}')"


class EventDatabase:
    """
    Manages the central event database stored in a JSON file.
    Events are assumed to be sorted by initial_time in the file.
    """

    def __init__(self, db_path: str = "events_database.json"):
        """
        Initialize the event database manager.

        Args:
            db_path: Path to the JSON file containing events
        """
        self.db_path = Path(db_path)
        self._ensure_db_exists()

    def _ensure_db_exists(self):
        """Create the database file if it doesn't exist."""
        if not self.db_path.exists():
            with open(self.db_path, 'w') as f:
                json.dump({"events": []}, f)

    def load_all_events(self) -> List[Event]:
        """
        Load all events from the database.

        Returns:
            List of Event objects sorted by initial_time
        """
        with open(self.db_path, 'r') as f:
            data = json.load(f)

        return [Event.from_dict(event_dict) for event_dict in data.get("events", [])]

    def get_events_at_timestep(self, current_time: int) -> List[Event]:
        """
        Get all events that should be released at the current timestep.
        This function also removes these events from the database file as an optimization.

        Args:
            current_time: The current simulation timestep

        Returns:
            List of Event objects with initial_time == current_time
        """
        with open(self.db_path, 'r') as f:
            data = json.load(f)

        events = data.get("events", [])
        current_events = []
        remaining_events = []

        # Since events are sorted by time, we can optimize by breaking early
        for event_dict in events:
            if event_dict["initial_time"] == current_time:
                current_events.append(Event.from_dict(event_dict))
            elif event_dict["initial_time"] > current_time:
                # All remaining events are in the future
                remaining_events.append(event_dict)
                remaining_events.extend(events[events.index(event_dict) + 1:])
                break
            # Events with initial_time < current_time are skipped (already processed)

        # Write back the remaining events (optimization: remove processed ones)
        with open(self.db_path, 'w') as f:
            json.dump({"events": remaining_events}, f, indent=2)

        return current_events

    def add_events(self, events: List[Event]):
        """
        Add new events to the database (useful for initialization or testing).
        Maintains sorted order by initial_time.

        Args:
            events: List of Event objects to add
        """
        existing_events = self.load_all_events()
        existing_events.extend(events)

        # Sort by initial_time to maintain database invariant
        existing_events.sort(key=lambda e: e.initial_time)

        with open(self.db_path, 'w') as f:
            json.dump({
                "events": [e.to_dict() for e in existing_events]
            }, f, indent=2)

    def save_events(self, events: List[Event]):
        """
        Replace all events in the database with the provided list.
        Useful for initialization.

        Args:
            events: List of Event objects to save
        """
        # Sort by initial_time
        events.sort(key=lambda e: e.initial_time)

        with open(self.db_path, 'w') as f:
            json.dump({
                "events": [e.to_dict() for e in events]
            }, f, indent=2)

    def clear_database(self):
        """Clear all events from the database."""
        with open(self.db_path, 'w') as f:
            json.dump({"events": []}, f)

    def get_event_count(self) -> int:
        """Get the total number of events remaining in the database."""
        with open(self.db_path, 'r') as f:
            data = json.load(f)
        return len(data.get("events", []))


# Global function to be used by other modules

def get_events_for_current_timestep(current_time: int, db_path: str = "events_database.json") -> List[Event]:
    """
    Convenience function to get events at current timestep.

    Args:
        current_time: Current simulation timestep
        db_path: Path to the database file

    Returns:
        List of Event objects for the current timestep
    """
    db = EventDatabase(db_path)
    return db.get_events_at_timestep(current_time)


if __name__ == "__main__":
    # Example usage and testing
    db = EventDatabase("test_events.json")

    # Create sample events
    sample_events = [
        Event(
            event_id="evt_001",
            initial_time=10,
            source_nodes=["portal_A", "portal_B", "portal_C"],
            tagline="Tech Company Announces New Product",
            description="Major tech company unveils revolutionary AI assistant"
        ),
        Event(
            event_id="evt_002",
            initial_time=10,
            source_nodes=["portal_B"],
            tagline="Market Volatility Increases",
            description="Stock market experiences unusual trading patterns"
        ),
        Event(
            event_id="evt_003",
            initial_time=15,
            source_nodes=["portal_A", "portal_D"],
            tagline="Economic Report Released",
            description="Quarterly GDP numbers exceed expectations"
        ),
    ]

    # Save events to database
    db.save_events(sample_events)
    print(f"Saved {len(sample_events)} events to database")

    # Retrieve events at timestep 10
    events_at_10 = db.get_events_at_timestep(10)
    print(f"\nEvents at timestep 10: {len(events_at_10)}")
    for event in events_at_10:
        print(f"  {event}")

    # Check remaining events
    print(f"\nRemaining events in database: {db.get_event_count()}")

