“””
Source Node (Portal) Module for Prediction Market Simulation

This module defines the SourceNode class which represents information portals
in the network. Portals maintain a chronological feed of events and allow
agents to read and post information.
“””

from typing import List, Dict, Optional, Set

# Import Event from the sibling data module

import sys
from pathlib import Path

# Add parent directory to path to import from sibling module

parent_dir = Path(**file**).parent.parent
if str(parent_dir) not in sys.path:
sys.path.insert(0, str(parent_dir))

from data.data_module import Event

# Global time function - should be implemented by the main simulation

_current_time = 0

def get_current_time() -> int:
“””
Global utility to get the current simulation tick time.
This should be set by the main simulation loop.

```
Returns:
    Current timestep as integer
"""
return _current_time
```

def set_current_time(time: int):
“””
Set the global current time (used by simulation loop).

```
Args:
    time: Current timestep
"""
global _current_time
_current_time = time
```

class SourceNode:
“””
Represents a portal/source node in the information network.

```
Each source node maintains a chronological feed of events and tracks
when each event was posted to it. Agents can read from and post to
source nodes.

Attributes:
    node_id: Unique identifier for this source node
    event_feed: Chronologically ordered list of events posted to this node
    event_post_times: Maps event_id to the time it was posted to this node
    event_ids_posted: Set of event IDs already posted (for duplicate prevention)
"""

def __init__(self, node_id: str):
    """
    Initialize a source node.
    
    Args:
        node_id: Unique identifier for this portal
    """
    self.node_id = node_id
    self.event_feed: List[Event] = []
    self.event_post_times: Dict[str, int] = {}
    self.event_ids_posted: Set[str] = set()

def post_event(self, event: Event) -> bool:
    """
    Post an event to this source node's feed.
    Prevents duplicate posts of the same event.
    
    Args:
        event: Event object to post
    
    Returns:
        True if event was posted successfully, False if it was a duplicate
    """
    # Check if event has already been posted
    if event.event_id in self.event_ids_posted:
        return False
    
    # Get current simulation time
    current_time = get_current_time()
    
    # Add event to feed (maintains chronological order by posting time)
    self.event_feed.append(event)
    
    # Track when this event was posted to this node
    self.event_post_times[event.event_id] = current_time
    
    # Mark as posted
    self.event_ids_posted.add(event.event_id)
    
    return True

def get_events_since_index(self, last_read_index: int) -> List[Event]:
    """
    Get all events posted since the last read index.
    
    Args:
        last_read_index: The index of the last event the agent read
                        (0-indexed, -1 means hasn't read anything yet)
    
    Returns:
        List of Event objects from index (last_read_index + 1) to end
    """
    # If last_read_index is -1, return all events
    # Otherwise return events from (last_read_index + 1) onwards
    start_index = last_read_index + 1
    
    if start_index >= len(self.event_feed):
        return []
    
    return self.event_feed[start_index:]

def get_current_feed_length(self) -> int:
    """
    Get the current length of the event feed.
    Useful for agents to track their last read position.
    
    Returns:
        Number of events currently in the feed
    """
    return len(self.event_feed)

def get_latest_index(self) -> int:
    """
    Get the index of the most recent event.
    Returns -1 if no events have been posted.
    
    Returns:
        Index of latest event or -1 if feed is empty
    """
    return len(self.event_feed) - 1

def get_event_at_index(self, index: int) -> Optional[Event]:
    """
    Get a specific event by index.
    
    Args:
        index: Index in the event feed
    
    Returns:
        Event object if index is valid, None otherwise
    """
    if 0 <= index < len(self.event_feed):
        return self.event_feed[index]
    return None

def get_event_post_time(self, event_id: str) -> Optional[int]:
    """
    Get the time when a specific event was posted to this node.
    
    Args:
        event_id: Unique identifier of the event
    
    Returns:
        Timestamp when event was posted, or None if not posted here
    """
    return self.event_post_times.get(event_id)

def has_event(self, event_id: str) -> bool:
    """
    Check if an event has been posted to this node.
    
    Args:
        event_id: Unique identifier of the event
    
    Returns:
        True if event exists in this node's feed
    """
    return event_id in self.event_ids_posted

def get_all_events(self) -> List[Event]:
    """
    Get all events in the feed.
    
    Returns:
        Complete list of events in chronological posting order
    """
    return self.event_feed.copy()

def clear_feed(self):
    """
    Clear all events from the feed.
    Useful for testing or resetting the simulation.
    """
    self.event_feed.clear()
    self.event_post_times.clear()
    self.event_ids_posted.clear()

def __repr__(self) -> str:
    return f"SourceNode(id={self.node_id}, events={len(self.event_feed)})"

def get_feed_summary(self) -> Dict:
    """
    Get a summary of the current feed state.
    
    Returns:
        Dictionary with feed statistics
    """
    return {
        "node_id": self.node_id,
        "total_events": len(self.event_feed),
        "event_ids": list(self.event_ids_posted),
        "latest_post_time": max(self.event_post_times.values()) if self.event_post_times else None
    }
```

class SourceNodeManager:
“””
Manages multiple source nodes in the network.
Convenience class for the simulation to manage all portals.
“””

```
def __init__(self):
    """Initialize the source node manager."""
    self.nodes: Dict[str, SourceNode] = {}

def create_node(self, node_id: str) -> SourceNode:
    """
    Create a new source node.
    
    Args:
        node_id: Unique identifier for the node
    
    Returns:
        The created SourceNode object
    """
    if node_id in self.nodes:
        raise ValueError(f"Node {node_id} already exists")
    
    node = SourceNode(node_id)
    self.nodes[node_id] = node
    return node

def get_node(self, node_id: str) -> Optional[SourceNode]:
    """
    Get a source node by ID.
    
    Args:
        node_id: Identifier of the node
    
    Returns:
        SourceNode object or None if not found
    """
    return self.nodes.get(node_id)

def post_event_to_nodes(self, event: Event, node_ids: List[str]) -> Dict[str, bool]:
    """
    Post an event to multiple source nodes.
    
    Args:
        event: Event object to post
        node_ids: List of node IDs to post to
    
    Returns:
        Dictionary mapping node_id to success status
    """
    results = {}
    for node_id in node_ids:
        node = self.get_node(node_id)
        if node:
            results[node_id] = node.post_event(event)
        else:
            results[node_id] = False
    return results

def get_all_nodes(self) -> List[SourceNode]:
    """Get all source nodes."""
    return list(self.nodes.values())

def clear_all_nodes(self):
    """Clear all events from all nodes."""
    for node in self.nodes.values():
        node.clear_feed()
```

if **name** == “**main**”:
# Example usage and testing
from data.data_module import Event

```
# Set simulation time
set_current_time(100)

# Create source nodes
portal_a = SourceNode("portal_A")
portal_b = SourceNode("portal_B")

# Create sample events
event1 = Event(
    event_id="evt_001",
    initial_time=100,
    source_nodes=["portal_A", "portal_B"],
    tagline="Breaking News",
    description="Important event occurs"
)

event2 = Event(
    event_id="evt_002",
    initial_time=95,  # Older event posted later
    source_nodes=["portal_A"],
    tagline="Old News Resurfaces",
    description="Previously unknown information comes to light"
)

# Post events to portals
print(f"Posting event1 to portal_A: {portal_a.post_event(event1)}")
print(f"Posting event1 to portal_B: {portal_b.post_event(event1)}")

set_current_time(105)
print(f"\nPosting event2 to portal_A: {portal_a.post_event(event2)}")

# Try to post duplicate
print(f"Posting event1 again to portal_A: {portal_a.post_event(event1)}")

# Simulate agent reading
print(f"\nPortal A feed length: {portal_a.get_current_feed_length()}")
print(f"Portal A latest index: {portal_a.get_latest_index()}")

# Agent reads from beginning
new_events = portal_a.get_events_since_index(-1)
print(f"\nAgent reads all events (from index -1): {len(new_events)}")
for i, event in enumerate(new_events):
    print(f"  {i}: {event.tagline} (posted at t={portal_a.get_event_post_time(event.event_id)})")

# Agent reads only new events
last_read = 0
new_events = portal_a.get_events_since_index(last_read)
print(f"\nAgent reads events since index {last_read}: {len(new_events)}")
for event in new_events:
    print(f"  {event.tagline}")

# Test SourceNodeManager
print("\n--- Testing SourceNodeManager ---")
manager = SourceNodeManager()
manager.create_node("portal_C")
manager.create_node("portal_D")

event3 = Event(
    event_id="evt_003",
    initial_time=110,
    source_nodes=["portal_C", "portal_D"],
    tagline="Simultaneous Post",
    description="Event posted to multiple portals at once"
)

results = manager.post_event_to_nodes(event3, ["portal_C", "portal_D"])
print(f"Posting to multiple nodes: {results}")
```
