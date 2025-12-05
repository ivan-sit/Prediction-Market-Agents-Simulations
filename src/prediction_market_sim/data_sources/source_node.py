"""
Source Node (Portal) Module for Prediction Market Simulation

This module defines the SourceNode class which represents information portals
in the network. Portals maintain a chronological feed of events and allow
agents to read and post information.
"""

from typing import List, Dict, Optional, Set

import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from ..data.data_module import Event

_current_time = 0

def get_current_time() -> int:
    """
    Global utility to get the current simulation tick time.
    This should be set by the main simulation loop.

    Returns:
        Current timestep as integer
    """
    return _current_time


def set_current_time(time: int):
    """
    Set the global current time (used by simulation loop).

    Args:
        time: Current timestep
    """
    global _current_time
    _current_time = time


class SourceNode:
    """
    Represents a portal/source node in the information network.

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

    def get_recent_events(self, limit: int = 10) -> List[Dict[str, object]]:
        """Return the most recent events as dictionaries with metadata."""
        recent = self.event_feed[-limit:]
        items = []
        for ev in recent:
            items.append(
                {
                    "event_id": ev.event_id,
                    "tagline": ev.tagline,
                    "description": ev.description,
                    "timestamp": self.event_post_times.get(ev.event_id, None),
                }
            )
        return items

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
        """Return string representation of the SourceNode."""
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


class SourceNodeManager:
    """
    Manages multiple source nodes in the network.
    Convenience class for the simulation to manage all portals.
    """

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