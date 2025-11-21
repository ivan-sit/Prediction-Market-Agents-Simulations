"""
Adapters to connect data/data_sources modules to simulation engine protocols.

This bridges the gap between your colleagues' implementations and the
simulation engine's expected interfaces.
"""

from typing import List, Mapping, Iterable
from pathlib import Path

from ..data.data_module import EventDatabase, Event, get_events_for_current_timestep
from ..data_sources.source_node import SourceNodeManager, SourceNode, set_current_time
from ..simulation.interfaces import MessageStream, PortalNetwork


class EventDatabaseStream(MessageStream):
    """
    Adapter that wraps EventDatabase to provide MessageStream interface.
    
    Reads events from JSON database and serves them at appropriate timesteps.
    """
    
    def __init__(self, db_path: str = "events_database.json"):
        self.db = EventDatabase(db_path=db_path)
        self.current_timestep = 0
        self._finished = False
        
    @property
    def finished(self) -> bool:
        """Stream is finished when no more events remain."""
        return self._finished
    
    def bootstrap(self, *, seed: int | None = None) -> None:
        """Initialize the stream (EventDatabase doesn't need seeding)."""
        self.current_timestep = 0
        self._finished = False
        
    def next_batch(self) -> List[Mapping[str, object]]:
        """
        Get all events for the current timestep and advance.
        
        Returns:
            List of event dictionaries ready for portal routing
        """
        # Get events at current timestep
        events = self.db.get_events_at_timestep(self.current_timestep)
        
        # Convert Event objects to message dictionaries
        messages = []
        for event in events:
            messages.append({
                'event_id': event.event_id,
                'timestamp': event.initial_time,
                'source_nodes': event.source_nodes,
                'tagline': event.tagline,
                'description': event.description,
                'type': 'event'  # Add type field for portal routing
            })
        
        # Check if finished (no more events in database)
        # Only mark finished when database is actually empty
        if self.db.get_event_count() == 0:
            self._finished = True
        
        self.current_timestep += 1
        
        return messages


class SourceNodeNetworkAdapter(PortalNetwork):
    """
    Adapter that wraps SourceNodeManager to provide PortalNetwork interface.
    
    Routes messages from EventDatabase to agent inboxes via source nodes.
    """
    
    def __init__(self, manager: SourceNodeManager | None = None):
        """
        Initialize the portal network adapter.
        
        Args:
            manager: Optional pre-configured SourceNodeManager. 
                    If None, creates empty manager (nodes added separately)
        """
        self.manager = manager if manager is not None else SourceNodeManager()
        self.agent_subscriptions: dict[str, list[str]] = {}  # agent_id -> [node_ids]
        
    def add_node(self, node: SourceNode) -> None:
        """Convenience method to add a source node to the network."""
        self.manager.add_node(node)
        
    def subscribe_agent(self, agent_id: str, node_ids: list[str]) -> None:
        """
        Subscribe an agent to specific source nodes.
        
        Args:
            agent_id: Unique agent identifier
            node_ids: List of source node IDs to subscribe to
        """
        self.agent_subscriptions[agent_id] = node_ids
        
    def route(self, messages: Iterable[Mapping[str, object]]) -> Mapping[str, List[dict]]:
        """
        Route messages to agent inboxes based on source node subscriptions.
        
        Args:
            messages: Event messages from the stream
            
        Returns:
            Mapping of agent_id -> list of messages for that agent
        """
        # Post messages to appropriate source nodes
        for msg in messages:
            # Each event specifies which source_nodes it should appear on
            source_nodes = msg.get('source_nodes', [])
            
            for node_id in source_nodes:
                node = self.manager.get_node(node_id)
                if node:
                    # Post event to this source node - create Event object
                    event = Event(
                        event_id=msg['event_id'],
                        initial_time=msg['timestamp'],
                        source_nodes=source_nodes,  # List of portals
                        tagline=msg['tagline'],
                        description=msg['description']
                    )
                    node.post_event(event)
        
        # Now route to agents based on subscriptions
        agent_inboxes: dict[str, list[dict]] = {}
        
        for agent_id, subscribed_nodes in self.agent_subscriptions.items():
            inbox = []
            
            # Read from each subscribed node
            for node_id in subscribed_nodes:
                node = self.manager.get_node(node_id)
                if node:
                    # Get all events from this node's feed
                    events = node.get_all_events()

                    for event in events:
                        inbox.append({
                            'source_node': node_id,
                            'event_id': event.event_id,
                            'tagline': event.tagline,
                            'description': event.description,
                            'timestamp': event.initial_time
                        })
            
            agent_inboxes[agent_id] = inbox
        
        return agent_inboxes
    
    def ingest_agent_feedback(self, agent_id: str, payload: Mapping[str, object]) -> None:
        """
        Handle agent feedback (e.g., agent posts to a portal).
        
        Args:
            agent_id: Agent making the post
            payload: Content to post (could contain commentary, analysis, etc.)
        """
        target_node = payload.get("target_node")
        content = payload.get("content")
        tagline = payload.get("tagline", "Agent post")
        if not target_node or not content:
            return
        node = self.manager.get_node(target_node)
        if not node:
            return

        # Create a lightweight Event from the post
        event = Event(
            event_id=payload.get("event_id", f"agent_post_{agent_id}"),
            initial_time=payload.get("timestamp", 0),
            source_nodes=[target_node],
            tagline=tagline,
            description=str(content),
        )
        node.post_event(event)


def create_event_stream(db_path: str = "events_database.json") -> EventDatabaseStream:
    """
    Factory function to create an EventDatabase message stream.
    
    Args:
        db_path: Path to the events JSON database
        
    Returns:
        MessageStream adapter wrapping EventDatabase
    """
    return EventDatabaseStream(db_path=db_path)


def create_portal_network(node_configs: list[dict] | None = None) -> SourceNodeNetworkAdapter:
    """
    Factory function to create a source node network.
    
    Args:
        node_configs: Optional list of node configurations like:
            [{'node_id': 'twitter', 'reliability': 0.8}, ...]
            
    Returns:
        PortalNetwork adapter wrapping SourceNodeManager
    """
    manager = SourceNodeManager()
    
    if node_configs:
        for config in node_configs:
            # Note: reliability is ignored as SourceNode doesn't support it yet
            manager.create_node(config['node_id'])
    
    return SourceNodeNetworkAdapter(manager=manager)
