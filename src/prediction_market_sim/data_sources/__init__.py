"""Data synthesis and portal/network components belong to this package."""
"""
Data Sources Module

This module provides the SourceNode and SourceNodeManager classes for managing
information portals in the prediction market simulation network.
"""

from .source_node import (
    SourceNode,
    SourceNodeManager,
    get_current_time,
    set_current_time
)
from .adapters import (
    EventDatabaseStream,
    SourceNodeNetworkAdapter,
    create_event_stream,
    create_portal_network
)

__all__ = [
    'SourceNode',
    'SourceNodeManager',
    'get_current_time',
    'set_current_time',
    'EventDatabaseStream',
    'SourceNodeNetworkAdapter',
    'create_event_stream',
    'create_portal_network',
]
