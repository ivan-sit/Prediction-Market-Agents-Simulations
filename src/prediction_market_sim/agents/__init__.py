"""Agent implementations live here (belief updates + betting policies).

Each teammate can add their own module without touching the simulator core.
"""

from .prediction_market_agent import (
    PredictionMarketAgent,
    MarketPlanningModule,
    MarketReasoningModule,
    MarketMemoryModule,
    PlaceholderMarketTools,
)
from .ollama_llm import OllamaLLM, OllamaEmbeddings
from .adapters import PredictionMarketAgentAdapter, create_prediction_agent

__all__ = [
    "PredictionMarketAgent",
    "MarketPlanningModule",
    "MarketReasoningModule",
    "MarketMemoryModule",
    "PlaceholderMarketTools",
    "OllamaLLM",
    "OllamaEmbeddings",
    "PredictionMarketAgentAdapter",
    "create_prediction_agent",
]
