#!/usr/bin/env python3
"""Test that the agents module is properly configured and usable."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from prediction_market_sim.agents import (
    PredictionMarketAgent,
    OllamaLLM,
    MarketPlanningModule,
    MarketMemoryModule,
    MarketReasoningModule,
    OllamaEmbeddings,
)

print("All imports successful!")

# Test that classes are accessible
print(f"PredictionMarketAgent: {PredictionMarketAgent.__name__}")
print(f"OllamaLLM: {OllamaLLM.__name__}")
print(f"MarketPlanningModule: {MarketPlanningModule.__name__}")
print(f"MarketMemoryModule: {MarketMemoryModule.__name__}")
print(f"MarketReasoningModule: {MarketReasoningModule.__name__}")
print(f"MarketReasoningModule: {MarketReasoningModule.__name__}")
print(f"OllamaEmbeddings: {OllamaEmbeddings.__name__}")

print("\nThe agents module is now a usable Python module!")
