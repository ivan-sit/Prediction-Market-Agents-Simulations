#!/usr/bin/env python3
"""Test that the agents module is properly configured and usable."""

from src.prediction_market_sim.agents import (
    PredictionMarketAgent,
    OllamaLLM,
    MarketPlanningModule,
    MarketMemoryModule,
    MarketReasoningModule,
    PlaceholderMarketTools,
    OllamaEmbeddings,
)

print("✅ All imports successful!")

# Test that classes are accessible
print(f"✅ PredictionMarketAgent: {PredictionMarketAgent.__name__}")
print(f"✅ OllamaLLM: {OllamaLLM.__name__}")
print(f"✅ MarketPlanningModule: {MarketPlanningModule.__name__}")
print(f"✅ MarketMemoryModule: {MarketMemoryModule.__name__}")
print(f"✅ MarketReasoningModule: {MarketReasoningModule.__name__}")
print(f"✅ PlaceholderMarketTools: {PlaceholderMarketTools.__name__}")
print(f"✅ OllamaEmbeddings: {OllamaEmbeddings.__name__}")

print("\n🎉 The agents module is now a usable Python module!")
