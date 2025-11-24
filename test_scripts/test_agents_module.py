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

print("[OK] All imports successful!")

# Test that classes are accessible
print(f"[OK] PredictionMarketAgent: {PredictionMarketAgent.__name__}")
print(f"[OK] OllamaLLM: {OllamaLLM.__name__}")
print(f"[OK] MarketPlanningModule: {MarketPlanningModule.__name__}")
print(f"[OK] MarketMemoryModule: {MarketMemoryModule.__name__}")
print(f"[OK] MarketReasoningModule: {MarketReasoningModule.__name__}")
print(f"[OK] PlaceholderMarketTools: {PlaceholderMarketTools.__name__}")
print(f"[OK] OllamaEmbeddings: {OllamaEmbeddings.__name__}")

print("\n[SUCCESS] The agents module is now a usable Python module!")
