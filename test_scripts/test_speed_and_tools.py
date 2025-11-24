import time
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from prediction_market_sim.agents.prediction_market_agent import PredictionMarketAgent, MarketMemoryModule
from prediction_market_sim.market.lmsr import LMSRMarket
from prediction_market_sim.agents.ollama_llm import OllamaLLM

class TestSpeedAndTools(unittest.TestCase):
    def setUp(self):
        self.market = LMSRMarket()
        self.llm = MagicMock(spec=OllamaLLM)
        self.llm.get_embedding_model.return_value = MagicMock()
        
        # Mock Memory Module to avoid ChromaDB issues
        with patch('prediction_market_sim.agents.prediction_market_agent.MarketMemoryModule') as MockMemory:
            self.mock_memory = MockMemory.return_value
            self.mock_memory.retriveMemory.return_value = ""
            self.agent = PredictionMarketAgent(llm=self.llm, market=self.market)
            # Ensure the agent uses the mock memory (though init should have set it)
            self.agent.memory = self.mock_memory

    def test_market_interaction(self):
        """Test that agent places trades against the real market."""
        # Mock LLM response to buy
        self.llm.return_value = "<analysis>Good</analysis><decision>BUY</decision><amount>100.0</amount><confidence>0.8</confidence>"
        
        event = {"event_id": "E1", "description": "Test Event", "outcome": "YES"}
        self.agent.insert_event(event)
        result = self.agent.workflow()
        
        self.assertEqual(result['decision'], 'BUY')
        self.assertEqual(result['amount'], 100.0)
        
        # Check market state
        self.assertGreater(self.market.get_price("YES"), 0.5) # Price should go up after buy
        self.assertEqual(len(self.market.get_trades()), 1)

    def test_k_retry_logic(self):
        """Test that agent retries on malformed XML."""
        # Sequence of responses: 1 for planning, then 2 failures, then success
        self.llm.side_effect = [
            "Plan",
            "I think we should buy.", # Invalid XML
            "<decision>BUY</decision>", # Missing tags
            "<analysis>Finally</analysis><decision>SELL</decision><amount>50.0</amount><confidence>0.6</confidence>"
        ]
        
        event = {"event_id": "E2", "description": "Retry Event", "outcome": "YES"}
        self.agent.insert_event(event)
        result = self.agent.workflow()
        
        self.assertEqual(result['decision'], 'SELL')
        self.assertEqual(self.llm.call_count, 4) # 1 planning + 3 reasoning attempts

    def test_k_retry_failure(self):
        """Test that agent handles complete failure after k retries."""
        # 1 for planning, 3 for reasoning retries
        self.llm.side_effect = ["Plan"] + ["Invalid"] * 3
        
        event = {"event_id": "E3", "description": "Fail Event", "outcome": "YES"}
        self.agent.insert_event(event)
        result = self.agent.workflow()
        
        self.assertEqual(result['decision'], 'SELL') # Default fallback
        self.assertEqual(result['amount'], 0.0)
        self.assertIn("Failed to parse", result['analysis'])

if __name__ == '__main__':
    unittest.main()
