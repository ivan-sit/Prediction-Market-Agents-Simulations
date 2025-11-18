"""
Adapters to connect agent implementations to simulation engine protocols.

This bridges the gap between your colleagues' LLM-based agent and the
simulation engine's expected Agent interface.
"""

from typing import Sequence, Mapping
import xml.etree.ElementTree as ET

from .prediction_market_agent import PredictionMarketAgent
from .ollama_llm import OllamaLLM, OllamaEmbeddings
from ..simulation.interfaces import MarketOrder


class PredictionMarketAgentAdapter:
    """
    Adapter that wraps PredictionMarketAgent to match simulation Agent protocol.
    
    The PredictionMarketAgent uses LLM reasoning with personality and memory,
    but we need to adapt it to the simulation's Agent interface.
    """
    
    def __init__(
        self,
        agent_id: str,
        personality: str = "rational trader",
        initial_cash: float = 10000.0,
        llm_model: str = "llama3",
        llm_base_url: str = "http://localhost:11434",
    ):
        """
        Initialize the agent adapter.
        
        Args:
            agent_id: Unique identifier for this agent
            personality: Personality/trading style description
            initial_cash: Starting cash balance
            llm_model: Ollama model name (e.g., "llama3", "mistral")
            llm_base_url: Ollama server URL
        """
        self.agent_id = agent_id
        self.personality = personality
        self.cash = initial_cash
        self.current_belief = 0.5  # Start neutral
        
        # Initialize LLM
        try:
            self.llm = OllamaLLM(model=llm_model, base_url=llm_base_url)
            self.embeddings = OllamaEmbeddings(model=llm_model, base_url=llm_base_url)
        except Exception as e:
            print(f"⚠️  Warning: Could not connect to Ollama: {e}")
            print(f"   Agent {agent_id} will use fallback random behavior")
            self.llm = None
            self.embeddings = None
        
        # Initialize the actual agent (if LLM available)
        if self.llm and self.embeddings:
            try:
                self.agent = PredictionMarketAgent(
                    agent_id=agent_id,
                    llm=self.llm,
                    embeddings=self.embeddings,
                    personality=personality
                )
            except Exception as e:
                print(f"⚠️  Warning: Could not initialize PredictionMarketAgent: {e}")
                self.agent = None
        else:
            self.agent = None
        
        # Message buffer
        self.inbox: list[dict] = []
        
    def ingest(self, messages: Sequence[Mapping[str, object]]) -> None:
        """
        Receive messages from the portal network.
        
        Args:
            messages: List of event messages from subscribed portals
        """
        self.inbox.extend(messages)
        
    def update_belief(self, timestep: int, market_price: float) -> float:
        """
        Update agent's belief about outcome probability based on new information.
        
        Args:
            timestep: Current simulation timestep
            market_price: Current market price (interpreted as probability)
            
        Returns:
            Updated belief probability (0.0 to 1.0)
        """
        if not self.inbox:
            # No new information, maintain current belief
            return self.current_belief
        
        if not self.agent or not self.llm:
            # Fallback: Simple random walk around market price
            import random
            self.current_belief = max(0.0, min(1.0, market_price + random.uniform(-0.05, 0.05)))
            return self.current_belief
        
        # Use LLM agent to analyze inbox messages and update belief
        try:
            # Prepare context from inbox
            event_context = "\n".join([
                f"- {msg.get('tagline', msg.get('description', 'Event'))}"
                for msg in self.inbox[-5:]  # Last 5 messages
            ])
            
            # Query agent's reasoning module
            task_desc = f"Market price: {market_price:.2%}. Recent events: {event_context}"
            
            response = self.agent.reasoning(
                task_description=task_desc,
                current_price=market_price
            )
            
            # Parse confidence from XML response
            confidence = self._parse_confidence(response)
            if confidence is not None:
                self.current_belief = confidence
            else:
                # Fallback if parsing failed
                self.current_belief = market_price
                
        except Exception as e:
            print(f"⚠️  Agent {self.agent_id} reasoning failed: {e}")
            self.current_belief = market_price
        
        return self.current_belief
    
    def generate_order(self, belief: float, market_price: float) -> MarketOrder | None:
        """
        Generate a market order based on belief vs. market price.
        
        Args:
            belief: Agent's belief about outcome probability
            market_price: Current market price
            
        Returns:
            MarketOrder if agent wants to trade, None otherwise
        """
        if not self.agent or not self.llm:
            # Fallback: Simple contrarian strategy
            threshold = 0.10  # Trade if belief differs by >10%
            if abs(belief - market_price) > threshold:
                side = "buy" if belief > market_price else "sell"
                size = min(100.0, self.cash * 0.1)  # Risk 10% of cash
                
                return MarketOrder(
                    agent_id=self.agent_id,
                    side=side,
                    size=size,
                    limit_price=market_price,
                    confidence=abs(belief - market_price),
                    metadata={'strategy': 'fallback_contrarian'}
                )
            return None
        
        # Use LLM agent to make trading decision
        try:
            event_context = "\n".join([
                f"- {msg.get('tagline', '')}"
                for msg in self.inbox[-3:]
            ])
            
            task_desc = f"Events: {event_context}. Your belief: {belief:.2%}, Market: {market_price:.2%}"
            
            response = self.agent.reasoning(
                task_description=task_desc,
                current_price=market_price
            )
            
            # Parse decision
            decision, amount, confidence = self._parse_decision(response)
            
            if decision and amount and amount > 0:
                return MarketOrder(
                    agent_id=self.agent_id,
                    side=decision.lower(),
                    size=min(amount, self.cash * 0.5),  # Don't over-commit
                    limit_price=market_price,
                    confidence=confidence or abs(belief - market_price),
                    metadata={'llm_reasoning': True}
                )
                
        except Exception as e:
            print(f"⚠️  Agent {self.agent_id} order generation failed: {e}")
        
        return None
    
    def _parse_confidence(self, xml_response: str) -> float | None:
        """Parse confidence from LLM XML response."""
        try:
            root = ET.fromstring(f"<root>{xml_response}</root>")
            conf_elem = root.find('.//confidence')
            if conf_elem is not None and conf_elem.text:
                return float(conf_elem.text)
        except:
            pass
        return None
    
    def _parse_decision(self, xml_response: str) -> tuple[str | None, float | None, float | None]:
        """Parse decision, amount, confidence from LLM XML response."""
        decision = None
        amount = None
        confidence = None
        
        try:
            root = ET.fromstring(f"<root>{xml_response}</root>")
            
            dec_elem = root.find('.//decision')
            if dec_elem is not None and dec_elem.text:
                decision = dec_elem.text.strip().upper()
                if decision not in ['BUY', 'SELL']:
                    decision = None
            
            amt_elem = root.find('.//amount')
            if amt_elem is not None and amt_elem.text:
                # Parse dollar amount
                amt_text = amt_elem.text.strip().replace('$', '').replace(',', '')
                amount = float(amt_text)
            
            conf_elem = root.find('.//confidence')
            if conf_elem is not None and conf_elem.text:
                confidence = float(conf_elem.text)
                
        except Exception as e:
            print(f"⚠️  Failed to parse LLM response: {e}")
        
        return decision, amount, confidence


def create_prediction_agent(
    agent_id: str,
    personality: str = "rational trader",
    initial_cash: float = 10000.0,
    llm_model: str = "llama3"
) -> PredictionMarketAgentAdapter:
    """
    Factory function to create a prediction market agent.
    
    Args:
        agent_id: Unique identifier
        personality: Trading personality/style
        initial_cash: Starting cash balance
        llm_model: Ollama model to use
        
    Returns:
        Agent adapter ready for simulation
    """
    return PredictionMarketAgentAdapter(
        agent_id=agent_id,
        personality=personality,
        initial_cash=initial_cash,
        llm_model=llm_model
    )

