from typing import Sequence, Mapping

from .prediction_market_agent import PredictionMarketAgent
from .ollama_llm import OllamaLLM
from ..simulation.interfaces import MarketOrder


class PredictionMarketAgentAdapter:

    def __init__(
        self,
        agent_id: str,
        personality: str = "rational trader",
        initial_cash: float = 10000.0,
        llm_model: str = "llama3.1:8b",
        llm_base_url: str = "http://localhost:11434",
    ):
        self.agent_id = agent_id
        self.cash = initial_cash

        llm = OllamaLLM(model=llm_model, base_url=llm_base_url)
        self.agent = PredictionMarketAgent(
            llm=llm,
            personality_prompt=personality,
            initial_bankroll=initial_cash
        )

        self.inbox = []
        self._cached_decision = None

    def ingest(self, messages: Sequence[Mapping[str, object]]) -> None:
        self.inbox.extend(messages)

    def update_belief(self, timestep: int, market_price: float) -> float:
        if not self.inbox:
            return self._cached_decision.get('confidence', 0.5) if self._cached_decision else 0.5

        self._run_workflow()
        self.inbox.clear()

        return self._cached_decision.get('confidence', 0.5)

    def generate_order(self, belief: float, market_price: float) -> MarketOrder | None:
        if not self._cached_decision:
            return None

        decision = self._cached_decision

        if decision['decision'] not in ['BUY', 'SELL'] or decision['amount'] <= 0:
            return None

        return MarketOrder(
            agent_id=self.agent_id,
            side=decision['decision'].lower(),
            size=min(decision['amount'], self.cash * 0.5),
            limit_price=market_price,
            confidence=decision['confidence'],
            metadata={'analysis': decision.get('analysis', '')}
        )

    def _run_workflow(self):
        try:
            event = {
                'event_id': f'BATCH_{len(self.inbox)}',
                'description': '\n'.join([
                    msg.get('tagline', msg.get('description', ''))
                    for msg in self.inbox[-5:]
                ]),
                'outcome': 'YES'
            }

            self.agent.insert_event(event)
            self._cached_decision = self.agent.workflow()

            self.cash = self._cached_decision.get('bankroll', self.cash)

        except Exception as e:
            print(f"⚠️  Agent {self.agent_id} workflow failed: {e}")
            self._cached_decision = {
                'decision': 'SELL',
                'amount': 0.0,
                'confidence': 0.5,
                'analysis': f'Error: {e}',
                'bankroll': self.cash
            }


def create_prediction_agent(
    agent_id: str,
    personality: str = "rational trader",
    initial_cash: float = 10000.0,
    llm_model: str = "llama3.1:8b"
) -> PredictionMarketAgentAdapter:
    return PredictionMarketAgentAdapter(
        agent_id=agent_id,
        personality=personality,
        initial_cash=initial_cash,
        llm_model=llm_model
    )
