from typing import Sequence, Mapping, Optional
import random

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
        persona: Optional[dict] = None,
    ):
        self.agent_id = agent_id
        self.cash = initial_cash

        # Persona-driven defaults
        self._posting_prob = 0.0
        self._posting_channels: list[tuple[str, float]] = []
        if persona:
            personality = persona.get("personality_prompt", personality)
            self._posting_prob = float(persona.get("posting_probability", 0.0))
            self._posting_channels = [
                (ch.get("channel"), float(ch.get("weight", 0.0)))
                for ch in persona.get("posting_channels", [])
                if ch.get("channel")
            ]

        self.persona = persona or {}

        llm = OllamaLLM(model=llm_model, base_url=llm_base_url)
        self.agent = PredictionMarketAgent(
            llm=llm,
            personality_prompt=personality,
            initial_bankroll=initial_cash
        )

        self.inbox = []
        self._last_inbox = []
        self._cached_decision = None

    def ingest(self, messages: Sequence[Mapping[str, object]]) -> None:
        self.inbox.extend(messages)

    def update_belief(self, timestep: int, market_price: float) -> float:
        if not self.inbox:
            return self._cached_decision.get('confidence', 0.5) if self._cached_decision else 0.5

        self._last_inbox = list(self.inbox)
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

    def generate_posts(self, timestep: int) -> list[dict]:
        """Optionally produce portal posts based on persona settings.

        Returns a list of payloads with keys: target_node, content, tagline, timestamp.
        """
        feed = self.inbox or self._last_inbox
        if self._posting_prob <= 0 or not feed:
            return []

        posts = []
        channel_choices = self._posting_channels or [(None, 1.0)]
        weights = [w for _, w in channel_choices]
        total = sum(weights)
        if total <= 0:
            return []
        weights = [w / total for w in weights]

        for msg in feed[-3:]:  # consider recent messages
            if random.random() > self._posting_prob:
                continue
            target, _ = random.choices(channel_choices, weights=weights, k=1)[0]
            if not target:
                continue
            posts.append(
                {
                    "target_node": target,
                    "content": msg.get("description") or msg.get("tagline") or str(msg),
                    "tagline": msg.get("tagline", "Agent update"),
                    "timestamp": timestep,
                    "event_id": f"post_{self.agent_id}_{timestep}_{random.randint(0, 9999)}",
                }
            )
        return posts

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
            print(f"[WARNING] Agent {self.agent_id} workflow failed: {e}")
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
    llm_model: str = "llama3.1:8b",
    persona: Optional[dict] = None,
) -> PredictionMarketAgentAdapter:
    return PredictionMarketAgentAdapter(
        agent_id=agent_id,
        personality=personality,
        initial_cash=initial_cash,
        llm_model=llm_model,
        persona=persona,
    )
