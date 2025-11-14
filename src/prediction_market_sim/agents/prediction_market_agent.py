from typing import Dict, Any, List, Optional
from .websocietysimulator.agent import SimulationAgent
from .websocietysimulator.llm import LLMBase
from .websocietysimulator.agent.modules.planning_modules import PlanningBase
from .websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from .websocietysimulator.agent.modules.memory_modules import MemoryBase
from langchain_chroma import Chroma
from langchain_core.documents import Document
import json
import xml.etree.ElementTree as ET
import uuid
import random

class MarketPlanningModule(PlanningBase):
    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)

    def create_prompt(self, task_type: str, task_description: str, feedback: str, few_shot: str) -> str:
        base = '''Analyze event for trading decision.
sub-task 1: {{"description": "Extract event information", "reasoning instruction": "Identify factors"}}
sub-task 2: {{"description": "Retrieve historical events", "reasoning instruction": "Compare patterns"}}
sub-task 3: {{"description": "Analyze market price", "reasoning instruction": "Evaluate opportunity"}}
sub-task 4: {{"description": "Make trading decision", "reasoning instruction": "Apply personality"}}

Task: {task_description}'''
        if feedback:
            base = f"Reflection: {feedback}\n\n" + base
        return base.format(task_description=task_description)


class MarketReasoningModule(ReasoningBase):
    def __init__(self, personality_prompt: str, memory: MemoryBase, llm: LLMBase):
        super().__init__(profile_type_prompt=personality_prompt, memory=memory, llm=llm)
        self.personality_prompt = personality_prompt

    def __call__(self, task_description: str, current_price: Optional[float] = None, historical_context: str = '') -> str:
        prompt = f'''Personality: {self.personality_prompt}

{historical_context}

Event: {task_description}
Price: {current_price if current_price else "N/A"}

Decide BUY or SELL and amount.

Output format (use exact XML tags):
<analysis>2-3 sentence reasoning</analysis>
<decision>BUY or SELL</decision>
<amount>dollar amount</amount>
<confidence>0.0 to 1.0</confidence>'''

        messages = [{"role": "user", "content": prompt}]
        return self.llm(messages=messages, temperature=0.2, max_tokens=500)


class MarketMemoryModule(MemoryBase):
    def __init__(self, llm: LLMBase):
        super().__init__(memory_type='prediction_market', llm=llm)
        self.event_history = []

    def addMemory(self, event_data: str, outcome: Optional[str] = None, trade_result: Optional[Dict] = None):
        self.event_history.append({
            'event': event_data,
            'outcome': outcome,
            'trade': trade_result,
            'timestep': len(self.event_history)
        })

        memory_content = f"Event: {event_data}"
        if outcome:
            memory_content += f"\nOutcome: {outcome}"
        if trade_result:
            memory_content += f"\nTrade: {json.dumps(trade_result)}"

        doc = Document(
            page_content=event_data,
            metadata={
                "event": event_data,
                "outcome": outcome or "unknown",
                "trade": json.dumps(trade_result) if trade_result else "{}",
                "timestep": len(self.event_history) - 1,
                "full_context": memory_content
            }
        )
        self.scenario_memory.add_documents([doc])

    def retriveMemory(self, query_event: str, k: int = 3) -> str:
        if self.scenario_memory._collection.count() == 0:
            return "No historical events."

        results = self.scenario_memory.similarity_search_with_score(
            query_event, k=min(k, self.scenario_memory._collection.count())
        )

        context = "Similar Events:\n"
        for i, (result, score) in enumerate(results, 1):
            context += f"{i}. {result.metadata['full_context']}\n"
        return context

    def get_full_history(self) -> List[Dict]:
        return self.event_history.copy()


class PlaceholderMarketTools:
    @staticmethod
    def get_market_price(event_id: str) -> float:
        return 0.5

    @staticmethod
    def place_market_order(event_id: str, action: str, amount: float, outcome: str) -> Dict[str, Any]:
        return {
            'success': True,
            'order_id': f"ORDER_{uuid.uuid4().hex[:8]}",
            'event_id': event_id,
            'action': action,
            'amount': amount,
            'outcome': outcome,
            'price': 0.5
        }


class PredictionMarketAgent(SimulationAgent):
    def __init__(self, llm: LLMBase, personality_prompt: str = "", initial_bankroll: float = 10000.0,
                 subscribed_sources: List[str] = None, cross_post_probability: float = 0.02):
        super().__init__(llm=llm)
        self.bankroll = initial_bankroll
        self.initial_bankroll = initial_bankroll
        self.trade_history = []
        self.personality_prompt = personality_prompt or "Balanced analytical trader with moderate risk tolerance."
        self.memory = MarketMemoryModule(llm=self.llm)
        self.planning = MarketPlanningModule(llm=self.llm)
        self.reasoning = MarketReasoningModule(
            personality_prompt=self.personality_prompt,
            memory=self.memory,
            llm=self.llm
        )
        self.market_tools = PlaceholderMarketTools()
        self.subscribed_sources = subscribed_sources or []
        self.cross_post_probability = cross_post_probability
        self.portal_network = None
        self.cross_post_history = []

    def set_portal_network(self, portal_network):
        self.portal_network = portal_network

    def insert_event(self, event: Dict[str, Any]):
        self.current_event = event

    def _should_cross_post(self, decision: Dict[str, Any], event: Dict[str, Any]) -> bool:
        return random.random() < self.cross_post_probability

    def _select_target_sources(self, source_of_event: str) -> List[str]:
        all_sources = ["twitter", "reddit", "news", "discord", "telegram"]
        available_sources = [s for s in all_sources if s != source_of_event]
        num_targets = random.randint(1, 2)
        return random.sample(available_sources, min(num_targets, len(available_sources)))

    def _create_cross_post_payload(self, decision: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'type': 'agent_signal',
            'agent_id': getattr(self, 'agent_id', 'unknown'),
            'event_id': event.get('event_id', 'UNKNOWN'),
            'event_description': event.get('description', ''),
            'signal': decision['decision'],
            'confidence': decision['confidence'],
            'analysis_summary': decision['analysis'][:200],
            'timestamp': event.get('timestamp', None)
        }

    def cross_post_to_sources(self, decision: Dict[str, Any], event: Dict[str, Any]):
        if self.portal_network is None:
            return

        if not self._should_cross_post(decision, event):
            return

        source_of_event = event.get('source', 'unknown')
        target_sources = self._select_target_sources(source_of_event)

        if not target_sources:
            return

        payload = self._create_cross_post_payload(decision, event)

        for target_source in target_sources:
            try:
                self.cross_post_history.append({
                    'from_source': source_of_event,
                    'to_source': target_source,
                    'event_id': event.get('event_id'),
                    'confidence': decision['confidence'],
                    'payload': payload
                })
                print(f"[CROSS-POST] {source_of_event} -> {target_source}: {payload['signal']} (conf: {payload['confidence']:.2f})")
            except Exception as e:
                print(f"[CROSS-POST ERROR] {target_source}: {e}")

    def process_events(self, events: List[str]) -> List[Dict[str, Any]]:
        results = []
        for i, event_desc in enumerate(events):
            event = {
                'event_id': f'EVENT_{i:03d}',
                'description': event_desc,
                'outcome': 'YES'
            }
            self.insert_event(event)
            results.append(self.workflow())
        return results

    def workflow(self) -> Dict[str, Any]:
        try:
            event = getattr(self, 'current_event', None)
            if not event:
                return self._error_result("No event")

            event_id = event.get('event_id', 'UNKNOWN')
            event_description = event.get('description', str(event))

            task_description = f"Analyze: {event_description}"
            plan = self.planning(
                task_type='Trading',
                task_description=task_description,
                feedback='',
                few_shot=''
            )

            historical_context = ""
            current_price = None

            for sub_task in plan:
                desc = sub_task.get('description', '')
                if 'historical' in desc.lower() or 'memory' in desc.lower():
                    historical_context = self.memory.retriveMemory(event_description)
                elif 'price' in desc.lower():
                    current_price = self.market_tools.get_market_price(event_id)

            reasoning_result = self.reasoning(
                task_description=event_description,
                current_price=current_price,
                historical_context=historical_context
            )

            decision = self._parse_decision(reasoning_result)

            if decision['amount'] > self.bankroll:
                decision['amount'] = self.bankroll * 0.1
                decision['analysis'] += " [Capped at 10%]"

            if decision['amount'] > 0:
                order_result = self.market_tools.place_market_order(
                    event_id=event_id,
                    action=decision['decision'],
                    amount=decision['amount'],
                    outcome=event.get('outcome', 'YES')
                )
                if decision['decision'] == 'BUY':
                    self.bankroll -= decision['amount']
                decision['order'] = order_result

            self.memory.addMemory(
                event_data=event_description,
                outcome=None,
                trade_result=decision
            )

            self.trade_history.append({
                'event_id': event_id,
                'decision': decision,
                'bankroll_after': self.bankroll
            })

            self.cross_post_to_sources(decision, event)

            return {
                'decision': decision['decision'],
                'amount': decision['amount'],
                'confidence': decision['confidence'],
                'analysis': decision['analysis'],
                'bankroll': self.bankroll,
                'event_id': event_id
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return self._error_result(str(e))

    def _parse_decision(self, reasoning_result: str) -> Dict[str, Any]:
        try:
            root = ET.fromstring(f"<response>{reasoning_result}</response>")
            analysis = root.find('analysis')
            decision = root.find('decision')
            amount = root.find('amount')
            confidence = root.find('confidence')

            return {
                'analysis': analysis.text.strip() if analysis is not None and analysis.text else "No analysis",
                'decision': decision.text.strip().upper() if decision is not None and decision.text else "SELL",
                'amount': float(amount.text.strip()) if amount is not None and amount.text else 0.0,
                'confidence': float(confidence.text.strip()) if confidence is not None and confidence.text else 0.5
            }
        except Exception as e:
            return {
                'analysis': reasoning_result[:200],
                'decision': 'SELL',
                'amount': 0.0,
                'confidence': 0.0
            }

    def _error_result(self, error_msg: str) -> Dict[str, Any]:
        return {
            'decision': 'SELL',
            'amount': 0.0,
            'confidence': 0.0,
            'analysis': f'Error: {error_msg}',
            'bankroll': self.bankroll,
            'event_id': 'ERROR'
        }

    def get_bankroll(self) -> float:
        return self.bankroll

    def get_trade_history(self) -> List[Dict]:
        return self.trade_history.copy()

    def get_performance_summary(self) -> Dict[str, Any]:
        return {
            'initial_bankroll': self.initial_bankroll,
            'current_bankroll': self.bankroll,
            'profit_loss': self.bankroll - self.initial_bankroll,
            'profit_loss_pct': ((self.bankroll - self.initial_bankroll) / self.initial_bankroll * 100),
            'total_trades': len(self.trade_history),
            'event_history_size': len(self.memory.get_full_history()),
            'cross_posts': len(self.cross_post_history)
        }

    def get_cross_post_history(self) -> List[Dict]:
        return self.cross_post_history.copy()
