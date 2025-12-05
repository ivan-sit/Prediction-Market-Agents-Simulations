from typing import Dict, Any, List, Optional
from ..market.lmsr import LMSRMarket
from .websocietysimulator.agent import SimulationAgent
from .websocietysimulator.llm import LLMBase
from .websocietysimulator.agent.modules.planning_modules import PlanningBase
from .websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from .websocietysimulator.agent.modules.memory_modules import MemoryBase
from langchain_chroma import Chroma
from langchain_core.documents import Document
import json
import xml.etree.ElementTree as ET
import dataclasses

class MarketPlanningModule(PlanningBase):
    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)

    def create_prompt(self, task_type: str, task_description: str, feedback: str, few_shot: str) -> str:
        base = '''This is a fictional market simulation game. You are an AI agent trading virtual assets.
Analyze event for trading decision.
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
        prompt = f'''This is a fictional market simulation game. You are an AI agent trading virtual assets.
Personality: {self.personality_prompt}

{historical_context}

Event: {task_description}
Price: {current_price if current_price else "N/A"}

Decide BUY or SELL and amount.

Output format (use exact XML tags):
<analysis>2-3 sentence reasoning</analysis>
<decision>BUY or SELL</decision>
<amount>100.0</amount>  <!-- Dollar amount as decimal number, e.g., 100.0 for $100 -->
<confidence>0.75</confidence>  <!-- Decimal between 0.0 and 1.0, e.g., 0.75 for 75% confident -->'''

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

class PredictionMarketAgent(SimulationAgent):
    def __init__(self, llm: LLMBase, market: Optional[LMSRMarket] = None, personality_prompt: str = "", initial_bankroll: float = 10000.0, agent_id: str = "unknown"):
        super().__init__(llm=llm)
        self.agent_id = agent_id
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
        self.market = market

    def set_market(self, market: LMSRMarket):
        self.market = market

    def insert_event(self, event: Dict[str, Any]):
        self.current_event = event

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

    def workflow(self, timestep: int = 0) -> Dict[str, Any]:
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
                    current_price = self.market.get_price("YES") # Assuming binary YES/NO for now

            # K-Retry Loop
            max_retries = 3
            decision = None
            last_error = ""

            for attempt in range(max_retries):
                try:
                    reasoning_result = self.reasoning(
                        task_description=event_description + (f"\n\nPrevious attempt failed: {last_error}" if last_error else ""),
                        current_price=current_price,
                        historical_context=historical_context
                    )
                    
                    decision = self._parse_decision(reasoning_result)
                    break # Success
                except ValueError as e:
                    last_error = str(e)
                    print(f"Attempt {attempt+1}/{max_retries} failed to parse: {e}")
                    if attempt == max_retries - 1:
                         # Fallback on final failure
                        decision = {
                            'analysis': f"Failed to parse after {max_retries} attempts. Last error: {last_error}",
                            'decision': 'SELL',
                            'amount': 0.0,
                            'confidence': 0.0
                        }

            if decision['amount'] > self.bankroll:
                decision['amount'] = self.bankroll * 0.1
                decision['analysis'] += " [Capped at 10%]"

            if decision['amount'] > 0:
                # Execute trade against real market
                try:
                    if decision['decision'] == 'BUY':
                        # Buying YES shares
                        trade = self.market.buy_up_to_price(
                            agent_id=getattr(self, 'agent_id', 'unknown'),
                            outcome='YES',
                            max_cost=decision['amount'],
                            timestamp=timestep
                        )
                        if trade:
                            self.bankroll -= trade.cost
                            decision['order'] = dataclasses.asdict(trade)
                        else:
                             decision['order'] = {'success': False, 'reason': 'Market rejected trade (cost too low?)'}
                    else:
                        # Selling (Buying NO shares)
                        trade = self.market.buy_up_to_price(
                            agent_id=getattr(self, 'agent_id', 'unknown'),
                            outcome='NO',
                            max_cost=decision['amount'],
                            timestamp=timestep
                        )
                        if trade:
                            self.bankroll -= trade.cost
                            decision['order'] = dataclasses.asdict(trade)
                        else:
                             decision['order'] = {'success': False, 'reason': 'Market rejected trade'}
                except Exception as e:
                    print(f"Market execution error: {e}")
                    decision['order'] = {'success': False, 'reason': str(e)}

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

            if decision is None or decision.text is None:
                raise ValueError("Missing <decision> tag")
            if amount is None or amount.text is None:
                raise ValueError("Missing <amount> tag")

            return {
                'analysis': analysis.text.strip() if analysis is not None and analysis.text else "No analysis",
                'decision': decision.text.strip().upper(),
                'amount': float(amount.text.strip()),
                'confidence': float(confidence.text.strip()) if confidence is not None and confidence.text else 0.5
            }
        except Exception as e:
            raise ValueError(f"XML parsing failed: {e}")

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
            'event_history_size': len(self.memory.get_full_history())
        }
