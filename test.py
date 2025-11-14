from typing import Dict, Any, List, Optional, Union
import json
import xml.etree.ElementTree as ET
import uuid
import requests

class OllamaLLM:
    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def __call__(self, messages: List[Dict[str, str]], model: Optional[str] = None,
                 temperature: float = 0.0, max_tokens: int = 500,
                 stop_strs: Optional[List[str]] = None, n: int = 1) -> Union[str, List[str]]:
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": model or self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens}
            }
        )
        if response.status_code == 200:
            return response.json()['message']['content']
        else:
            raise Exception(f"Ollama error: {response.text}")

class MarketPlanningModule:
    def __init__(self, llm):
        self.plan = []
        self.llm = llm

    def __call__(self, task_type, task_description, feedback, few_shot=''):
        import re, ast
        prompt = f'''Analyze event for trading decision.
sub-task 1: {{"description": "Extract event information"}}
sub-task 2: {{"description": "Retrieve historical events"}}
sub-task 3: {{"description": "Analyze market price"}}
sub-task 4: {{"description": "Make trading decision"}}

Task: {task_description}'''
        messages = [{"role": "user", "content": prompt}]
        string = self.llm(messages=messages, temperature=0.1)
        dict_strings = re.findall(r"\{[^{}]*\}", string)
        self.plan = [ast.literal_eval(ds) for ds in dict_strings] if dict_strings else [{"description": "analyze"}]
        return self.plan


class MarketReasoningModule:
    def __init__(self, personality_prompt: str, llm):
        self.personality_prompt = personality_prompt
        self.llm = llm

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


class SimpleMemoryModule:
    def __init__(self):
        self.event_history = []

    def addMemory(self, event_data: str, outcome: Optional[str] = None, trade_result: Optional[Dict] = None):
        self.event_history.append({'event': event_data, 'outcome': outcome, 'trade': trade_result})

    def retriveMemory(self, query_event: str, k: int = 3) -> str:
        if not self.event_history:
            return "No historical events."
        recent = self.event_history[-k:]
        context = "Recent Events:\n"
        for i, mem in enumerate(recent, 1):
            context += f"{i}. Event: {mem['event']}\n"
        return context

    def get_full_history(self) -> List[Dict]:
        return self.event_history.copy()


# Standalone Agent
class PredictionMarketAgent:
    def __init__(self, llm, personality_prompt: str = "", initial_bankroll: float = 10000.0):
        self.llm = llm
        self.bankroll = initial_bankroll
        self.initial_bankroll = initial_bankroll
        self.trade_history = []
        self.personality_prompt = personality_prompt or "Balanced trader"
        self.memory = SimpleMemoryModule()
        self.planning = MarketPlanningModule(llm=self.llm)
        self.reasoning = MarketReasoningModule(personality_prompt=self.personality_prompt, llm=self.llm)

    def insert_event(self, event: Dict[str, Any]):
        self.current_event = event

    def workflow(self) -> Dict[str, Any]:
        try:
            event = getattr(self, 'current_event', None)
            if not event:
                return self._error_result("No event")

            event_id = event.get('event_id', 'UNKNOWN')
            event_description = event.get('description', str(event))

            plan = self.planning('Trading', f"Analyze: {event_description}", '', '')
            historical_context = self.memory.retriveMemory(event_description)
            reasoning_result = self.reasoning(event_description, 0.5, historical_context)

            decision = self._parse_decision(reasoning_result)

            if decision['amount'] > self.bankroll:
                decision['amount'] = self.bankroll * 0.1

            if decision['decision'] == 'BUY' and decision['amount'] > 0:
                self.bankroll -= decision['amount']

            self.memory.addMemory(event_description, None, decision)
            self.trade_history.append({'event_id': event_id, 'decision': decision, 'bankroll_after': self.bankroll})

            return {'decision': decision['decision'], 'amount': decision['amount'],
                    'confidence': decision['confidence'], 'analysis': decision['analysis'],
                    'bankroll': self.bankroll, 'event_id': event_id}

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
            return {'analysis': reasoning_result[:200], 'decision': 'SELL', 'amount': 0.0, 'confidence': 0.0}

    def _error_result(self, error_msg: str) -> Dict[str, Any]:
        return {'decision': 'SELL', 'amount': 0.0, 'confidence': 0.0, 'analysis': f'Error: {error_msg}',
                'bankroll': self.bankroll, 'event_id': 'ERROR'}

    def get_bankroll(self) -> float:
        return self.bankroll

    def get_performance_summary(self) -> Dict[str, Any]:
        return {
            'initial_bankroll': self.initial_bankroll,
            'current_bankroll': self.bankroll,
            'profit_loss': self.bankroll - self.initial_bankroll,
            'profit_loss_pct': ((self.bankroll - self.initial_bankroll) / self.initial_bankroll * 100),
            'total_trades': len(self.trade_history)
        }


# TEST
print("="*70)
print("PREDICTION MARKET AGENT TEST")
print("="*70)

print("\n1. Creating Ollama LLM...")
llm = OllamaLLM(model="llama3.1:8b")
print("✓ LLM created")

print("\n2. Creating PredictionMarketAgent...")
agent = PredictionMarketAgent(
    llm=llm,
    personality_prompt="Conservative trader, risk-averse, prefers 5-10% position sizes",
    initial_bankroll=10000.0
)
print(f"✓ Agent created with ${agent.get_bankroll():.2f} bankroll")

events = [
    "Will Bitcoin reach $100,000 by end of 2025?",
    "Will Apple release AR glasses in 2025?",
    "Will the US economy enter recession in 2025?",
]

print(f"\n3. Processing {len(events)} events...\n")

results = []
for i, event_desc in enumerate(events, 1):
    print(f"Event {i}: {event_desc}")

    agent.insert_event({'event_id': f'EVENT_{i:03d}', 'description': event_desc, 'outcome': 'YES'})
    result = agent.workflow()
    results.append(result)

    print(f"  → Decision: {result['decision']}")
    print(f"  → Amount: ${result['amount']:.2f}")
    print(f"  → Confidence: {result['confidence']:.2f}")
    print(f"  → Analysis: {result['analysis'][:100]}...")
    print(f"  → Bankroll remaining: ${result['bankroll']:.2f}\n")

print("="*70)
print("SUMMARY")
print("="*70)

summary = agent.get_performance_summary()
print(f"\nInitial Bankroll:  ${summary['initial_bankroll']:.2f}")
print(f"Final Bankroll:    ${summary['current_bankroll']:.2f}")
print(f"Profit/Loss:       ${summary['profit_loss']:.2f} ({summary['profit_loss_pct']:.1f}%)")
print(f"Total Trades:      {summary['total_trades']}")

print("\n" + "="*70)
print("DETAILED RESULTS")
print("="*70)

for i, result in enumerate(results, 1):
    print(f"\nEvent {i}: {events[i-1]}")
    print(f"  Decision:   {result['decision']}")
    print(f"  Amount:     ${result['amount']:.2f}")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Analysis:   {result['analysis']}")

print("\n✅ TEST COMPLETE")
