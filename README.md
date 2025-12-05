Prediction Market Simulation

A simulation framework for multi-agent prediction markets using LLM-powered agents.

SETUP

1. Create virtual environment:
   python -m venv venv
   source venv/bin/activate

2. Install dependencies:
   pip install -r requirements.txt

3. Install Ollama, pull model, and start server:
   ollama pull llama3.2:3b
   ollama serve

4. Copy and edit config:
   cp config.env.example config.env


RUNNING A SIMULATION

Basic run (uses config.env defaults):
   python examples/run_simulation.py

With custom options:
   python examples/run_simulation.py --events data/nba_finals_2025_okc_pacers.json --agents 5

Available options:
   --events FILE      Event data JSON file
   --market TYPE      lmsr or orderbook
   --agents N         Number of agents
   --timesteps N      Max simulation steps
   --personas FILE    YAML file with agent personalities
   --seed N           Random seed (-1 for random)


EVALUATING RESULTS

After a simulation:
   python examples/evaluate_simulation.py --run-name prediction_sim --actual-outcome 1.0 --plot


GENERATING ANIMATIONS

Create HTML visualization of market activity:
   PYTHONPATH=src python -m prediction_market_sim.visualization.animate artifacts/ --run-id prediction_sim_run1


OUTPUT FILES

Simulation outputs go to artifacts/:
   - prediction_sim_run1_market.json    Market prices and volume
   - prediction_sim_run1_beliefs.json   Agent beliefs over time
   - prediction_sim_run1_flow.json      Information flow data

Dashboards go to artifacts/dashboards/:
   - prediction_sim_pnl.png             Agent profit/loss
   - prediction_sim_price.png           Price evolution
   - prediction_sim_beliefs.png         Belief convergence