# Agent persona configs

These YAML files are **docs/templates only** so you can feed rich personas into an LLM (e.g., Claude) to generate detailed behavior scripts later. Nothing in the code reads these yet.

Each agent entry should include demographics, posting behavior, and a personality prompt. You can duplicate the examples in `agent_personas.yaml` and tweak fields per agent/event.

Suggested fields: `agent_id`, `name`, `age`, `gender`, `occupation`, `net_worth`, `hometown`, `risk_tolerance`, `trading_style`, `posting_probability`, `posting_channels`, `subscriptions`, and a multi-line `personality_prompt`.

Drop additional YAML files in this folder for new agents or events; keep `agent_id` in sync with whatever the simulator uses.
