"""Utilities for loading agent personas from YAML/JSON configs.

These helpers are optional; they simply parse persona files into dictionaries
that you can pass when constructing agents (e.g., PredictionMarketAgentAdapter).
The simulator does not automatically read these files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import json
import os

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def load_personas(path: str | Path) -> List[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Persona file not found: {path}")

    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("pyyaml is required to load YAML persona files")
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    else:
        with open(path, "r") as f:
            data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Persona file must contain a list of agent entries")
    return data


def personas_by_id(path: str | Path) -> Dict[str, dict]:
    personas = load_personas(path)
    return {p["agent_id"]: p for p in personas if "agent_id" in p}


def list_persona_files(config_dir: str | Path) -> List[Path]:
    """List YAML/JSON persona files in a directory."""
    config_dir = Path(config_dir)
    if not config_dir.exists():
        return []
    return sorted([p for p in config_dir.iterdir() if p.suffix.lower() in {".yaml", ".yml", ".json"}])


def select_persona_file(
    *,
    config_dir: str | Path = "configs/agents",
    env_var: str = "AGENT_PERSONAS_FILE",
) -> Optional[Path]:
    """Select a persona file based on an env var or defaults.

    Logic:
    - If env var is set, match by filename or basename in config_dir.
    - If only one persona file exists in config_dir, return it.
    - Otherwise return None.
    """

    config_dir = Path(config_dir)
    files = list_persona_files(config_dir)
    if not files:
        return None

    env_val = os.getenv(env_var)
    if env_val:
        target = Path(env_val)
        # If absolute/relative path exists, use it
        if target.exists():
            return target
        # Otherwise match by name in config_dir
        for f in files:
            if f.name == env_val or f.stem == env_val:
                return f
        return None

    if len(files) == 1:
        return files[0]
    return None
