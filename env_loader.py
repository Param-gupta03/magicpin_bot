"""
env_loader.py — Lightweight .env file loader for magicpin challenge
Loads environment variables from .env into os.environ if not already set.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env(env_file: str | Path = ".env") -> None:
    """Read key=value pairs from .env and populate os.environ."""
    env_path = Path(env_file)
    if not env_path.is_file():
        # Try looking one level up or at repo root
        root_path = Path(__file__).resolve().parent / ".env"
        if root_path.is_file():
            env_path = root_path
        else:
            return

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k:
                    # If not already present in environment, or if present but empty, set it
                    if not os.environ.get(k):
                        os.environ[k] = v
    except Exception:
        pass


# Automatically load on import
load_env()
