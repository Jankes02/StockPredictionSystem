"""
Secrets and environment-level settings, read from `.env` / the process
environment. Never log or print these values.
"""
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AlpacaSettings(BaseModel):
    api_key: SecretStr
    secret_key: SecretStr
    paper: bool = True


def _parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"Not a boolean: {value!r}")


def load_alpaca_settings(env_file: Optional[Path] = None) -> AlpacaSettings:
    """
    Load Alpaca credentials. Values already present in the process
    environment win over the `.env` file (so a VPS/container can inject them).
    Only paper trading is allowed at this stage of the project.
    """
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)
    key = os.environ.get("ALPACA_API_KEY", "").strip()
    secret = os.environ.get("ALPACA_SECRET_KEY", "").strip()
    if not key or not secret:
        raise RuntimeError(
            "Missing ALPACA_API_KEY / ALPACA_SECRET_KEY. Copy .env.example to .env "
            "and fill in the keys of your Alpaca PAPER account."
        )
    paper = _parse_bool(os.environ.get("ALPACA_PAPER", "true"))
    if not paper:
        raise RuntimeError("ALPACA_PAPER=false is not supported: only paper trading is enabled.")
    return AlpacaSettings(api_key=key, secret_key=secret, paper=paper)
