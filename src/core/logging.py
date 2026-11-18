import logging
import sys
from pathlib import Path
from typing import Optional

FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """Configure root logging to stderr and, optionally, an append-mode file."""
    handlers: list = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level.upper(), format=FORMAT, handlers=handlers, force=True)
    # Third-party libraries are chatty at INFO.
    for noisy in ("urllib3", "websockets", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
