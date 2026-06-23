from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_dir: Path, debug: bool = False) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "mdc.log"),
        ],
    )
    return logging.getLogger("mdc")
