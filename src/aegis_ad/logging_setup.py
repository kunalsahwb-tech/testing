"""Standard logging initialization for AEGIS-AD."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aegis_ad.settings import Settings


def setup_logging(settings: Settings | None = None) -> logging.Logger:
    """Configure standard root logging according to application settings."""
    level_name = "INFO"
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    log_file_path: Path | None = None

    if settings is not None:
        level_name = str(settings.logging.get("level", "INFO")).upper()
        fmt = str(settings.logging.get("format", fmt))
        log_file_str = settings.logging.get("log_file")
        if log_file_str:
            log_file_path = Path(log_file_str)

    numeric_level = getattr(logging, level_name, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Avoid adding duplicate handlers if already configured
    if not root_logger.handlers:
        formatter = logging.Formatter(fmt)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # File handler (optional)
        if log_file_path:
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_file_path), encoding="utf-8")
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
    else:
        # Update existing handlers level
        for handler in root_logger.handlers:
            handler.setLevel(numeric_level)

    logger = logging.getLogger("aegis_ad")
    logger.debug("Logging initialized at level %s", level_name)
    return logger
