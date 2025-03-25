"""Logging configuration for AxiomMCP."""

import logging
import sys
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def configure_logging(level: LogLevel = "INFO") -> None:
    """Configure logging for AxiomMCP."""
    logging.basicConfig(
        level=level, format="%(levelname)s: %(message)s", stream=sys.stdout, force=True
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)
