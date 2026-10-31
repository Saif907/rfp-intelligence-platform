"""Centralized logging configuration for the RFP Intelligence Platform.

Configures structured console logging with ISO-8601 timestamps,
module origins, and log levels.
"""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configures root logger with a standardized format and handler."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    # Avoid duplicate handlers if setup_logging is called multiple times
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    if not root_logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    else:
        for handler in root_logger.handlers:
            handler.setLevel(log_level)


def get_logger(name: str) -> logging.Logger:
    """Convenience factory returning a named logger."""
    return logging.getLogger(name)

