"""
Centralized logging configuration for the Trading Application.
"""

import logging
import logging.config
from config.settings import LOGGING_CONFIG, LOGS_DIR
import os

def setup_logging():
    """Initialize logging configuration."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized successfully")
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module."""
    return logging.getLogger(name)

# Initialize logging on module import
setup_logging()
