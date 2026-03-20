import logging
import os
from pathlib import Path

def get_file_logger(name: str) -> logging.Logger:
    """
    Returns a logger that writes only to log/backend/<name>.log (no terminal output).
    Usage: logger = get_file_logger(__name__)
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Remove all handlers to avoid duplicate logs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Ensure log directory exists
    log_dir = Path(__file__).parent.parent / "log" / "backend"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name.split('.')[-1]}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger
