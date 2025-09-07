import logging
from typing import Optional


def get_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
    to_file: Optional[str] = None
) -> logging.Logger:
    """
    Create and configure a logger instance.

    Args:
        name (Optional[str]): Logger name (default: module name).
        level (int): Logging level (default: INFO).
        to_file (Optional[str]): If provided, logs will also be written to this file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name or __name__)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
        )

        # Console handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # Optional file handler
        if to_file:
            file_handler = logging.FileHandler(to_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        logger.setLevel(level)
        logger.propagate = False

    return logger


def setup_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Alias for get_logger to maintain compatibility with existing code.
    """
    return get_logger(name)
