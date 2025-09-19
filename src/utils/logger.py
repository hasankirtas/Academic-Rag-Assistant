"""
Logging Utilities Module

Provides consistent logging across the project.
"""

import logging
import sys
from typing import Optional
from pathlib import Path


def get_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True,
    detailed: bool = False
) -> logging.Logger:
    """
    Creates and configures a logger.
    
    Args:
        name: Logger name (default: module name)
        level: Logging level
        log_file: Path to log file (None for console only)
        console_output: Whether to log to console
        detailed: Use detailed format (filename and line number)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name or __name__)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    if detailed:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def setup_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Alias for backward compatibility.
    """
    return get_logger(name)


def setup_project_logging(
    level: int = logging.INFO,
    log_dir: str = "logs",
    console_output: bool = True,
    detailed: bool = False
):
    """
    Configures project-wide logging.
    
    Args:
        level: Global logging level
        log_dir: Directory for log files
        console_output: Whether to log to console
        detailed: Use detailed format (filename and line number)
    """
    main_log_file = Path(log_dir) / "app.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    if detailed:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    log_path = Path(main_log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(main_log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    return root_logger
