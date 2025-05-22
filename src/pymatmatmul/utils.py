"""Utility functions for the project, not directly related to the main functionality."""
import os
import yaml
import logging
from rich.logging import RichHandler
from logging import Logger


def setup_logger(level: str = "INFO") -> Logger:
    """
    Sets up a logger with a specified logging level.

    Args:
    - level (str): The logging level to set. Default is "INFO".
      available levels are: DEBUG, INFO, WARNING, ERROR, CRITICAL.

    Returns:
    - Logger: A configured logger instance.
    """
    if level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        raise ValueError(
            f"Invalid logging level: {level}. Available levels are: DEBUG, INFO, WARNING, ERROR, CRITICAL."
        )
    logging.basicConfig(
        format="%(message)s",
        datefmt="[%X]",
        level=level,
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    logger = logging.getLogger("rich")
    return logger


def read_config(file: str) -> dict:
    """
    Reads a YAML configuration file and returns its contents as a dictionary.

    Args:
    - file (str): The name of the YAML configuration file (without the extension).

    Returns:
    - dict: A dictionary containing the key-value pairs from the YAML file.
    """
    filepath: str = os.path.abspath(f'{file}.yaml')

    with open(filepath, 'r', encoding='utf8') as stream:
        kwargs = yaml.safe_load(stream)
    return kwargs

