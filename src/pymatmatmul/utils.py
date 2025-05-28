"""Utility functions for the project, not directly related to the main functionality."""
import os
import yaml
import logging
from rich.logging import RichHandler
from logging import Logger
from typing import List, Dict

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
            "Invalid logging level: %s. Available levels are: DEBUG, INFO, WARNING, ERROR, CRITICAL." % level
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



def validate_config(config: dict) -> None:
    """
    Validates the configuration dictionary.

    Args:
        config (dict): The configuration dictionary to validate.

    Raises:
        AttributeError: If any required keys are missing or invalid.
    """
    error_msg: str = '''
        Minimal configuration file requirements are not met.
        Please provide the following keys in your configuration file:
        - dimensions:
            A: [<int>, <int>]
            B: [<int>, <int>]
        - genRandomMatrices: <bool>
        '''

    if "dimensions" not in config or "genRandomMatrices" not in config:
        raise AttributeError(error_msg)

    dimensions: Dict[str, List[int, int]] = config["dimensions"]
    if not isinstance(dimensions, dict) or "A" not in dimensions or "B" not in dimensions:
        raise AttributeError(error_msg)

    assert isinstance(dimensions["A"], list) and len(dimensions["A"]) == 2, error_msg
    assert isinstance(dimensions["B"], list) and len(dimensions["B"]) == 2, error_msg
    assert all(isinstance(i, int) for i in dimensions["A"]), error_msg
    assert all(isinstance(i, int) for i in dimensions["B"]), error_msg

    if config["dimensions"]["A"][0] <= 0 or config["dimensions"]["A"][1] <= 0:
        raise AttributeError(
            "Matrix A dimensions must be positive integers."
        )

    if dimensions["A"][1] != dimensions["B"][0]:
        raise AttributeError(
            "Matrix A's columns (%d) must match Matrix B's rows (%d) for multiplication."
             % (dimensions["A"][1], dimensions["B"][0])
        )

    # temporarly, if genRandomMatrices is not set to True, say not implemented
    if not config["genRandomMatrices"]:
        raise NotImplementedError(
            "Reading matrices from file is not implemented yet. Please set genRandomMatrices to True."
        )

    # default values for optional parameters
    config.setdefault("logLevel", "INFO")
    config.setdefault("backend", "naive")
    config.setdefault("generationMin", 0.0)
    config.setdefault("generationMax", 1.0)
    config.setdefault("dtype", "float64")
    config.setdefault("profiler", "null")




    if not (isinstance(config["generationMin"], (int, float)) and isinstance(config["generationMax"], (int, float))):
        raise AttributeError(
            "generationMin and generationMax must be either int or float."
        )

    if config["generationMin"] >= config["generationMax"]:
        raise AttributeError(
            "generationMin must be less than generationMax."
        )



