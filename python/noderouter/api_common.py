# api_common.py

from joblib import Memory

from enum import IntEnum
from typing import TypedDict
import sys

from loguru import logger
import rustworkx as rx

import api_data_store as ds

memory = Memory(location=".cache", verbose=0)

# Constants
GREAT_OCEAN_TERRITORY = 5
OQUILLAS_EYE_KEY = 1727
SUPER_ROOT = 99999


class NodeType(IntEnum):
    normal = 0
    village = 1
    city = 2
    gate = 3
    farm = 4
    trade = 5
    collect = 6
    quarry = 7
    logging = 8
    dangerous = 9
    finance = 10
    fish_trap = 11
    minor_finance = 12
    monopoly_farm = 13
    craft = 14
    excavation = 15
    count = 16


class ResultDict(TypedDict):
    solution_graph: rx.PyDiGraph
    solution: list[int]
    objective: int
    duration: float


def set_logger(config: dict):
    log_level = config.get("logger", {}).get("level", "INFO")
    log_format = config.get("logger", {}).get("format", "<level>{message}</level>")
    logger.remove()
    logger.add(sys.stdout, colorize=True, level=log_level, format=log_format)
    return logger
