# api_common.py

import sys

from joblib import Memory
from loguru import logger

memory = Memory(location=".cache", verbose=0)

# This is the max budget for which an optimal solution is available
# in the data store from the EmpireOptimizer.
MAX_BUDGET = 550


def set_logger(config: dict):
    log_level = config.get("logger", {}).get("level", "INFO")
    log_format = config.get("logger", {}).get("format", "<level>{message}</level>")
    logger.remove()
    logger.add(sys.stdout, colorize=True, level=log_level, format=log_format)
    return logger
