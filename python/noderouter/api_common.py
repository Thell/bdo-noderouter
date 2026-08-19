# api_common.py

import sys

from joblib import Memory
from loguru import logger

memory = Memory(location=".cache", verbose=0)

# This is the max budget for which an optimal solution is available
# in the data store from the EmpireOptimizer.
MAX_BUDGET = 550

PAYLOAD_WEIGHT_KEY: str = "need_exploration_point"
HYPERNODE_CONTENTS_KEY = "collapsed_nodes"

# Matches Rust u64::MAX representation for INF
INT_INF = 18446744073709551615


def set_logger(config: dict):
    """Sets logger accoring to the config dict's 'logger' options.

    Defaults to short format INFO logging to stdout.
    """
    log_level = config.get("logger", {}).get("level", "INFO")
    log_format = config.get("logger", {}).get("format", "<level>{message}</level>")
    logger.remove()
    logger.add(sys.stdout, colorize=True, level=log_level, format=log_format)
    return logger
