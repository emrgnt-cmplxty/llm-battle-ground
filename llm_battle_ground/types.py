import os
from enum import Enum


class DataDirectories(Enum):
    DATASETS = (
        "datasets_debug"
        if os.getenv("LLM_BATTLE_GROUND_DEBUG") == "True"
        else "datasets"
    )
    RESULTS = (
        "results_debug"
        if os.getenv("LLM_BATTLE_GROUND_DEBUG") == "True"
        else "results"
    )


class Datasets(Enum):
    LEETCODE_FULL = "leetcode_full.csv"
    LEETCODE_DEMO = "leetcode_demo.csv"
