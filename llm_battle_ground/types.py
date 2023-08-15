from enum import Enum

# FIXME - Revert these to real defaults.


class DataDirectories(Enum):
    DATASETS = "datasets"
    RESULTS = "results"


class Datasets(Enum):
    LEETCODE_FULL = "leetcode_full.csv"
    LEETCODE_DEMO = "leetcode_demo.csv"
