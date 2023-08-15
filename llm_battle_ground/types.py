from enum import Enum


class DataDirectories(Enum):
    DATASETS = "datasets"
    RESULTS = "results"


class Datasets(Enum):
    LEETCODE_FULL = "leetcode_full.csv"
    LEETCODE_DEMO = "leetcode_demo.csv"
