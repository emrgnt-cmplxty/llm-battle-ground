import json
import logging
import os

import numpy as np


def read_jsonl(in_jsonl_path: str) -> list[dict]:
    with open(in_jsonl_path, "r") as json_file:
        json_list = list(json_file)

    jsonl_loaded = []
    for json_str in json_list:
        result = json.loads(json_str)
        jsonl_loaded.append(result)

    return jsonl_loaded


def get_root_fpath() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def calc_similarity(
    embedding_a: np.ndarray, embedding_b: np.ndarray
) -> np.ndarray:
    dot_product = np.dot(embedding_a, embedding_b)
    magnitude_a = np.sqrt(np.dot(embedding_a, embedding_a))
    magnitude_b = np.sqrt(np.dot(embedding_b, embedding_b))
    return dot_product / (magnitude_a * magnitude_b)


def get_configured_logger(log_level: str) -> logging.Logger:
    log_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger()
