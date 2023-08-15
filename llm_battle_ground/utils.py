import os
import json


def read_jsonl(in_jsonl_path: str) -> list:
    with open(in_jsonl_path, "r") as json_file:
        json_list = list(json_file)

    jsonl_loaded = []
    for json_str in json_list:
        result = json.loads(json_str)
        jsonl_loaded.append(result)

    return jsonl_loaded


def get_root_fpath() -> str:
    return os.path.dirname(os.path.abspath(__file__))
