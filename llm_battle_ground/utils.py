import json
import logging
import os
import re

import pandas as pd
import numpy as np


def read_jsonl(in_jsonl_path: str) -> list[dict]:
    with open(in_jsonl_path, "r") as json_file:
        json_list = list(json_file)

    jsonl_loaded = []
    for json_str in json_list:
        result = json.loads(json_str)
        jsonl_loaded.append(result)

    return jsonl_loaded


def write_df_to_jsonl(df: pd.DataFrame, out_jsonl_path: str):
    with open(out_jsonl_path, "w") as file:
        for index, row in df.iterrows():
            json_record = json.dumps(row.to_dict())
            file.write(json_record + "\n")
    print(f"JSONL file saved to {out_jsonl_path}.")


def get_root_fpath() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def calc_similarity(
    embedding_a: np.ndarray, embedding_b: np.ndarray
) -> np.ndarray:
    dot_product = np.dot(embedding_a, embedding_b)
    magnitude_a = np.sqrt(np.dot(embedding_a, embedding_a))
    magnitude_b = np.sqrt(np.dot(embedding_b, embedding_b))
    return dot_product / (magnitude_a * magnitude_b)


def get_configured_logger(name: str, log_level: str) -> logging.Logger:
    log_level = getattr(logging, log_level.upper(), "INFO")
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(name)


def extract_code(raw_response: str) -> str:
    def _extract_unformatted(raw_response):
        # Extract the class definition as before
        class_definition_match = re.search(
            r"class\s\S*:\s*.*?(?=\n\n|$)", raw_response, re.DOTALL
        )
        class_definition = (
            class_definition_match[0] if class_definition_match else None
        )

        # Find the position of the class definition in the raw_response
        class_position = (
            class_definition_match.start() if class_definition_match else -1
        )

        # Extract the lines before the class definition
        lines_before_class = raw_response[:class_position].strip().splitlines()

        # Extract the import statements by filtering lines that start with 'import' or 'from'
        import_statements = [
            line
            for line in lines_before_class
            if line.startswith(("import", "from"))
        ]

        # Combine the import statements and the class definition
        return "\n".join(import_statements + [class_definition])

    if "```python" in raw_response:
        cleaned_response = raw_response.split("```python")[1]
        return cleaned_response.split("```")[0]
    elif "```" in raw_response:
        cleaned_response = raw_response.split("```")[1]
        return cleaned_response.split("```")[0]
    else:
        return _extract_unformatted(raw_response)
