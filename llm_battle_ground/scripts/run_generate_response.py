import logging
import numpy as np
import os
import random

import openai
import pandas as pd
from evalplus.data import write_jsonl

random.seed(0)

from llm_battle_ground.completion_provider import CompletionProvider, RunMode
from llm_battle_ground.helpers import LeetCodeProcessor
from llm_battle_ground.scripts import common_arg_parser
from llm_battle_ground.types import DataDirectories, Datasets
from llm_battle_ground.utils import (
    get_configured_logger,
    get_root_fpath,
    read_jsonl,
)

IN_DIR = os.path.join(get_root_fpath(), DataDirectories.DATASETS.value)
IN_FILE_NAME = Datasets.LEETCODE_FULL.value
OUT_DIR = os.path.join(get_root_fpath(), DataDirectories.RESULTS.value)

PROVIDER = "openai"
RUN_MODE = "vanilla-zero-shot"
MODEL = "gpt-4-0613"
TEMPERATURE = 0.7
N_PASS = 1

# Define sample rates for different difficulties
DIFFICULTY_SAMPLE_RATES = {
    1: 0.25,  # Easy
    2: 0.50,  # Medium
    3: 1.0,  # Hard
}


OUTPUT_FILE_NAME = (
    "leetcode_generation__provider_eq_{PROVIDER}__{RUN_MODE}__filter_ez_eq_%s_filter_med_eq_%s_filter_hrd_eq_%s__model_eq_{MODEL}__temperature_eq_{TEMPERATURE}__n_pass_{N_PASS}_test.jsonl"
    % (
        DIFFICULTY_SAMPLE_RATES[1],
        DIFFICULTY_SAMPLE_RATES[2],
        DIFFICULTY_SAMPLE_RATES[3],
    )
)


def main(
    logger: logging.Logger,
    in_path: str,
    out_path: str,
    provider: str,
    run_mode: RunMode,
    model: str,
    temperature: float,
    n_pass: int,  # TODO - Implement this.
) -> None:
    logger.info(f"Loading dataset file from {in_path}.")
    dataset = pd.read_csv(in_path).sort_values(by=["frontend_question_id"])

    provider = CompletionProvider(
        run_mode=run_mode,
        model=model,
        temperature=temperature,
        provider=provider,
    )
    processor = LeetCodeProcessor()

    if os.path.exists(out_path):
        logger.info(f"Loading existing results from {out_path}.")
        outputs = read_jsonl(out_path)
    else:
        outputs = []

    observed_ids = {x["frontend_question_id"] for x in outputs}
    max_observed_id = max(observed_ids, default=0)

    for loc in range(len(dataset)):
        entry = dataset.iloc[loc]
        difficulty = entry["difficulty"]
        sample_rate = DIFFICULTY_SAMPLE_RATES.get(
            difficulty, 0.25
        )  # Use difficulty to determine sample rate, default to 25% if not-set (RARE)
        rnd_gen = random.random()

        if rnd_gen > sample_rate:
            logger.info("Continuing due to sample rate.")
            continue

        # Skipping past previously processed problems
        frontend_question_id = dataset.iloc[loc]["frontend_question_id"]
        if frontend_question_id in observed_ids:
            logger.info(
                "Continuing because this problem has been processed previously."
            )
            continue

        # Since we use rng, we need to skip past old max when re-loading data
        if frontend_question_id <= max_observed_id:
            logger.info(
                "Continuing because this problem has been processed previously."
            )
            continue
        else:
            if max_observed_id != 0:
                logger.info(
                    "Processing first example, resetting max_observed_id to 0."
                )
                max_observed_id = 0

        # Generating input for completion
        task_input = processor.clean_html_content(entry["raw_content"])
        code_snippet = entry["python3_snippet"]
        raw_response = provider.get_completion(
            task_input=task_input, code_snippet=code_snippet
        )
        result = {
            "task_input": task_input,
            "code_snippet": code_snippet,
            "raw_response": raw_response,
            "n_pass": n_pass,
            "difficulty": difficulty,
            "frontend_question_id": frontend_question_id,
            "loc": loc,
        }
        outputs.append(result)
        # write_jsonl(out_path, outputs)
        break
    return outputs


if __name__ == "__main__":
    parser = common_arg_parser()

    args = parser.parse_args()
    logger = get_configured_logger(__name__, args.log_level)

    openai.api_key = os.getenv("OPENAI_API_KEY_LOCAL", "")

    in_path = os.path.join(
        args.in_dir or IN_DIR, args.in_file_name or IN_FILE_NAME
    )

    out_file_name = args.out_file_name or OUTPUT_FILE_NAME.format(
        PROVIDER=args.provider,
        MODEL=args.model,
        TEMPERATURE=args.temperature,
        N_PASS=args.n_pass,
        RUN_MODE=args.run_mode or RUN_MODE,
    )
    out_path = os.path.join(args.out_dir or OUT_DIR, out_file_name)

    outputs = main(
        logger,
        in_path,
        out_path,
        args.provider or PROVIDER,
        RunMode(args.run_mode or RUN_MODE),
        args.model or MODEL,
        args.temperature or TEMPERATURE,
        args.n_pass or N_PASS,
    )
    write_jsonl(out_path, outputs)
