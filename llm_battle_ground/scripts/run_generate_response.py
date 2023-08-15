import argparse
import logging
import os
import re

import dotenv
import numpy as np
import openai
import pandas as pd
from automata.llm import OpenAIEmbeddingProvider
from evalplus.data import write_jsonl
import random

random.seed(0)

from llm_battle_ground.completion_provider import CompletionProvider, RunMode
from llm_battle_ground.utils import read_jsonl

dotenv.load_dotenv()
script_directory = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_PATH = os.path.join(script_directory, "..", "datasets")
DEFAULT_INPUT_FILE_NAME = "leetcode.csv"
DEFAULT_OUTPUT_PATH = os.path.join(script_directory, "..", "results")

NUM_INPUT_EXAMPLES = 10
NUM_OUTPUT_EXAMPLES = 5
MODEL = "gpt-4-0613"
PROVIDER = "openai"
TEMPERATURE = 0.7
RUN_MODE = "vanilla-zero-shot"
N_PASS = 1

# Define sample rates for different difficulties
DIFFICULTY_SAMPLE_RATES = {
    1: 0.25,  # Easy
    2: 0.50,  # Medium
    3: 1.0,  # Hard
}

OUTPUT_FILE_NAME = (
    "leetcode_{RUN_MODE}__filter_ez_eq_%s_filter_med_eq_%s_filter_hrd_eq_%s__model_eq_{MODEL}__temperature_eq_{TEMPERATURE}__n_pass_{N_PASS}.jsonl"
    % (
        DIFFICULTY_SAMPLE_RATES[1],
        DIFFICULTY_SAMPLE_RATES[2],
        DIFFICULTY_SAMPLE_RATES[3],
    )
)


def main(
    logger: logging.Logger,
    in_fpath: str,
    out_fpath: str,
    run_mode: RunMode,
    model: str,
    provider: str,
    temperature: float,
    n_pass: int,
) -> None:
    logger.info(f"Loading dataset file from {in_fpath}.")
    dataset = pd.read_csv(in_fpath).sort_values(by=["frontend_question_id"])

    provider = CompletionProvider(
        run_mode=run_mode,
        model=model,
        temperature=temperature,
        provider=provider,
    )

    if os.path.exists(out_fpath):
        logger.info(f"Loading existing results from {out_fpath}.")
        outputs = read_jsonl(out_fpath)
    else:
        outputs = []
    ids = {x["frontend_question_id"] for x in outputs}

    for loc in range(0, len(dataset)):
        entry = dataset.iloc[loc]
        difficulty = entry["difficulty"]
        print("difficulty = ", difficulty)
        sample_rate = DIFFICULTY_SAMPLE_RATES.get(
            difficulty, 0.25
        )  # Use difficulty to determine sample rate
        rnd_gen = random.random()
        print("rnd_gen = ", rnd_gen)
        if rnd_gen > sample_rate:
            continue

        frontend_question_id = dataset.iloc[loc]["frontend_question_id"]
        if frontend_question_id in ids:
            continue

        end_snippet = "....\n"
        task_input = f"{entry.cleaned_snippet.replace(end_snippet,'.')} {entry.cleaned_forward_completion}"
        code_prompt = entry["python3_snippet"]
        raw_response = provider.get_completion(
            task_input=task_input, code_prompt=code_prompt
        )
        if "```python" in raw_response:
            cleaned_response = raw_response.split("```python")[1]
            cleaned_response = cleaned_response.split("```")[0]
        elif "```" in raw_response:
            cleaned_response = raw_response.split("```")[1]
            cleaned_response = cleaned_response.split("```")[0]
        else:
            cleaned_response = raw_response
        result = {
            "task_input": task_input,
            "code_prompt": code_prompt,
            "raw_response": raw_response,
            "cleaned_response": cleaned_response,
            "n_pass": n_pass,
            "difficulty": difficulty,
            "frontend_question_id": frontend_question_id,
            "loc": loc,
        }
        outputs.append(result)
        write_jsonl(out_fpath, outputs)

    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level, e.g. DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )
    parser.add_argument(
        "--in-path",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help="Directory path to the input dataset",
    )
    parser.add_argument(
        "--in-file-name",
        type=str,
        default=DEFAULT_INPUT_FILE_NAME,
        help="Filename for input file.",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save the results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL,
        help="Model to use for experiment",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=PROVIDER,
        help="Provider to use for experiment",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help="Temperature to use for experiment",
    )
    parser.add_argument(
        "--run-mode",
        type=str,
        default=RUN_MODE,
        help="Which run mode to use for experiment",
    )
    parser.add_argument(
        "--n-pass",
        type=int,
        default=N_PASS,
        help="Number of passes.",
    )

    args = parser.parse_args()
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger()

    openai.api_key = os.getenv("OPENAI_API_KEY_LOCAL", "")

    in_fpath = os.path.join(args.in_path, args.in_file_name)

    out_fname = OUTPUT_FILE_NAME.format(
        MODEL=args.model,
        TEMPERATURE=args.temperature,
        N_PASS=args.n_pass,
        RUN_MODE=args.run_mode,
    )
    out_fpath = os.path.join(args.out_path, out_fname)

    outputs = main(
        logger=logger,
        in_fpath=in_fpath,
        out_fpath=out_fpath,
        run_mode=RunMode(args.run_mode),
        model=args.model,
        provider=args.provider,
        temperature=args.temperature,
        n_pass=args.n_pass,
    )
    write_jsonl(out_fpath, outputs)
