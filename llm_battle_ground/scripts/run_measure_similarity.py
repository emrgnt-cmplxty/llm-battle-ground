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

from llm_battle_ground.completion_provider import CompletionProvider, RunMode
from llm_battle_ground.utils import read_jsonl

dotenv.load_dotenv()
script_directory = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_PATH = os.path.join(script_directory, "..", "datasets")
DEFAULT_INPUT_FILE_NAME = "leetcode.csv"
DEFAULT_OUTPUT_PATH = os.path.join(script_directory, "..", "results")

STEP_SIZE = 25
MODEL = "gpt-4-0613"
TEMPERATURE = 0.7
N_PASS = 1
MODE = "vanilla-zero-shot"
NUM_INPUT_EXAMPLES = 1
NUM_OUTPUT_EXAMPLES = 1

OUTPUT_FILE_NAME = "leetcode_{MODE}_step_size_eq_{STEP_SIZE}__model_eq_{MODEL}__temperature_eq_{TEMPERATURE}__n_pass_{N_PASS}.jsonl"


def calc_similarity(
    embedding_a: np.ndarray, embedding_b: np.ndarray
) -> np.ndarray:
    dot_product = np.dot(embedding_a, embedding_b)
    magnitude_a = np.sqrt(np.dot(embedding_a, embedding_a))
    magnitude_b = np.sqrt(np.dot(embedding_b, embedding_b))
    return dot_product / (magnitude_a * magnitude_b)


def parse_text(text: str) -> dict:
    examples = re.findall(
        r"Example (\d+):\s+LeetCode Problem #(\d+)\s+Title: (.+?)\nDescription:\n(.+?)(?=\nExample|\n$)",
        text,
        re.S,
    )
    return {
        int(example_num): {
            "problem_number": int(problem_num),
            "title": title,
            "description": description,
        }
        for example_num, problem_num, title, description in examples
    }


def evaluate_common_problems(dict1: dict, dict2: dict) -> dict:
    common_problems = {}
    for key, value in dict2.items():
        if any(
            value["problem_number"] == item["problem_number"]
            for item in dict1.values()
        ):
            common_problems[key] = value
    return common_problems


def unpack_to_string(data: dict) -> str:
    result = ""
    for example_num, details in data.items():
        result += f"Example {example_num}:\nLeetCode Problem #{details['problem_number']}\nTitle: {details['title']}\nDescription:\n{details['description']}\n\n"
    return result


def clean_response(response: str) -> str:
    replacements = [f"Example {i}:" for i in range(1, 1_000)] + [
        "Input:",
        "Output:",
        "Explanation:",
        "Constraints:",
        "Title: ",
        "Description:",
    ]
    for replace_string in replacements:
        response = response.replace(replace_string, "")
    return "\n".join(
        line for line in response.split("\n") if "LeetCode Problem" not in line
    )


def build_embedded_response(
    response: str, embedding_provider=OpenAIEmbeddingProvider()
) -> np.ndarray:
    cleaned_response = clean_response(response)
    return embedding_provider.build_embedding_vector(cleaned_response)


def main(
    logger: logging.Logger,
    in_fpath: str,
    out_fpath: str,
    num_input_examples: int,
    num_forward_examples: int,
    step_size: int,
    model: str,
    temperature: float,
    n_pass: int,
    buffer: int = 10,
) -> None:
    logger.info(f"Loading dataset file from {in_fpath}.")
    dataset = pd.read_csv(in_fpath).sort_values(by=["frontend_question_id"])

    provider = CompletionProvider(
        run_mode=RunMode("similarity"),
        model=model,
        temperature=temperature,
    )

    if os.path.exists(out_fpath):
        logger.info(f"Loading existing results from {out_fpath}.")
        outputs = read_jsonl(out_fpath)
        start_index = (
            outputs[-1]["iloc"] + step_size
        )  # Assuming 'iloc' is the correct index to continue from
    else:
        outputs = []
        start_index = num_input_examples + 1

    for iloc in range(start_index, len(dataset), step_size):
        try:
            if iloc + num_forward_examples >= len(dataset):
                break
            entry_suffix = "....\n"
            input_context = "\n".join(
                f"Example {counter + 1}: {dataset.iloc[i]['cleaned_snippet'].replace(entry_suffix, '. ')}"
                for counter, i in enumerate(
                    range(iloc - num_input_examples, iloc)
                )
            )

            expected_response = "\n".join(
                f"Example {counter + 1}: {dataset.iloc[i]['cleaned_snippet'].replace(entry_suffix, '. ')}"
                for counter, i in enumerate(
                    range(
                        iloc,
                        iloc + num_forward_examples,
                    ),
                    start=num_input_examples,
                )
            )

            cleaned_expected_response = clean_response(expected_response)
            embedded_cleaned_expected_response = build_embedded_response(
                cleaned_expected_response
            )

            prompt = provider.get_formatted_instruction(
                task_input=input_context, num_forward_examples=num_forward_examples + buffer
            )

            result = {
                "input_context": input_context,
                "expected_response": expected_response,
                "iloc": iloc,
                "frontend_question_id": dataset.iloc[
                    iloc + num_forward_examples
                ]["frontend_question_id"],
                "n_pass": n_pass,
            }

            similarity = 0
            for _ in range(n_pass):
                while True:
                    response = provider.generate_vanilla_completion(prompt)
                    if "as an AI language model" not in response:
                        break

                parsed_expected_response = parse_text(expected_response)
                parsed_response = parse_text(response)
                joined_response = evaluate_common_problems(
                    parsed_expected_response, parsed_response
                )

                cleaned_response = clean_response(
                    unpack_to_string(joined_response)
                )
                embedded_clean_response = build_embedded_response(
                    cleaned_response
                )
                latest_similarity_calc = calc_similarity(
                    embedded_cleaned_expected_response,
                    embedded_clean_response,
                )
                logger.info(f"Latest Similarity = {latest_similarity_calc}")
                similarity += latest_similarity_calc / n_pass
                result["response"] = response
                result["cleaned_response"] = clean_response(response)
                result["similarity"] = similarity

            logger.info(
                f"Expected Response = {expected_response}\nResponse = {response}"
            )

            logger.info(f"Similarity = {similarity}")

            outputs.append(result)

            # Write after each step, in case progress is halted.
            write_jsonl(out_fpath, outputs)
            break
        except:
            logger.exception(f"Failed to process {iloc}.")
            continue

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
        "--num-input-examples",
        type=int,
        default=NUM_INPUT_EXAMPLES,
        help="Number of preceeding examples to include in N-shot context.",
    )
    parser.add_argument(
        "--num-output-examples",
        type=int,
        default=NUM_OUTPUT_EXAMPLES,
        help="Number of preceeding examples to include in N-shot context.",
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
        "--step-size",
        type=int,
        default=STEP_SIZE,
        help="Iteration step size when running the experiment",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL,
        help="Model to use for experiment",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help="Temperature to use for experiment",
    )
    parser.add_argument(
        "--n-pass",
        type=int,
        default=N_PASS,
        help="Number of calculations to make for similarity.",
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
        MODE=MODE,
        NUM_INPUT_EXAMPLES=args.num_input_examples,
        NUM_OUTPUT_EXAMPLES=args.num_output_examples,
        STEP_SIZE=args.step_size,
        MODEL=args.model,
        TEMPERATURE=args.temperature,
        N_PASS=args.n_pass,
    )
    out_fpath = os.path.join(args.out_path, out_fname)

    outputs = main(
        logger,
        in_fpath,
        out_fpath,
        args.num_input_examples,
        args.num_output_examples,
        args.step_size,
        args.model,
        args.temperature,
        args.n_pass,
    )
    write_jsonl(out_fpath, outputs)
