import argparse
import logging
import os

import dotenv
import numpy as np
import openai
import pandas as pd
from automata.llm import OpenAIEmbeddingProvider
from evalplus.data import write_jsonl

from llm_battle_ground.completion_provider import CompletionProvider, RunMode

dotenv.load_dotenv()
script_directory = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_PATH = os.path.join(script_directory, "..", "datasets")
DEFAULT_INPUT_FILE_NAME = "leetcode.csv"
DEFAULT_OUTPUT_PATH = os.path.join(script_directory, "..", "results")

NUM_INPUT_EXAMPLES = 10
NUM_OUTPUT_EXAMPLES = 5
STEP_SIZE = 50
MODEL = "gpt-4-0613"
TEMPERATURE = 0.7

OUTPUT_FILE_NAME = "leetcode_similarity__num_input_eq_{NUM_INPUT_EXAMPLES}__num_output_eq_{NUM_OUTPUT_EXAMPLES}__step_size_eq_{STEP_SIZE}__model_eq_{MODEL}__temperature_eq_{TEMPERATURE}.jsonl"


def calc_similarity(embedding_a, embedding_b):
    dot_product = np.dot(embedding_a, embedding_b)
    magnitude_a = np.sqrt(np.dot(embedding_a, embedding_a))
    magnitude_b = np.sqrt(np.dot(embedding_b, embedding_b))
    return dot_product / (magnitude_a * magnitude_b)


def clean_response(response):
    replacements = [f"Example {i}:" for i in range(1, 25)] + [
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
    response, embedding_provider=OpenAIEmbeddingProvider()
):
    cleaned_response = clean_response(response)
    return embedding_provider.build_embedding_vector(cleaned_response)


def main(
    logger: logging.Logger,
    in_fpath: str,
    num_input_examples: int,
    num_forward_examples: int,
    step_size: int,
    model: str,
    temperature: float,
) -> None:
    logger.info(f"Loading dataset file from {in_fpath}.")
    dataset = pd.read_csv(in_fpath).sort_values(by=["frontend_question_id"])

    provider = CompletionProvider(
        run_mode=RunMode.VANILLA,
        model=model,
        temperature=temperature,
    )

    outputs = []
    for iloc in range(num_input_examples + 1, len(dataset), step_size):
        if iloc + num_forward_examples >= len(dataset):
            break
        entry_suffix = "....\n"
        input_context = "\n".join(
            f"Example {counter + 1}: {dataset.iloc[i]['cleaned_snippet'].replace(entry_suffix, '. ')}"
            for counter, i in enumerate(range(iloc - num_input_examples, iloc))
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

        prompt = provider.get_formatted_instruction(
            input_context, num_forward_examples
        )
        response = provider.generate_vanilla_completion(prompt)

        cleaned_expected_response = clean_response(expected_response)
        embedded_cleaned_expected_response = build_embedded_response(
            cleaned_expected_response
        )
        cleaned_response = clean_response(response)
        embedded_clean_response = build_embedded_response(cleaned_response)

        similarity = calc_similarity(
            embedded_cleaned_expected_response, embedded_clean_response
        )
        logger.info(
            f"Expected Response = {expected_response}\nResponse = {response}"
        )

        logger.info(f"Similarity = {similarity}")

        result = {
            "input_context": input_context,
            "expected_response": expected_response,
            "cleaned_expected_response": cleaned_expected_response,
            "response": response,
            "cleaned_response": cleaned_response,
            "similarity": similarity,
            "iloc": iloc,
            "frontend_question_id": dataset.iloc[iloc + num_forward_examples][
                "frontend_question_id"
            ],
        }
        outputs.append(result)

        # Write after each step, in case progress is halted.
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
        "--num-input-examples",
        type=str,
        default=NUM_INPUT_EXAMPLES,
        help="Number of preceeding examples to include in N-shot context.",
    )
    parser.add_argument(
        "--num-output-examples",
        type=str,
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
        type=str,
        default=STEP_SIZE,
        help="Iteration step size when running the experiment",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL,
        help="Iteration step size when running the experiment",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help="Iteration step size when running the experiment",
    )

    args = parser.parse_args()
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger()

    openai.api_key = os.getenv("OPENAI_API_KEY_LOCAL", "")

    outputs = main(
        logger,
        os.path.join(args.in_path, args.in_file_name),
        args.num_input_examples,
        args.num_output_examples,
        args.step_size,
        args.model,
        args.temperature,
    )

    out_fname = OUTPUT_FILE_NAME.format(
        NUM_INPUT_EXAMPLES=NUM_INPUT_EXAMPLES,
        NUM_OUTPUT_EXAMPLES=NUM_OUTPUT_EXAMPLES,
        STEP_SIZE=STEP_SIZE,
        MODEL=MODEL,
        TEMPERATURE=TEMPERATURE,
    )
    out_fpath = os.path.join(args.out_path, out_fname)
    write_jsonl(out_fpath, outputs)
