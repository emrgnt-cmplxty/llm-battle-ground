import logging
import os
import re

import numpy as np
import openai
import pandas as pd
from automata.llm import OpenAIEmbeddingProvider
from evalplus.data import write_jsonl
from bs4 import BeautifulSoup

from llm_battle_ground.completion_provider import CompletionProvider, RunMode
from llm_battle_ground.types import Datasets, DataDirectories
from llm_battle_ground.utils import (
    get_configured_logger,
    get_root_fpath,
    calc_similarity,
    read_jsonl,
)
from llm_battle_ground.scripts import common_arg_parser

# Pathing
IN_DIR = os.path.join(get_root_fpath(), DataDirectories.DATASETS.value)
IN_FILE_NAME = Datasets.LEETCODE_FULL.value
OUT_DIR = os.path.join(get_root_fpath(), DataDirectories.RESULTS.value)
OUT_FILE_NAME = "leetcode_{RUN_MODE}_step_size_eq_{STEP_SIZE}__num_input_examples_eq_{NUM_OUTPUT_EXAMPLES}__num_output_examples_eq_{NUM_OUTPUT_EXAMPLES}__buffer_eq_{BUFFER}__model_eq_{MODEL}__temperature_eq_{TEMPERATURE}__n_pass_{N_PASS}.jsonl"

# Local constants
NUM_INPUT_EXAMPLES = 20
NUM_OUTPUT_EXAMPLES = 20
STEP_SIZE = 40
BUFFER = 10
# Default input constants
MODEL = "gpt-4-0613"
TEMPERATURE = 0.7
N_PASS = 1
RUN_MODE = "similarity"


class LeetCodeProcessor:
    """Class to process LeetCode problems."""

    @staticmethod
    def clean_html_content(html_content: str) -> str:
        """Clean the HTML content of a LeetCode problem description"""
        if not isinstance(html_content, str):
            return html_content
        # Parse the HTML content
        soup = BeautifulSoup(html_content, "html.parser")

        # Convert HTML to text with newline characters
        text_content = soup.get_text(separator="\n", strip=True)

        # Replace scientific notation numbers
        cleaned_content = re.sub(
            r"(\b10)\s+(\d+\b)", r"\1e\2", text_content
        )  # Replace "10 4" with "10e4"

        # Specific handling for power notation
        cleaned_content = re.sub(
            r"(\b\d+)\s+(\d+\b)", r"\1^\2", cleaned_content
        )

        # Spaces around operators
        cleaned_content = re.sub(r"(\s*<=\s*)", " <= ", cleaned_content)

        # Replace specific patterns with newline characters
        cleaned_content = cleaned_content.replace(
            " . ", ".\n"
        )  # Newline after periods

        # Specific handling for "O ( n 2 )" pattern
        cleaned_content = cleaned_content.replace("O ( n^2 )", "O(n^2)")

        # Replace unnecessary characters and whitespace
        cleaned_content = re.sub(
            r"\s+", " ", cleaned_content
        )  # Collapse multiple whitespace

        # Remove spaces after commas and before periods
        cleaned_content = re.sub(r"\s*,\s*", ", ", cleaned_content)
        cleaned_content = re.sub(r"\s*\.\s*", ". ", cleaned_content)

        # Specific handling for .length accessor
        cleaned_content = cleaned_content.replace(" . length", ".length")

        # Remove leading asterisks from problem numbers
        cleaned_content = re.sub(r"\*+(\d+)\.?", r"\1.", cleaned_content)

        # Remove trailing asterisks
        cleaned_content = re.sub(r"\*+$", "", cleaned_content)

        # Cleanup .length accessor
        cleaned_content = cleaned_content.replace(". length", ".length")
        # Cleanup leading asterisks (in case cmd above missfired)
        cleaned_content = cleaned_content.replace("***", "")
        # Cleanup period formatting for programmatic statements.
        cleaned_content = cleaned_content.replace("'. '", ".")

        return cleaned_content.strip()

    @staticmethod
    def parse_text(text: str) -> dict:
        """Parse the text into a dictionary."""
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

    @staticmethod
    def clean_response(response: str) -> str:
        """Clean the response in order to reduce the baseline overlap between the expected and actual completion."""
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
            line
            for line in response.split("\n")
            if "LeetCode Problem" not in line
        )

    @staticmethod
    def augment_dataset(dataset: pd.DataFrame) -> None:
        dataset["example_header"] = dataset["raw_content"].apply(
            lambda x: (
                x.split("</p>")[0] + "</p>" if isinstance(x, str) else ""
            )
        )
        dataset["example_statement"] = dataset.apply(
            lambda x: f"\nLeetCode Problem #{int(x['frontend_question_id'])}\nTitle: {x['question_title']}\nDescription:\n{LeetCodeProcessor.clean_html_content(x['example_header'])}",
            axis=1,
        )


class ResponseEvaluator:
    """Class to evaluate LLM responses."""

    def __init__(
        self,
        embedding_provider: OpenAIEmbeddingProvider = OpenAIEmbeddingProvider(),
    ) -> None:
        self.embedding_provider = embedding_provider

    @staticmethod
    def evaluate_common_problems(dict1: dict, dict2: dict) -> dict:
        """Evaluate the common problems between two dictionaries."""
        return {
            key: value
            for key, value in dict2.items()
            if any(
                value["problem_number"] == item["problem_number"]
                for item in dict1.values()
            )
        }

    @staticmethod
    def unpack_to_string(data: dict) -> str:
        """Unpacks an appropriately formatted LeetCode problem dictionary into a string."""
        return "".join(
            f"Example {example_num}:\nLeetCode Problem #{details['problem_number']}\nTitle: {details['title']}\nDescription:\n{details['description']}\n\n"
            for example_num, details in data.items()
        )

    def clean_and_build_embedded_response(self, response: str) -> np.ndarray:
        """Helper function to clean and build embedded response."""
        cleaned_response = LeetCodeProcessor.clean_response(response)
        return self.embedding_provider.build_embedding_vector(cleaned_response)


class SimilarityExperiment:
    def __init__(self, logger: logging.Logger, args):
        self.logger = logger
        self.args = args
        self.args.model = self.args.model or MODEL
        self.args.temperature = self.args.temperature or TEMPERATURE
        self.args.n_pass = self.args.n_pass or N_PASS
        self.processor = LeetCodeProcessor()
        self.evaluator = ResponseEvaluator()
        self.provider = CompletionProvider(
            run_mode=RunMode(RUN_MODE),
            model=self.args.model,
            temperature=self.args.temperature,
        )

        self.output_file_name = (
            self.args.out_file_name
            or OUT_FILE_NAME.format(
                NUM_INPUT_EXAMPLES=self.args.num_input_examples,
                NUM_OUTPUT_EXAMPLES=self.args.num_output_examples,
                BUFFER=self.args.buffer,
                STEP_SIZE=self.args.step_size,
                MODEL=self.args.model,
                TEMPERATURE=self.args.temperature,
                RUN_MODE=RUN_MODE,
                N_PASS=self.args.n_pass,
            ).replace("-", "_")
        )

        self.out_dir = os.path.join(
            self.args.out_dir or OUT_DIR,
            self.output_file_name,
        )
        print("self.out_dir = ", self.out_dir)

    def run(self) -> list:
        """Run the experiment."""

        in_dir = os.path.join(
            self.args.in_dir or IN_DIR,
            self.args.in_file_name or IN_FILE_NAME,
        )

        self.logger.info(f"Loading dataset file from {in_dir}.")
        dataset = pd.read_csv(in_dir).sort_values(by=["frontend_question_id"])
        self.processor.augment_dataset(dataset)

        if os.path.exists(self.out_dir):
            self.logger.info(f"Loading existing results from {self.out_dir}.")
            outputs = read_jsonl(self.out_dir)
            if len(outputs) > 0:
                start_index = outputs[-1]["iloc"] + self.args.step_size
            else:
                start_index = self.args.num_input_examples + 1
        else:
            outputs = []
            start_index = self.args.num_input_examples + 1

        self._generate_similarity(dataset, start_index, self.provider, outputs)
        write_jsonl(self.out_dir, outputs)
        return outputs

    def _generate_similarity(
        self,
        dataset: pd.DataFrame,
        start_index: int,
        provider: CompletionProvider,
        outputs: list,
    ) -> None:
        for iloc in range(start_index, len(dataset), self.args.step_size):
            try:
                if iloc + self.args.num_output_examples >= len(dataset):
                    break
                input_context = "\n".join(
                    f"Example {counter + 1}: {dataset.iloc[i]['example_statement']}"
                    for counter, i in enumerate(
                        range(iloc - self.args.num_input_examples, iloc)
                    )
                )

                expected_response = "\n".join(
                    f"Example {counter + 1}: {dataset.iloc[i]['example_statement']}"
                    for counter, i in enumerate(
                        range(
                            iloc,
                            iloc + self.args.num_output_examples,
                        ),
                        start=self.args.num_input_examples,
                    )
                )

                cleaned_expected_response = self.processor.clean_response(
                    expected_response
                )
                embedded_cleaned_expected_response = (
                    self.evaluator.clean_and_build_embedded_response(
                        cleaned_expected_response
                    )
                )

                prompt = provider.get_formatted_instruction(
                    task_input=input_context,
                    num_forward_examples=self.args.num_output_examples
                    + self.args.buffer,
                )

                result = {
                    "input_context": input_context,
                    "expected_response": expected_response,
                    "iloc": iloc,
                    "frontend_question_id": dataset.iloc[
                        iloc + self.args.num_output_examples
                    ]["frontend_question_id"],
                    "n_pass": self.args.n_pass,
                }

                similarity = 0
                for _ in range(self.args.n_pass):
                    while True:
                        response = provider.generate_vanilla_completion(prompt)
                        if "as an AI language model" not in response:
                            break

                    parsed_expected_response = self.processor.parse_text(
                        expected_response
                    )
                    parsed_response = self.processor.parse_text(response)
                    joined_response = self.evaluator.evaluate_common_problems(
                        parsed_expected_response, parsed_response
                    )

                    cleaned_response = self.processor.clean_response(
                        self.evaluator.unpack_to_string(joined_response)
                    )
                    embedded__clean_response = (
                        self.evaluator.clean_and_build_embedded_response(
                            cleaned_response
                        )
                    )
                    latest_similarity_calc = calc_similarity(
                        embedded_cleaned_expected_response,
                        embedded__clean_response,
                    )
                    self.logger.info(
                        f"Latest Similarity = {latest_similarity_calc}"
                    )
                    similarity += latest_similarity_calc / self.args.n_pass
                    result["response"] = response
                    result[
                        "cleaned_response"
                    ] = LeetCodeProcessor.clean_response(response)
                    result["similarity"] = similarity

                self.logger.info(
                    f"Expected Response = {expected_response}\nResponse = {response}"
                )
                self.logger.info(f"Similarity = {similarity}")

                outputs.append(result)
                write_jsonl(self.out_dir, outputs)
            except:
                self.logger.exception(f"Failed to process {iloc}.")
                continue


if __name__ == "__main__":
    # Initialization
    openai.api_key = os.getenv("OPENAI_API_KEY_LOCAL", "")
    # TODO - Rename this to `OPENAI_API_KEY`
    openai.api_key = os.getenv("OPENAI_API_KEY_LOCAL", "")

    parser = common_arg_parser()
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
        "--step-size",
        type=int,
        default=STEP_SIZE,
        help="Iteration step size when running the experiment",
    )
    parser.add_argument(
        "--buffer",
        type=int,
        default=BUFFER,
        help="The size of the forward example buffer",
    )

    args = parser.parse_args()
    logger = get_configured_logger(args.log_level)

    # Create the experiment and run it
    experiment = SimilarityExperiment(logger, args)
    outputs = experiment.run()

    # Finalize
    write_jsonl(experiment.out_dir, outputs)
