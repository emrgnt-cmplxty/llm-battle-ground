import logging
import os

import pandas as pd
from evalplus.data import write_jsonl

from llm_battle_ground.completion_provider import CompletionProvider, RunMode
from llm_battle_ground.helpers.leet_code_processor import LeetCodeProcessor
from llm_battle_ground.helpers.similarity_response_evaluator import (
    SimilarityResponseEvaluator,
)
from llm_battle_ground.types import DataDirectories, Datasets, LLMProviders
from llm_battle_ground.utils import calc_similarity, get_root_fpath, read_jsonl

# Pathing
IN_DIR = os.path.join(get_root_fpath(), DataDirectories.DATASETS.value)
IN_FILE_NAME = Datasets.LEETCODE_FULL.value
OUT_DIR = os.path.join(
    get_root_fpath(),
    DataDirectories.RESULTS.value,
    "leetcode",
    "provider",
    "{PROVIDER}",
    "{MODEL}",
    "{RUN_MODE}",
)

OUT_FILE_NAME = "similarity_{IN_FILE_NAME}__{MODEL}__step_size_eq_{STEP_SIZE}__num_input_examples_eq_{NUM_OUTPUT_EXAMPLES}__num_output_examples_eq_{NUM_OUTPUT_EXAMPLES}__buffer_eq_{BUFFER}__temperature_eq_{TEMPERATURE}__n_pass_{N_PASS}.jsonl"

# Default input constants
MODEL = "gpt-4-0613"
TEMPERATURE = 0.7
N_PASS = 1
RUN_MODE = "similarity"
PROVIDER = "openai"


class SimilarityExperimentRunner:
    def __init__(self, logger: logging.Logger, args):
        self.logger = logger
        self.args = args
        self.args.model = self.args.model or MODEL
        self.args.temperature = self.args.temperature or TEMPERATURE
        self.args.n_pass = self.args.n_pass or N_PASS
        self.processor = LeetCodeProcessor()
        self.evaluator = SimilarityResponseEvaluator()
        self.provider = CompletionProvider(
            run_mode=RunMode(RUN_MODE),
            model=self.args.model,
            temperature=self.args.temperature,
            provider=LLMProviders(self.args.provider or PROVIDER),
        )

        self.output_file_name = (
            self.args.out_file_name
            or OUT_FILE_NAME.format(
                IN_FILE_NAME=(args.in_file_name or IN_FILE_NAME)
                .replace(".csv", "")
                .replace(".jsonl", ""),
                MODEL=self.args.model.replace("-", "_").replace(".", "p"),
                STEP_SIZE=self.args.step_size,
                NUM_INPUT_EXAMPLES=self.args.num_input_examples,
                NUM_OUTPUT_EXAMPLES=self.args.num_output_examples,
                BUFFER=self.args.buffer,
                TEMPERATURE=str(self.args.temperature).replace(".", "p"),
                N_PASS=self.args.n_pass,
            ).replace("-", "_")
        )

        self.out_path = os.path.join(
            self.args.out_dir
            or OUT_DIR.format(
                PROVIDER=self.args.provider or PROVIDER,
                MODEL=self.args.model.replace("-", "_").replace(".", "p"),
                RUN_MODE=RUN_MODE,
            ),
            self.output_file_name,
        )
        if not os.path.exists(os.path.dirname(self.out_path)):
            os.makedirs(os.path.dirname(self.out_path))

    def run(self) -> list:
        """Run the experiment."""

        in_dir = os.path.join(
            self.args.in_dir or IN_DIR,
            self.args.in_file_name or IN_FILE_NAME,
        )

        self.logger.info(f"Loading dataset file from {in_dir}.")
        dataset = pd.read_csv(in_dir).sort_values(by=["frontend_question_id"])

        self.processor.augment_dataset(dataset)
        # Filter out rows with empty raw_content
        dataset = dataset[pd.notna(dataset["raw_content"])]

        if os.path.exists(self.out_path):
            self.logger.info(f"Loading existing results from {self.out_path}.")
            outputs = read_jsonl(self.out_path)
            if len(outputs) > 0:
                start_index = outputs[-1]["iloc"] + self.args.step_size
            else:
                start_index = self.args.num_input_examples + 1
        else:
            outputs = []
            start_index = self.args.num_input_examples + 1

        self._generate_similarity(dataset, start_index, self.provider, outputs)
        write_jsonl(self.out_path, outputs)
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
                    f"Example {counter + 1}: {dataset.iloc[i]['example_statement']}..."
                    for counter, i in enumerate(
                        range(iloc - self.args.num_input_examples, iloc)
                    )
                )

                logging.info(f"Running for input context:\n{input_context}")

                expected_response = "\n".join(
                    f"Example {counter + 1}: {dataset.iloc[i]['example_statement']}..."
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

                logging.info(f"Running for with formatted prompt:\n{prompt}")

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
                write_jsonl(self.out_path, outputs)
            except:
                self.logger.exception(f"Failed to process {iloc}.")
                continue
