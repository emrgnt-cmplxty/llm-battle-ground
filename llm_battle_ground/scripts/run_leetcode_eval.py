import argparse
import logging
import os
from time import sleep
from typing import Tuple

import pandas as pd
from evalplus.data import write_jsonl

from llm_battle_ground.constants import RESULTS_DIRECTORY
from llm_battle_ground.leetcode_hard_gym.leetcode_env.environment import (
    LeetCodeEnv,
)
from llm_battle_ground.leetcode_hard_gym.leetcode_env.types import (
    LeetCodeSubmission,
    ProgrammingLanguage,
)
from llm_battle_ground.scripts import common_arg_parser
from llm_battle_ground.types import DataDirectories, Datasets
from llm_battle_ground.utils import (
    extract_code,
    get_configured_logger,
    get_root_fpath,
    read_jsonl,
)

USER_WAIT_TIME = 10  # seconds, per client
IP_WAIT_TIME = 5  # seconds, per IP
TIMEOUT_WAIT = 7
MAX_SAMPLES = int(1e10)


class SessionManager:
    sessions = os.environ["LEETCODE_SESSIONS"].split(",")

    SESSIONS = [f"SESSION_ID_{i}" for i in range(len(sessions))]

    def __init__(self):
        self.counter = 0
        self.envs = []
        for index in range(len(self.SESSIONS)):
            self.set_env(index)
            env = LeetCodeEnv(USER_WAIT_TIME)
            logging.info(
                f"Creating a LeetCodeEnv leetcode_session = {os.environ['LEETCODE_SESSION']}"
            )
            self.envs.append(env)

    def set_env(self, index: int) -> None:
        session_id = self.SESSIONS[index % len(self.SESSIONS)]
        logger.info(f"Setting session_id = {session_id}")
        os.environ["LEETCODE_SESSION"] = self.sessions[
            index % len(self.SESSIONS)
        ]

    def get_next_env(self) -> LeetCodeEnv:
        env = self.envs[self.counter % len(self.envs)]
        self.counter += 1
        return env


def load_data(args: dict, in_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    leetcode_reference_data_path = os.path.join(
        get_root_fpath(),
        DataDirectories.DATASETS.value,
        Datasets.LEETCODE_FULL.value,
    )
    leetcode_reference_data = pd.read_csv(leetcode_reference_data_path)

    generated_answers = pd.DataFrame(read_jsonl(in_path))

    return leetcode_reference_data, generated_answers


def establish_output_path(args: argparse.Namespace, in_file_name: str) -> str:
    out_dir = args.out_dir or RESULTS_DIRECTORY
    out_file_name = args.out_file_name or in_file_name.replace(
        "generation", "evaluation"
    )
    return os.path.join(out_dir, out_file_name)


def read_existing_results(out_path: str) -> list[dict]:
    """Reads existing results from out_path if it exists, otherwise returns empty list"""
    return read_jsonl(out_path) if os.path.exists(out_path) else []


def build_result(answer: pd.Series, lookup_entry: pd.Series):
    return {
        "frontend_question_id": int(answer.frontend_question_id),
        "question_id": int(lookup_entry.question_id),
        "question_slug": lookup_entry.question_slug,
        "difficulty": lookup_entry.difficulty,
    }


def process_submission(
    loc: int,
    answer,
    lookup_entry,
    result: dict,
    logger: logging.Logger,
    new_results: list[dict],
    session_manager: SessionManager,
):
    try:
        extracted_code = extract_code(answer.raw_response)
    except Exception as e:
        logger.error(
            f"Failed to extract code for {loc} with {e}", exc_info=True
        )
        result["status"] = "Wrong Answer"
        result["reward"] = False
        result["done"] = False
        new_results.append(result)
        return

    sub = LeetCodeSubmission(
        code=extracted_code,
        lang=ProgrammingLanguage.PYTHON3,
        question_id=str(int(lookup_entry.question_id)),
        question_slug=lookup_entry.question_slug,
        timeout=TIMEOUT_WAIT,
    )
    (
        status,
        reward,
        done,
        submission_result,
    ) = session_manager.get_next_env().step(sub)
    logger.info(
        f"Status:{status}, Reward:{reward}, Done:{done}, Result:{submission_result}"
    )
    result["status"] = status
    result["reward"] = reward
    result["done"] = done

    logger.info("Sleeping now...")
    sleep(IP_WAIT_TIME)  # TODO - Why can't we rely on client sleep?

    new_results.append(result)


def process_answer(
    loc,
    answer,
    leetcode_reference_data,
    existing_frontend_ids,
    logger,
    new_results,
    session_manager,
):
    logger.info("Checking if answer already exists...")
    if answer.frontend_question_id in existing_frontend_ids:
        logger.info(
            f"Returning early because answer {answer.frontend_question_id} already exists"
        )

        return False

    logger.info(f"Creating submission for {answer.frontend_question_id}")
    lookup_entry = leetcode_reference_data[
        leetcode_reference_data.frontend_question_id
        == answer.frontend_question_id
    ].iloc[0]

    logger.info("Building result...")
    result = build_result(answer, lookup_entry)

    logger.info("Processing submission...")
    process_submission(
        loc,
        answer,
        lookup_entry,
        result,
        logger,
        new_results,
        session_manager,
    )

    return True


def process_answers(
    leetcode_reference_data: pd.DataFrame,
    generated_answers: pd.DataFrame,
    logger: logging.Logger,
    out_path: str,
    session_manager: SessionManager,
    max_samples: int,
):
    logger.info(f"Loding existing results from {out_path}...")
    new_results = read_existing_results(out_path)
    existing_frontend_ids = {
        result["frontend_question_id"] for result in new_results
    }

    logger.info(f"Loaded {len(new_results)} existing results")
    logger.info(f"Looping over {len(generated_answers)} generated answers...")
    generated_answers = generated_answers.sort_values(
        by=["frontend_question_id"]
    )[::-1]

    print(f"generated_answers = {generated_answers}")
    for loc in range(len(generated_answers)):
        logger.info(f"Processing answer at location {loc}...")
        answer = generated_answers.iloc[loc]
        logger.info("Processing answer...")
        if len(new_results) >= max_samples:
            break
        try:
            result = process_answer(
                loc,
                answer,
                leetcode_reference_data,
                existing_frontend_ids,
                logger,
                new_results,
                session_manager,
            )
            if result:
                write_jsonl(out_path, new_results)
        except Exception as e:
            logger.error(f"Failed to process answer with {e}", exc_info=True)
            logger.info("Sleeping full downtime...")
            sleep(IP_WAIT_TIME)  # TODO - Why can't we rely on client sleep?


if __name__ == "__main__":
    parser = common_arg_parser()
    args = parser.parse_args()
    logger = get_configured_logger(__name__, "DEBUG")

    in_dir = args.in_dir or os.path.join(
        get_root_fpath(), DataDirectories.RESULTS.value
    )
    in_path = os.path.join(in_dir, args.in_file_name)

    leetcode_reference_data, generated_answers = load_data(args, in_path)
    out_path = establish_output_path(args, in_path)
    logger.info(f"Saving results to {out_path}")
    session_manager = SessionManager()
    logger.info("Processing provided answers...")
    process_answers(
        leetcode_reference_data,
        generated_answers,
        logger,
        out_path,
        session_manager,
        args.max_samples or MAX_SAMPLES,
    )
