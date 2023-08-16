"""
# original script
```python

import os
from time import sleep

import pandas as pd
from evalplus.data import write_jsonl

from llm_battle_ground.types import DataDirectories, Datasets

# Now we can import any Python file from the parent directory
from llm_battle_ground.leetcode_hard_gym.leetcode_env.environment import (
    LeetCodeEnv,
)
from llm_battle_ground.leetcode_hard_gym.leetcode_env.types import (
    LeetCodeSubmission,
    ProgrammingLanguage,
)
from llm_battle_ground.scripts import common_arg_parser
from llm_battle_ground.utils import (
    extract_code,
    get_configured_logger,
    get_root_fpath,
    read_jsonl,
)


SESSION_ID_ONE = ...
SESSION_ID_TWO = ...
SESSION_ID_THREE = ...
SESSIONS = [SESSION_ID_ONE, SESSION_ID_TWO, SESSION_ID_THREE]

if __name__ == "__main__":
    parser = common_arg_parser()
    args = parser.parse_args()
    logger = get_configured_logger(__name__, "DEBUG")

    # Load the LeetCode dataset
    leetcode_reference_data = pd.read_csv(
        os.path.join(
            os.path.join(get_root_fpath(), DataDirectories.DATASETS),
            Datasets.LEETCODE_DATASET,
        )
    )

    # Load the generated_answers
    in_dir = args.in_dir or os.path.join(
        get_root_fpath(), DataDirectories.RESULTS
    )
    in_file_name = os.path.join(in_dir, args.in_file_name)
    in_path = os.path.join(in_dir, in_file_name)
    generated_answers = pd.DataFrame(read_jsonl(in_path))

    # Establish the output path
    out_dir = args.out_dir or DataDirectories.RESULTS
    out_file_name = args.out_file_name or in_file_name.replace(
        "_generate__", "_evaluate__"
    )
    out_path = os.path.join(out_dir, out_file_name)

    # Load existing results if the file exists
    new_results = []
    if os.path.exists(out_path):
        new_results = read_jsonl(out_path)
        existing_frontend_ids = {
            result["frontend_question_id"] for result in new_results
        }
    else:
        existing_frontend_ids = set()

    # Establish the LeetCode environment
    env = LeetCodeEnv()

    counter = 0
    for loc in range(len(generated_answers)):
        try:
            answer = generated_answers.iloc[loc]

            # Skip if the problem already exists in the output
            if answer.frontend_question_id in existing_frontend_ids:
                continue

            # Lookup the underlying entry for env input
            lookup_entry = leetcode_reference_data[
                leetcode_reference_data.frontend_question_id.astype(int)
                == int(answer.frontend_question_id)
            ].iloc[0]

            logger.info(
                f"At location {loc}, answer = {answer}, lookup_entry = {lookup_entry}"
            )

            # Build the result
            result = {
                "frontend_question_id": answer.frontend_question_id,
                "question_id": lookup_entry.question_id,
                "question_slug": lookup_entry.question_slug,
                "difficulty": lookup_entry.difficulty,
            }

            # Attempt to extract the code from the answer
            # failure to extract code will result in a failure result
            try:
                extracted_code = extract_code(answer.raw_response)
            except Exception as e:
                logger.error(
                    f"Failed to extract code for {loc}", exc_info=True
                )
                # append a failure result
                result["status"] = "Wrong Answer"
                result["reward"] = False
                result["done"] = False
                new_results.append(result)
                continue

            sub = LeetCodeSubmission(
                code=extract_code(answer.raw_response),
                lang=ProgrammingLanguage.PYTHON3,
                question_id=int(lookup_entry.question_id),
                question_slug=lookup_entry.question_slug,
            )
            counter += 1
            new_session_id = SESSIONS[counter % len(SESSIONS)]
            os.environ["LEETCODE_SESSION"] = new_session_id
            logger.info(f"New session id = {new_session_id}")
            status, reward, done, submission_result = env.step(sub)
            logger.info(
                f"Status:\n{status}\nReward:\n{reward}\nDone:\n{done}\nResult:{submission_result}"
            )
            result["status"] = status
            result["reward"] = reward
            result["done"] = done
            new_results.append(result)
            write_jsonl(out_path, new_results)
            sleep(5)
        except Exception as e:
            logger.error(
                f"Failed to process submission at {loc} with exception {e}.",
                exc_info=True,
            )
            sleep(5)
```
"""
import logging
from typing import List, Tuple
import os
from time import sleep
import pandas as pd
from llm_battle_ground.types import DataDirectories, Datasets
from llm_battle_ground.leetcode_hard_gym.leetcode_env.environment import (
    LeetCodeEnv,
)
from llm_battle_ground.leetcode_hard_gym.leetcode_env.types import (
    LeetCodeSubmission,
    ProgrammingLanguage,
)
from llm_battle_ground.scripts import common_arg_parser
from llm_battle_ground.utils import (
    extract_code,
    get_configured_logger,
    get_root_fpath,
    read_jsonl,
)
from evalplus.data import write_jsonl


class SessionManager:
    SESSION_ID_ONE = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJfYXV0aF91c2VyX2lkIjoiMTA0MjcyNDUiLCJfYXV0aF91c2VyX2JhY2tlbmQiOiJkamFuZ28uY29udHJpYi5hdXRoLmJhY2tlbmRzLk1vZGVsQmFja2VuZCIsIl9hdXRoX3VzZXJfaGFzaCI6IjEzZTUxMTcxY2IzNGMxMGJiNWFhZDdiNmY3MmU1ZTg2N2UwYmY4YmQiLCJpZCI6MTA0MjcyNDUsImVtYWlsIjoiYW5vbjFAZW1lcmdlbnRhZ2kub3JnIiwidXNlcm5hbWUiOiJhbm9uMTEyMjMzNDQ1NSIsInVzZXJfc2x1ZyI6ImFub24xMTIyMzM0NDU1IiwiYXZhdGFyIjoiaHR0cHM6Ly9zMy11cy13ZXN0LTEuYW1hem9uYXdzLmNvbS9zMy1sYy11cGxvYWQvYXNzZXRzL2RlZmF1bHRfYXZhdGFyLmpwZyIsInJlZnJlc2hlZF9hdCI6MTY5MjE1NDIyNCwiaXAiOiIyNjAxOjY0Nzo1YTgzOjI3OTA6YTliNTo1MDMxOjIwOTc6YjdkMyIsImlkZW50aXR5IjoiM2I4ZDM5OWI1NmZiOWRmNTU5MmIwNTFmZGUzNmM5MDMiLCJzZXNzaW9uX2lkIjo0NDQxMTI5NiwiX3Nlc3Npb25fZXhwaXJ5IjoxMjA5NjAwfQ.6XlkczVrQbQM6R4jdpfG5KdvuuKaVvA02aODnDeupLg"
    SESSION_ID_TWO = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJfYXV0aF91c2VyX2lkIjoiMTA0MjczMjAiLCJfYXV0aF91c2VyX2JhY2tlbmQiOiJhbGxhdXRoLmFjY291bnQuYXV0aF9iYWNrZW5kcy5BdXRoZW50aWNhdGlvbkJhY2tlbmQiLCJfYXV0aF91c2VyX2hhc2giOiI5NDFkZWQwOWM0N2FlNWY4M2NjZDUxNjU5NGIwN2NlNDdiOGUzMGYyIiwiaWQiOjEwNDI3MzIwLCJlbWFpbCI6ImFub25AZW1lcmdlbnRhaS5nZyIsInVzZXJuYW1lIjoiYW5vbjExMjIzMzQ0NTU2NiIsInVzZXJfc2x1ZyI6ImFub24xMTIyMzM0NDU1NjYiLCJhdmF0YXIiOiJodHRwczovL3MzLXVzLXdlc3QtMS5hbWF6b25hd3MuY29tL3MzLWxjLXVwbG9hZC9hc3NldHMvZGVmYXVsdF9hdmF0YXIuanBnIiwicmVmcmVzaGVkX2F0IjoxNjkyMTU0NTUzLCJpcCI6IjI2MDE6NjQ3OjVhODM6Mjc5MDphOWI1OjUwMzE6MjA5NzpiN2QzIiwiaWRlbnRpdHkiOiIxMmNmZjdjNDAxYjI0MTU4MWMxZjM0N2VjZTczNTQzZCIsInNlc3Npb25faWQiOjQ0NDExNDk3LCJfc2Vzc2lvbl9leHBpcnkiOjEyMDk2MDB9.x-nSbsZEtcB1z_gm9Y3mpss65AEsHW9jeYrmi05Vo1g"
    SESSION_ID_THREE = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJfYXV0aF91c2VyX2lkIjoiMTA0MjczNDMiLCJfYXV0aF91c2VyX2JhY2tlbmQiOiJhbGxhdXRoLmFjY291bnQuYXV0aF9iYWNrZW5kcy5BdXRoZW50aWNhdGlvbkJhY2tlbmQiLCJfYXV0aF91c2VyX2hhc2giOiJhZjRkYzdkZTFjZGI1MGQ1ZTIwM2Q2ZmRmOWJmYjg1NmY2ZjdmNjkzIiwiaWQiOjEwNDI3MzQzLCJlbWFpbCI6ImFub251c2VyMkBlbWVyZ2VudGFnaS5vcmciLCJ1c2VybmFtZSI6ImFub24xMjM0NTYxMjM0NTYiLCJ1c2VyX3NsdWciOiJhbm9uMTIzNDU2MTIzNDU2IiwiYXZhdGFyIjoiaHR0cHM6Ly9zMy11cy13ZXN0LTEuYW1hem9uYXdzLmNvbS9zMy1sYy11cGxvYWQvYXNzZXRzL2RlZmF1bHRfYXZhdGFyLmpwZyIsInJlZnJlc2hlZF9hdCI6MTY5MjE1NDg0MCwiaXAiOiIyNjAxOjY0Nzo1YTgzOjI3OTA6YTliNTo1MDMxOjIwOTc6YjdkMyIsImlkZW50aXR5IjoiMTJjZmY3YzQwMWIyNDE1ODFjMWYzNDdlY2U3MzU0M2QiLCJzZXNzaW9uX2lkIjo0NDQxMTY1MywiX3Nlc3Npb25fZXhwaXJ5IjoxMjA5NjAwfQ.I3z08FiiX9YNJxq9m6W_KcnEUQngyn2sceoEY-PQMrM"

    SESSIONS = [
        "SESSION_ID_ONE",
        "SESSION_ID_TWO",
        "SESSION_ID_THREE",
    ]

    def __init__(self):
        self.counter = 0

    def get_next_session(self):
        session_id = self.SESSIONS[self.counter % len(self.SESSIONS)]
        os.environ["LEETCODE_SESSION"] = session_id
        self.counter += 1
        return session_id


def load_data(args: dict, in_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    leetcode_reference_data_path = os.path.join(
        get_root_fpath(),
        DataDirectories.DATASETS.value,
        Datasets.LEETCODE_DATASET.value,
    )
    leetcode_reference_data = pd.read_csv(leetcode_reference_data_path)

    generated_answers = pd.DataFrame(read_jsonl(in_path))

    return leetcode_reference_data, generated_answers


def establish_output_path(args: dict, in_file_name: str) -> str:
    out_dir = args.out_dir or DataDirectories.RESULTS.value
    out_file_name = args.out_file_name or in_file_name.replace(
        "_generation__", "_evaluation__"
    )
    return os.path.join(out_dir, out_file_name)


def read_existing_results(out_path):
    return read_jsonl(out_path) if os.path.exists(out_path) else []


def build_result(answer: str, lookup_entry: pd.Series):
    return {
        "frontend_question_id": answer.frontend_question_id,
        "question_id": lookup_entry.question_id,
        "question_slug": lookup_entry.question_slug,
        "difficulty": lookup_entry.difficulty,
    }


def process_submission(
    loc: int,
    answer,
    lookup_entry,
    result: dict,
    logger: logging.Logger,
    env: LeetCodeEnv,
    new_results: List[dict],
    session_manager: SessionManager,
):
    try:
        extracted_code = extract_code(answer.raw_response)
    except Exception as e:
        logger.error(f"Failed to extract code for {loc}", exc_info=True)
        result["status"] = "Wrong Answer"
        result["reward"] = False
        result["done"] = False
        new_results.append(result)
        return

    sub = LeetCodeSubmission(
        code=extract_code(answer.raw_response),
        lang=ProgrammingLanguage.PYTHON3,
        question_id=int(lookup_entry.question_id),
        question_slug=lookup_entry.question_slug,
    )
    new_session_id = session_manager.get_next_session()
    logger.info(f"New session id = {new_session_id}")
    status, reward, done, submission_result = env.step(sub)
    logger.info(
        f"Status:{status}Reward:{reward}Done:{done}Result:{submission_result}"
    )
    result["status"] = status
    result["reward"] = reward
    result["done"] = done
    new_results.append(result)


def process_answer(
    loc,
    answer,
    leetcode_reference_data,
    existing_frontend_ids,
    logger,
    env,
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
        leetcode_reference_data.frontend_question_id.astype(int)
        == int(answer.frontend_question_id)
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
        env,
        new_results,
        session_manager,
    )

    return True


def process_answers(
    leetcode_reference_data: pd.DataFrame,
    generated_answers: pd.DataFrame,
    logger: logging.Logger,
    env: LeetCodeEnv,
    out_path: str,
    session_manager: SessionManager,
):
    logger.info(f"Loding existing results from {out_path}...")
    new_results = read_existing_results(out_path)
    existing_frontend_ids = {
        result["frontend_question_id"] for result in new_results
    }

    logger.info(f"Loaded {len(new_results)} existing results")
    print("new_results = ", new_results)
    print("generated_answers = ", generated_answers)
    print("out_path = ", out_path)
    logger.info(f"Looping over {len(generated_answers)} generated answers...")
    for loc in range(len(generated_answers)):
        logger.info(f"Processing answer at location {loc}...")
        answer = generated_answers.iloc[loc]
        logger.info("Processing answer...")
        result = process_answer(
            loc,
            answer,
            leetcode_reference_data,
            existing_frontend_ids,
            logger,
            env,
            new_results,
            session_manager,
        )
        if result:
            write_jsonl(out_path, new_results)
            sleep(5)


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
    env = LeetCodeEnv()
    session_manager = SessionManager()
    logger.info("Processing provided answers...")
    process_answers(
        leetcode_reference_data,
        generated_answers,
        logger,
        env,
        out_path,
        session_manager,
    )
