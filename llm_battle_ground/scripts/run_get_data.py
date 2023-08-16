import ast
import logging
import os
import time
from typing import Any

import leetcode
import leetcode.auth
import pandas as pd

from llm_battle_ground.scripts import common_arg_parser
from llm_battle_ground.types import DataDirectories, Datasets
from llm_battle_ground.utils import get_configured_logger, get_root_fpath

OUT_DIR = os.path.join(get_root_fpath(), DataDirectories.DATASETS.value)
OUT_FILE_NAME = Datasets.LEETCODE_FULL.value


def get_info(question_slug: str, api_instance) -> dict:
    """Requests the question info from the LeetCode API."""
    graphql_request = leetcode.GraphqlQuery(
        query="""
                query getQuestionDetail($titleSlug: String!) {
                question(titleSlug: $titleSlug) {
                    codeSnippets {
                        lang
                        langSlug
                        code
                        __typename
                    }
                    content
                    title 
                }
                }
            """,
        variables={"titleSlug": question_slug},
        operation_name="getQuestionDetail",
    )
    response = ast.literal_eval(
        str(api_instance.graphql_post(body=graphql_request))
    )
    return response["data"]["question"]


# TODO - Add typing for leetcode
def build_dataset(
    logger: logging.Logger, out_fpath: str, api_instance: Any
) -> pd.DataFrame:
    """Builds a dataset of the questions in the algorithms topic."""
    question_infos = api_instance.api_problems_topic_get(topic="algorithms")

    # Initialize the dataset
    df = pd.DataFrame()
    question_ids = set()

    # Load existing dataset if it exists
    if os.path.exists(out_fpath):
        df = pd.read_csv(out_fpath)
        question_ids = set(df["question_id"].values)
        logger.info(f"Dataset already exists, {len(df)} entries stored...")

    # Determine the counter, offset by 1
    # since incrementation is performed before adding
    counter = len(df) - 1

    # Build the dataset
    for ind, question in enumerate(question_infos.stat_status_pairs):
        question_id = int(question.stat.question_id)

        # Skip if question already exists
        if question_id in question_ids:
            logger.info(f"Already processed question {question_id}, skipping.")
            continue

        # Skip if question is paid
        if question.paid_only:
            logger.info(f"Skipping paid question {question_id}.")
            continue

        logger.info(f"Processing index at {ind}.")
        try:
            question_slug = question.stat.question__title_slug
            info = get_info(question_slug, api_instance)

            snippets = info["code_snippets"]

            # Increment the counter before adding
            counter += 1

            # Add the question to the dataset
            df.at[
                counter, "question_slug"
            ] = question.stat.question__title_slug
            df.at[counter, "question_title"] = question.stat.question__title
            df.at[counter, "frontend_question_id"] = int(
                question.stat.frontend_question_id
            )
            df.at[counter, "question_id"] = int(question.stat.question_id)
            df.at[counter, "raw_content"] = info["content"]
            df.at[counter, "difficulty"] = question.difficulty.level
            df.at[counter, "paid_only"] = question.paid_only

            for snippet in snippets:
                df.at[counter, snippet["lang_slug"] + "_snippet"] = snippet[
                    "code"
                ]

            logger.info(f"Resulting Entry = {df.iloc[counter]}.")
            df.to_csv(out_fpath, index=False)
            logger.info(f"Dataset is now {len(df)} entries long.")

            # Sleep to avoid rate limiting
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error {e} with question {question_slug}.")

    return df


def main(logger: logging.Logger, out_fpath: str) -> None:
    """Runs the main script"""
    configuration = leetcode.Configuration()

    # From Dev Tools/Application/Cookies/LEETCODE_SESSION
    leetcode_session = os.environ["LEETCODE_SESSION"]
    csrf_token = leetcode.auth.get_csrf_cookie(leetcode_session)

    # Build the API client
    configuration.api_key["x-csrftoken"] = csrf_token
    configuration.api_key["csrftoken"] = csrf_token
    configuration.api_key["LEETCODE_SESSION"] = leetcode_session
    configuration.api_key["Referer"] = "https://leetcode.com"
    configuration.debug = False
    api_instance = leetcode.DefaultApi(leetcode.ApiClient(configuration))

    # Build the dataset
    logger.info("Building dataset now...")
    dataset = build_dataset(logger, out_fpath, api_instance)

    dataset.to_csv(out_fpath, index=False)

    logger.info(
        f"Dataset successfully built, {len(dataset)} entries stored..."
    )


if __name__ == "__main__":
    parser = common_arg_parser()
    args = parser.parse_args()

    logger = get_configured_logger(__name__, args.log_level)

    out_fpath = os.path.join(
        args.out_dir or OUT_DIR, args.out_file_name or OUT_FILE_NAME
    )
    main(logger, out_fpath)
