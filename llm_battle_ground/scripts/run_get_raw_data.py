import argparse
import ast
import logging
import os
import time
from typing import Any

import dotenv
import leetcode
import leetcode.auth
import pandas as pd

dotenv.load_dotenv()


def get_info(question_slug: str, api_instance) -> dict:
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
    logger: logging.Logger, out_path: str, api_instance: Any
) -> pd.DataFrame:
    """Builds a dataset of the first 5 questions in the algorithms topic"""
    question_infos = api_instance.api_problems_topic_get(topic="algorithms")

    df = pd.DataFrame()
    for ind, question in enumerate(question_infos.stat_status_pairs):
        logger.info(f"Processing ind={ind}")
        try:
            question_slug = question.stat.question__title_slug
            info = get_info(question_slug, api_instance)
            snippets = info["code_snippets"]

            df.at[ind, "question_slug"] = question.stat.question__title_slug
            df.at[ind, "question_title"] = question.stat.question__title
            df.at[ind, "frontend_question_id"] = int(
                question.stat.frontend_question_id
            )
            df.at[ind, "question_id"] = int(question.stat.question_id)
            df.at[ind, "raw_content"] = info["content"]
            df.at[ind, "difficulty"] = question.difficulty.level
            df.at[ind, "paid_only"] = question.paid_only

            for snippet in snippets:
                df.at[ind, snippet["lang_slug"] + "_snippet"] = snippet["code"]

            logger.info(f"Resulting Entry = {df.iloc[ind]}")
            df.to_csv(out_path, index=False)

            time.sleep(1)
        except Exception as e:
            logger.error(f"Error {e} with question {question_slug}")

    return df


def main(logger: logging.Logger, out_path: str) -> None:
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
    dataset = build_dataset(logger, out_path, api_instance)
    dataset.to_csv(out_path, index=True)
    logger.info(
        f"Dataset successfully built, {len(dataset)} entries stored..."
    )


if __name__ == "__main__":
    # get path to script
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level, e.g. DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )

    script_directory = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument(
        "--out-path",
        type=str,
        default=os.path.join(
            script_directory, "..", "datasets", "leetcode.csv"
        ),
        help="Path to save the dataset, defaults to $WORKDIR/datasets/leetcode.csv",
    )

    args = parser.parse_args()
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger()

    main(logger, args.out_path)
