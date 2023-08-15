import argparse
import ast
import logging
import os
import re
import time
from typing import Any

import dotenv
import leetcode
import leetcode.auth
import pandas as pd

from bs4 import BeautifulSoup

dotenv.load_dotenv()


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
    cleaned_content = re.sub(r"(\b\d+)\s+(\d+\b)", r"\1^\2", cleaned_content)

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

    cleaned_content = cleaned_content.replace(". length", ".length")
    cleaned_content = cleaned_content.replace("***", "")
    cleaned_content = cleaned_content.replace("'. '", ".")
    return cleaned_content.strip()


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

    dataset["raw_snippet"] = dataset["raw_content"].apply(
        lambda x: (x.split("</p>")[0] + "</p>" if isinstance(x, str) else "")
    )
    dataset["raw_forward_completion"] = dataset["raw_content"].apply(
        lambda x: (
            "<p>" + "</p>".join(x.split("</p>")[1:])
            if isinstance(x, str) and "</p>" in x
            else ""
        )
    )

    dataset["cleaned_snippet"] = dataset.apply(
        lambda x: f"\nLeetCode Problem #{int(x['frontend_question_id'])}\nTitle: {x['question_title']}\nDescription:\n{clean_html_content(x['raw_snippet'])}...\n",
        axis=1,
    )
    dataset["cleaned_forward_completion"] = dataset[
        "raw_forward_completion"
    ].apply(lambda x: f"{clean_html_content(x)}\n")

    dataset["cleaned_content"] = dataset["raw_content"].apply(
        lambda x: clean_html_content(x)
    )

    dataset = dataset[dataset["paid_only"] == False]
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
