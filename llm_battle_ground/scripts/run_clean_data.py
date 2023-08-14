import argparse
import logging
import os
import re

import dotenv
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


def main(logger: logging.Logger, in_path: str, out_path: str) -> None:
    """Runs the main script"""
    logger.info(f"Loading dataset from {in_path}.")
    dataset = pd.read_csv(in_path)

    logger.info(f"Loaded dataset with {len(dataset)} entries.")

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
    dataset.to_csv(out_path, index=False)
    logger.info(f"Saved dataset to {out_path} with {len(dataset)} entries.")


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
        "--in-path",
        type=str,
        default=os.path.join(
            script_directory, "..", "datasets", "leetcode.csv"
        ),
        help="Path to load the dataset, defaults to $WORKDIR/datasets/leetcode.csv",
    )
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

    main(logger, args.in_path, args.out_path)
