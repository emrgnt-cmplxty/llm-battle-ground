# TODO - Add provider to this script.


import pandas as pd
from bs4 import BeautifulSoup


import re


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
