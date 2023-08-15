import os
import re
from time import sleep

import dotenv
import pandas as pd
from evalplus.data import write_jsonl

# Now we can import any Python file from the parent directory
from llm_battle_ground.leetcode_hard_gym.leetcode_env.environment import (
    LeetCodeEnv,
)  # type: ignore
from llm_battle_ground.leetcode_hard_gym.leetcode_env.types import (  # type: ignore
    LeetCodeSubmission,
    ProgrammingLanguage,
)
from llm_battle_ground.utils import (
    get_root_fpath,
    read_jsonl,
    get_configured_logger,
)

dotenv.load_dotenv()


def extract_code(raw_response: str) -> str:
    if "```python" in raw_response:
        cleaned_response = raw_response.split("```python")[1]
        return cleaned_response.split("```")[0]
    elif "```" in raw_response:
        cleaned_response = raw_response.split("```")[1]
        return cleaned_response.split("```")[0]
    else:
        # Extract the class definition as before
        class_definition_match = re.search(
            r"class\s\S*:\s*.*?(?=\n\n|$)", raw_response, re.DOTALL
        )
        class_definition = (
            class_definition_match.group(0) if class_definition_match else None
        )

        # Find the position of the class definition in the raw_response
        class_position = (
            class_definition_match.start() if class_definition_match else -1
        )

        # Extract the lines before the class definition
        lines_before_class = raw_response[:class_position].strip().splitlines()

        # Extract the import statements by filtering lines that start with 'import' or 'from'
        import_statements = [
            line
            for line in lines_before_class
            if line.startswith(("import", "from"))
        ]

        # Combine the import statements and the class definition
        return "\n".join(import_statements + [class_definition])


if __name__ == "__main__":
    LEETCODE_SCRAPED_DATA_FNAME = "leetcode_full.csv"
    leetcode_data = pd.read_csv(
        os.path.join(get_root_fpath(), "datasets", LEETCODE_SCRAPED_DATA_FNAME)
    )

    LEETCODE_RESULTS_FNAME = "leetcode_vanilla-zero-shot__filter_ez_eq_0.25_filter_med_eq_0.5_filter_hrd_eq_1.0__model_eq_gpt-4-0613__temperature_eq_0.7__n_pass_1.jsonl"
    out_dir = os.path.join(get_root_fpath(), "results", "new_results.jsonl")

    # Load existing results if the file exists
    new_results = []
    if os.path.exists(out_dir):
        new_results = read_jsonl(out_dir)
        existing_frontend_ids = {
            result["frontend_question_id"] for result in new_results
        }
    else:
        existing_frontend_ids = set()

    results = read_jsonl(
        os.path.join(get_root_fpath(), "results", LEETCODE_RESULTS_FNAME)
    )
    dataset = pd.DataFrame(results)
    env = LeetCodeEnv()
    logger = get_configured_logger("DEBUG")
    for loc in range(len(dataset)):
        try:
            solution_entry = dataset.iloc[loc]
            # Skip if the problem already exists in the output
            if solution_entry.frontend_question_id in existing_frontend_ids:
                continue

            lookup_entry = leetcode_data[
                leetcode_data.frontend_question_id
                == solution_entry.frontend_question_id
            ].iloc[0]

            sub = LeetCodeSubmission(
                code=extract_code(solution_entry.raw_response),
                lang=ProgrammingLanguage.PYTHON3,
                question_id=int(lookup_entry.question_id),
                question_slug=lookup_entry.question_slug,
            )
            status, reward, done, submission_result = env.step(sub)
            print(status, reward, done, submission_result)
            new_results.append(
                {
                    "frontend_question_id": solution_entry.frontend_question_id,
                    "question_id": lookup_entry.question_id,
                    "question_slug": lookup_entry.question_slug,
                    "difficulty": lookup_entry.difficulty,
                    "status": status,
                    "reward": reward,
                    "done": done,
                }
            )
            write_jsonl(out_dir, new_results)
            sleep(5)
        except Exception as e:
            logger.error(
                f"Failed to process submission at {loc}", exc_info=True
            )
            sleep(5)
