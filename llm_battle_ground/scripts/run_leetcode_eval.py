# import os
# import re
# from time import sleep

# import dotenv
# import pandas as pd
# from evalplus.data import write_jsonl

# # Now we can import any Python file from the parent directory
# from llm_battle_ground.leetcode_hard_gym.leetcode_env.environment import (
#     LeetCodeEnv,
# )  # type: ignore
# from llm_battle_ground.leetcode_hard_gym.leetcode_env.types import (  # type: ignore
#     LeetCodeSubmission,
#     ProgrammingLanguage,
# )
# from llm_battle_ground.utils import (
#     get_configured_logger,
#     get_root_fpath,
#     read_jsonl,
# )

# dotenv.load_dotenv()


# def extract_code(raw_response: str) -> str:
#     print("raw_response = ", raw_response)
#     if "```python" in raw_response:
#         cleaned_response = raw_response.split("```python")[1]
#         return cleaned_response.split("```")[0]
#     elif "```" in raw_response:
#         cleaned_response = raw_response.split("```")[1]
#         return cleaned_response.split("```")[0]
#     else:
#         # Extract the class definition as before
#         class_definition_match = re.search(
#             r"class\s\S*:\s*.*?(?=\n\n|$)", raw_response, re.DOTALL
#         )
#         class_definition = (
#             class_definition_match.group(0) if class_definition_match else None
#         )

#         # Find the position of the class definition in the raw_response
#         class_position = (
#             class_definition_match.start() if class_definition_match else -1
#         )

#         # Extract the lines before the class definition
#         lines_before_class = raw_response[:class_position].strip().splitlines()

#         print("lines_before_class = ", lines_before_class)
#         print("class_definition = ", class_definition)

#         # Extract the import statements by filtering lines that start with 'import' or 'from'
#         import_statements = [
#             line
#             for line in lines_before_class
#             if line.startswith(("import", "from"))
#         ]

#         # Combine the import statements and the class definition
#         return "\n".join(import_statements + [class_definition])


# if __name__ == "__main__":
#     LEETCODE_SCRAPED_DATA_FNAME = "leetcode_full.csv"
#     leetcode_data = pd.read_csv(
#         os.path.join(get_root_fpath(), "datasets", LEETCODE_SCRAPED_DATA_FNAME)
#     )

#     LEETCODE_RESULTS_FNAME = "leetcode_vanilla-zero-shot__filter_ez_eq_0.25_filter_med_eq_0.5_filter_hrd_eq_1.0__model_eq_gpt-4-0613__temperature_eq_0.7__n_pass_1.jsonl"
#     out_path = os.path.join(get_root_fpath(), "results", "new_results.jsonl")

#     # Load existing results if the file exists
#     new_results = []
#     if os.path.exists(out_path):
#         new_results = read_jsonl(out_path)
#         existing_frontend_ids = {
#             result["frontend_question_id"] for result in new_results
#         }
#     else:
#         existing_frontend_ids = set()

#     results = read_jsonl(
#         os.path.join(get_root_fpath(), "results", LEETCODE_RESULTS_FNAME)
#     )
#     dataset = pd.DataFrame(results)
#     env = LeetCodeEnv()
#     logger = get_configured_logger(__name__, "DEBUG")
#     for loc in range(len(dataset)):
#         # try:
#         solution_entry = dataset.iloc[loc]
#         # Skip if the problem already exists in the output
#         if solution_entry.frontend_question_id in existing_frontend_ids:
#             continue

#         lookup_entry = leetcode_data[
#             leetcode_data.frontend_question_id.astype(int)
#             == int(solution_entry.frontend_question_id)
#         ].iloc[0]

#         logger.info(
#             f"At location {loc}, solution_entry = {solution_entry}, lookup_entry = {lookup_entry}"
#         )
#         try:
#             extracted_code = extract_code(solution_entry.raw_response)
#         except:
#             print("Fail for ", solution_entry.raw_response)
#             continue
#         sub = LeetCodeSubmission(
#             code=extract_code(solution_entry.raw_response),
#             lang=ProgrammingLanguage.PYTHON3,
#             question_id=int(lookup_entry.question_id),
#             question_slug=lookup_entry.question_slug,
#         )
#         status, reward, done, submission_result = env.step(sub)
#         print(status, reward, done, submission_result)
#         new_results.append(
#             {
#                 "frontend_question_id": solution_entry.frontend_question_id,
#                 "question_id": lookup_entry.question_id,
#                 "question_slug": lookup_entry.question_slug,
#                 "difficulty": lookup_entry.difficulty,
#                 "status": status,
#                 "reward": reward,
#                 "done": done,
#             }
#         )
#         write_jsonl(out_path, new_results)
#         sleep(5)
#     # except Exception as e:
#     #     logger.error(
#     #         f"Failed to process submission at {loc} with exception {e}.",
#     #         exc_info=True,
#     #     )
#     #     sleep(5)


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
    get_configured_logger,
    get_root_fpath,
    read_jsonl,
)

dotenv.load_dotenv()


def extract_code(raw_response: str) -> str:
    print("raw_response = ", raw_response)
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

        print("lines_before_class = ", lines_before_class)
        print("class_definition = ", class_definition)

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
    out_path = os.path.join(get_root_fpath(), "results", "new_results.jsonl")

    # Load existing results if the file exists
    new_results = []
    if os.path.exists(out_path):
        new_results = read_jsonl(out_path)
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
    logger = get_configured_logger(__name__, "DEBUG")
    for loc in range(len(dataset)):
        try:
            solution_entry = dataset.iloc[loc]
            # Skip if the problem already exists in the output
            if solution_entry.frontend_question_id in existing_frontend_ids:
                continue

            lookup_entry = leetcode_data[
                leetcode_data.frontend_question_id.astype(int)
                == int(solution_entry.frontend_question_id)
            ].iloc[0]

            logger.info(
                f"At location {loc}, solution_entry = {solution_entry}, lookup_entry = {lookup_entry}"
            )
            result = {
                "frontend_question_id": solution_entry.frontend_question_id,
                "question_id": lookup_entry.question_id,
                "question_slug": lookup_entry.question_slug,
                "difficulty": lookup_entry.difficulty,
            }

            try:
                extracted_code = extract_code(solution_entry.raw_response)
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
                code=extract_code(solution_entry.raw_response),
                lang=ProgrammingLanguage.PYTHON3,
                question_id=int(lookup_entry.question_id),
                question_slug=lookup_entry.question_slug,
            )
            status, reward, done, submission_result = env.step(sub)
            print(status, reward, done, submission_result)
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
