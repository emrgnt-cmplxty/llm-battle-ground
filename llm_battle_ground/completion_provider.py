"""Module for the completion provider"""
import textwrap
from enum import Enum
from typing import Tuple

from automata.llm import OpenAIChatCompletionProvider, OpenAIConversation


class RunMode(Enum):
    """Specifies the mode of running the completion provider"""

    VANILLA = "vanilla"


class CompletionProvider:
    """Concrete class for completion providers"""

    def __init__(self, run_mode: RunMode, model: str, temperature: float):
        self.run_mode = run_mode
        self.model = model
        self.temperature = temperature

    def get_completion(self, task: str) -> str:
        """Returns the raw and cleaned completions for the given prompt"""
        if self.run_mode == RunMode.VANILLA:
            vanilla_instructions = self.get_formatted_instruction(task)
            raw_completion = self.generate_vanilla_completion(
                vanilla_instructions
            )
        return raw_completion

    def generate_vanilla_completion(self, instructions: str) -> str:
        """Generates a vanilla completion for the given prompt"""
        provider = OpenAIChatCompletionProvider(
            model=self.model,
            temperature=self.temperature,
            stream=True,
            conversation=OpenAIConversation(),
            functions=[],
        )
        return provider.standalone_call(instructions)

    def get_formatted_instruction(
        self, task_input: str, num_forward_examples: int = 1
    ) -> str:
        """Formats the instruction for the given prompt"""

        if self.run_mode == RunMode.VANILLA:
            return textwrap.dedent(
                """
                Closely examine the following five examples -

                Input:
                {TASK_INPUT}

                Now, use those five example to predict the next {NUM_FORWARD_EXAMPLES} examples that will follow -
                Output:
                """
            ).format(
                TASK_INPUT=task_input,
                NUM_FORWARD_EXAMPLES=num_forward_examples,
            )

        # if self.run_mode == RunMode.VANILLA:
        #     return textwrap.dedent(
        #         """
        #         Follow the three examples shown below to return the expected output for the final shown task.

        #         # Example 1
        #         Input:
        #         LeetCode Problem #100
        #         Title: Same Tree
        #         Description:
        #         Given the roots of two binary trees p and q, write a function to check if they are the same or not....

        #         Output:
        #         Two binary trees are considered the same if they are structurally identical, and the nodes have the same value. Example 1: Input: p = [1, 2, 3], q = [1, 2, 3] Output: true Example 2: Input: p = [1, 2], q = [1, null, 2] Output: false Example 3: Input: p = [1, 2, 1], q = [1, 1, 2] Output: false Constraints: The number of nodes in both trees is in the range [0, 100]. -10e4 <= Node. val <= 10e4

        #         # Example 2
        #         Input:
        #         LeetCode Problem #101
        #         Title: Symmetric Tree
        #         Description:
        #         Given the root of a binary tree, check whether it is a mirror of itself (i. e. , symmetric around its center)....

        #         Output:
        #         Example 1: Input: root = [1, 2, 2, 3, 4, 4, 3] Output: true Example 2: Input: root = [1, 2, 2, null, 3, null, 3] Output: false Constraints: The number of nodes in the tree is in the range [1, 1000]. -100 <= Node. val <= 100 Follow up: Could you solve it both recursively and iteratively?

        #         # Example 3
        #         Input:
        #         LeetCode Problem #102
        #         Title: Binary Tree Level Order Traversal
        #         Description:
        #         Given the root of a binary tree, return the level order traversal of its nodes' values. (i. e. , from left to right, level by level)....

        #         Output:
        #         Example 1: Input: root = [3, 9, 20, null, null, 15, 7] Output: [[3], [9, 20], [15, 7]] Example 2: Input: root = [1] Output: [[1]] Example 3: Input: root = [] Output: [] Constraints: The number of nodes in the tree is in the range [0, 2000]. -1000 <= Node. val <= 1000

        #         # Task
        #         Input:
        #         {TASK_INPUT}

        #         Output:
        #         """
        #     ).format(TASK_INPUT=task_input)

        else:
            raise ValueError("No such run mode.")
