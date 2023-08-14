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
                Closely examine the following examples -

                Input:
                {TASK_INPUT}

                Now, use those five example to predict the next {NUM_FORWARD_EXAMPLES} examples that will follow -
                Output:
                """
            ).format(
                TASK_INPUT=task_input,
                NUM_FORWARD_EXAMPLES=num_forward_examples,
            )
        else:
            raise ValueError("No such run mode.")
