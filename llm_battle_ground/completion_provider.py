"""Module for the completion provider"""
import textwrap
from enum import Enum

from llm_battle_ground.models.model import make_model

from automata.llm import OpenAIChatCompletionProvider, OpenAIConversation


class RunMode(Enum):
    """Specifies the mode of running the completion provider"""

    SIMILARITY = "similarity"
    VANILLA_ZERO_SHOT = "vanilla-zero-shot"


class CompletionProvider:
    """Concrete class for completion providers"""

    def __init__(
        self,
        run_mode: RunMode,
        model: str,
        temperature: float,
        provider: str = "openai",
    ):
        self.run_mode = run_mode
        self.provider = provider
        self.model = model
        self.temperature = temperature
        if self.provider == "huggingface":
            # means we need to load the model locally
            # TODO: batch size
            self.model = make_model(model, batch_size=1, temperature=temperature)

    def get_completion(self, **kwargs) -> str:
        """Returns the raw and cleaned completions for the given prompt"""

        if self.run_mode in [RunMode.SIMILARITY, RunMode.VANILLA_ZERO_SHOT]:
            vanilla_instructions = self.get_formatted_instruction(**kwargs)
            raw_completion = self.generate_vanilla_completion(
                vanilla_instructions
            )
        else:
            raise ValueError("No such run mode.")
        return raw_completion

    def generate_vanilla_completion(self, instructions: str) -> str:
        """Generates a vanilla completion for the given prompt"""
        if self.provider == "openai":
            provider = OpenAIChatCompletionProvider(
                model=self.model,
                temperature=self.temperature,
                stream=True,
                conversation=OpenAIConversation(),
                functions=[],
            )
            return provider.standalone_call(instructions)
        elif self.provider == "huggingface":
            # - PUT IMPLEMENTATION HERE -
            return self.model.codegen(instructions)[0]
        return ""

    def get_formatted_instruction(
        self,
        **kwargs,
    ) -> str:
        """Formats the instruction for the given prompt"""

        if self.run_mode == RunMode.SIMILARITY:
            task_input = kwargs.get("task_input")
            num_forward_examples = kwargs.get("num_forward_examples")
            if not task_input or not num_forward_examples:
                raise ValueError("Missing required arguments.")
            return textwrap.dedent(
                """
                Closely examine the following examples -

                Input:
                {TASK_INPUT}

                Now, use those examples to predict the next {NUM_FORWARD_EXAMPLES} examples that will follow. DO NOT OUTPUT ANY ADDITIONAL TEXT, ONLY THE NEXT {NUM_FORWARD_EXAMPLES} EXAMPLES.
                Output:
                """
            ).format(
                TASK_INPUT=task_input,
                NUM_FORWARD_EXAMPLES=num_forward_examples,
            )
        elif self.run_mode == RunMode.VANILLA_ZERO_SHOT:
            task_input = kwargs.get("task_input")
            code_prompt = kwargs.get("code_prompt")
            if not task_input or not code_prompt:
                raise ValueError("Missing required arguments.")

            return textwrap.dedent(
                """
    ### Introduction:
    {TASK_INPUT}

    ### Instruction:
    Provide a response which completes the following Python code: 

    code:
    ```python
    {CODE_PROMPT}
    ```

    ### Notes: 
    Respond with the entire complete function definition, including a re-stated function definition.
    Use only built-in libraries and numpy, assume no additional imports other than those provided and 'from typings import *'.
    Optimize your algorithm to run as efficiently as possible. This is a Hard LeetCode problem, and so in the vast majority of cases
    the appropriate solution will run in NlogN or faster. Lastly, start by re-stating the given tests into
    the local python environment, and ensure that your final solution passes all given tests. 

    ### Result:
    When you have completed the problem or have ran out of alotted iterations or tokens, return a markdown-snippet with your final algorithmic implementation using `call_termination`. 
    E.g. ```python\n{CODE_PROMPT}\n  #.... (Code Continued) ...```
    Your final result should follow EXACTLY the format shown above, except for additional imports which may be added.

                    """
            ).format(TASK_INPUT=task_input, CODE_PROMPT=code_prompt)

        else:
            raise ValueError("No such run mode.")
