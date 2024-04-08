import itertools
from typing import Any, Generator

import marvin
from marvin.beta.assistants import Assistant
from pydantic import BaseModel

from control_flow.flow import get_flow_messages
from control_flow.instructions import get_instructions


class DelegationStrategy(BaseModel):
    """
    A DelegationStrategy is a strategy for delegating tasks to AI assistants.
    """

    def run(self, assistants: list[Assistant]) -> Generator[Any, Any, Assistant]:
        """
        Given a list of potential assistants, delegate the task to the most qualified assistant.
        """

        raise NotImplementedError()


class Single(DelegationStrategy):
    """
    A Single delegation strategy delegates tasks to a single AI assistant.
    """

    assistant: Assistant

    def run(self, assistants: list[Assistant]) -> Generator[Any, Any, Assistant]:
        """
        Given a list of potential assistants, choose the first assistant in the list.
        """

        if self.assistant not in assistants:
            raise ValueError("Assistant not in list of assistants")

        while True:
            yield self.assistant


class RoundRobin(DelegationStrategy):
    """
    A RoundRobin delegation strategy delegates tasks to AI assistants in a round-robin fashion.
    """

    def run(self, assistants: list[Assistant]) -> Generator[Any, Any, Assistant]:
        """
        Given a list of potential assistants, choose the next assistant in the list.
        """

        yield from itertools.cycle(assistants)


class Moderator(DelegationStrategy):
    """
    A Moderator delegation strategy delegates tasks to the most qualified AI assistant, using a Marvin classifier
    """

    model: str = None

    def run(self, assistants: list[Assistant]) -> Generator[Any, Any, Assistant]:
        """
        Given a list of potential assistants, delegate the task to the most qualified assistant.
        """

        while True:
            instructions = get_instructions()
            history = get_flow_messages()

            context = dict(messages=history, global_instructions=instructions)
            assistant = marvin.classify(
                context,
                assistants,
                instructions="""
                Given the conversation context, choose the AI assistant most
                qualified to take the next turn at completing the tasks. Take into
                account the instructions, each assistant's own instructions, and the
                tools they have available.
                """,
                model_kwargs=dict(model=self.model),
            )

            yield assistant
