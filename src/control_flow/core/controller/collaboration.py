import itertools
from typing import TYPE_CHECKING, Any, Generator

from control_flow.core.agent import Agent

if TYPE_CHECKING:
    from control_flow.core.agent import Agent


def round_robin(
    agents: list[Agent], max_iterations: int = None
) -> Generator[Any, Any, Agent]:
    """
    Given a list of potential agents, delegate the tasks in a round-robin fashion.
    """
    cycle = itertools.cycle(agents)
    iteration = 0
    while True:
        yield next(cycle)
        iteration += 1
        if max_iterations and iteration >= max_iterations:
            break


# class Moderator(DelegationStrategy):
#     """
#     A Moderator delegation strategy delegates tasks to the most qualified AI assistant, using a Marvin classifier
#     """

#     model: str = None

#     def _next_agent(
#         self, agents: list["Agent"], tasks: list[Task], history: list[Message]
#     ) -> "Agent":
#         """
#         Given a list of potential agents, choose the most qualified assistant to complete the tasks.
#         """

#         instructions = get_instructions()

#         context = dict(tasks=tasks, messages=history, global_instructions=instructions)
#         agent = marvin.classify(
#             context,
#             [a for a in agents if a.status == AgentStatus.INCOMPLETE],
#             instructions="""
#             Given the conversation context, choose the AI agent most
#             qualified to take the next turn at completing the tasks. Take into
#             account the instructions, each agent's own instructions, and the
#             tools they have available.
#             """,
#             model_kwargs=dict(model=self.model),
#         )

#         return agent
