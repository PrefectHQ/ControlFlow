import marvin

from control_flow.core.agent import Agent
from control_flow.instructions import get_instructions
from control_flow.utilities.context import ctx
from control_flow.utilities.threads import get_history


def choose_agent(
    agents: list[Agent],
    instructions: str = None,
    context: dict = None,
    model: str = None,
) -> Agent:
    """
    Given a list of potential agents, choose the most qualified assistant to complete the tasks.
    """

    instructions = get_instructions()
    history = []
    if (flow := ctx.get("flow")) and flow.thread.id:
        history = get_history(thread_id=flow.thread.id)

    info = dict(
        history=history,
        global_instructions=instructions,
        context=context,
    )

    agent = marvin.classify(
        info,
        agents,
        instructions="""
            Given the conversation context, choose the AI agent most
            qualified to take the next turn at completing the tasks. Take into
            account the instructions, each agent's own instructions, and the
            tools they have available.
            """,
        model_kwargs=dict(model=model),
    )

    return agent
