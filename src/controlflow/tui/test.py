import asyncio
from math import inf

from pydantic import BaseModel

from controlflow import Task
from controlflow.flows import Flow
from controlflow.llm.messages import AIMessage
from controlflow.tui.app import TUIApp


class Name(BaseModel):
    first: str
    last: str


# Example usage with mock data

with Flow() as flow:
    # t = Task("get the user's name", result_type=str, user_access=True)
    t0 = Task(
        "Introduce yourself",
        # status="SUCCESSFUL",
        result_type=str,
        # result="this is my result",
    )
    t1 = Task(
        objective="Come up with a book title",
        result_type=str,
        depends_on=[t0],
    )
    t2 = Task(
        objective="write a short summary of the book",
        result_type=str,
        context=dict(title=t1),
    )
    t3 = Task(
        objective="rate the book from 1-5 and write a paragraph on why",
        result_type=int,
        depends_on=[t2],
    )


async def run():
    app = TUIApp(flow=flow)
    async with app.run_context(run=True, inline=True, hold=True):
        await asyncio.sleep(1)
        t0.mark_successful(
            result="this is my result\n\n and here is more  and here is more  and here is more and here is more and here is more and here is more\n\n and here is more and here is more and here is more"
        )
        await asyncio.sleep(1)
        t0.mark_failed(message="this is my result")
        app.update_message(AIMessage(content="hello there"))
        await asyncio.sleep(1)
        app.update_message(AIMessage(content="hello there"))
        await asyncio.sleep(1)
        app.update_message(AIMessage(content="hello there" * 50))
        await asyncio.sleep(1)
        app.update_message(AIMessage(content="hello there"))
        await asyncio.sleep(1)

        await asyncio.sleep(inf)


# run_task = asyncio.create_task(.run_async())


if __name__ == "__main__":
    # asyncio.run(run())
    flow.run()
