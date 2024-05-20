import asyncio

from pydantic import BaseModel

import controlflow
from controlflow import Task
from controlflow.core.flow import Flow
from controlflow.tui.app import App


class Name(BaseModel):
    first: str
    last: str


# Example usage with mock data

with Flow() as flow:
    t = Task("get the user's name", result_type=str, user_access=True)
    # t0 = Task("Post a message to the thread introducing yourself")
    # t1 = Task(
    #     objective="Come up with a book title",
    #     result_type=str,
    #     depends_on=[t0],
    # )
    # t2 = Task(
    #     objective="write a short summary of the book",
    #     result_type=str,
    #     context=dict(title=t1),
    # )
    # t3 = Task(
    #     objective="rate the book from 1-5 and write a paragraph on why",
    #     result_type=int,
    #     depends_on=[t2],
    # )


class Message(BaseModel):
    text: str


async def run():
    app = App(flow=flow)
    async with app.run_context(run=controlflow.settings.enable_tui, inline=False):
        for task in [t0, t1, t2, t3]:
            app.add_task(task)

        for _ in range(100):
            app.add_message(Message(text="This is a message"))
        while True:
            await asyncio.sleep(1)


# run_task = asyncio.create_task(.run_async())


if __name__ == "__main__":
    # asyncio.run(run())
    flow.run()
