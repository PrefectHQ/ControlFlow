import asyncio

from marvin.utilities.asyncio import run_sync

from controlflow import Task
from controlflow.core.flow import Flow
from controlflow.tui.app import TUIApp

run_sync
asyncio
with Flow() as flow:
    t = Task("get the user name", user_access=True)


async def run():
    app = TUIApp(flow=flow)
    async with app.run_context(run=True, inline=True, hold=True, headless=False):
        response = await app.get_input("hello")

    return response


# run_task = asyncio.create_task(.run_async())


if __name__ == "__main__":
    # r = asyncio.run(run())
    # print(r)
    flow.run()