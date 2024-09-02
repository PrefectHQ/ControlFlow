import asyncio

from controlflow import Task
from controlflow.flows import Flow
from controlflow.tui.app import TUIApp

asyncio
with Flow() as flow:
    t = Task("get the user name", interactive=True)


async def run():
    app = TUIApp(flow=flow)
    async with app.run_context(run=True, inline=True, hold=True, headless=False):
        response = await app.get_input("hello")

    return response


# run_task = asyncio.create_task(.run_async())


if __name__ == "__main__":
    r = asyncio.run(run())
    # print(r)
    # flow.run()
