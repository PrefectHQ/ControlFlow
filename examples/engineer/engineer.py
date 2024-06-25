from pathlib import Path

import controlflow as cf
import controlflow.tools.code
import controlflow.tools.filesystem
from pydantic import BaseModel

prompt = open(Path(__file__).parent / "prompt.md").read()

agent = cf.Agent(
    "Engineer",
    instructions=prompt,
    tools=[
        controlflow.tools.filesystem.ALL_TOOLS,
        controlflow.tools.code.python,
        controlflow.tools.code.shell,
    ],
)


class DesignDoc(BaseModel):
    root_dir: str
    goals: str
    design: str


@cf.flow
def engineer():
    design_doc = cf.Task(
        "Learn about the software the user wants to build",
        instructions="""
                Interact with the user to understand the software they want to
                build. What is its purpose? What language should you use? What does
                it need to do? Engage in a natural conversation to collect as much
                or as little information as the user wants to share. Once you have
                enough, write out a design document to complete the task.
                """,
        user_access=True,
        result_type=DesignDoc,
    )
    software = cf.Task(
        "Finish the software",
        instructions="Mark this task complete when you know that the software runs as expected.",
        result_type=None,
        context=dict(design_doc=design_doc),
        agents=[agent],
    )
    return software


if __name__ == "__main__":
    engineer()
