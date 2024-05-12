import glob as glob_module
from pathlib import Path

import control_flow
from control_flow import ai_task, flow
from marvin.beta.assistants import Assistant, Thread
from marvin.tools.filesystem import read, write

ROOT = Path(control_flow.__file__).parents[2]


def glob(pattern: str) -> list[str]:
    """
    Returns a list of paths matching a valid glob pattern.
    The pattern can include ** for recursive matching, such as
    '~/path/to/root/**/*.py'
    """
    return glob_module.glob(pattern, recursive=True)


assistant = Assistant(
    instructions="""
    You are an expert technical writer who writes wonderful documentation for 
    open-source tools and believes that documentation is a product unto itself.
    """,
    tools=[read, write, glob],
)


@ai_task(model="gpt-3.5-turbo")
def examine_source_code(source_dir: Path, extensions: list[str]):
    """
    Load all matching files in the root dir and all subdirectories and
    read them carefully.
    """


@ai_task(model="gpt-3.5-turbo")
def read_docs(docs_dir: Path):
    """
    Read all documentation in the docs dir and subdirectories, if any.
    """


@ai_task
def write_docs(docs_dir: Path, instructions: str = None):
    """
    Write new documentation based on the provided instructions.
    """


@flow(assistant=assistant)
def docs_flow(instructions: str):
    examine_source_code(ROOT / "src", extensions=[".py"])
    # read_docs(ROOT / "docs")
    write_docs(ROOT / "docs", instructions=instructions)


if __name__ == "__main__":
    thread = Thread()
    docs_flow(
        _thread=thread,
        instructions="Write documentation for the AI Flow class and save it in docs/flow.md",
    )
