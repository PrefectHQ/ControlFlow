![image](https://github.com/jlowin/controlflow/assets/153965/c2a8a2f0-8777-49a6-a79b-a0e101bd4a04)

# ControlFlow

ControlFlow is a Python framework that simplifies the process of building AI-powered applications by providing a structured approach to orchestrating AI agents alongside traditional code. It allows developers to engage specialized AI models for specific tasks, while seamlessly integrating their results back into the main application workflow.

## Why ControlFlow?

Building AI applications often involves wrestling with complex, monolithic models that can be difficult to control and integrate into existing software. ControlFlow offers a more targeted and developer-friendly approach:

- Break down your application into smaller, well-defined tasks
- Assign tasks to specialized AI agents, providing clear instructions and constraints
- Seamlessly integrate AI-generated results back into your application logic

This targeted approach results in AI systems that are easier to build, maintain, and understand.

## Key Concepts

- **Flow**: A container for an AI-enhanced workflow, defined using the `@flow` decorator. Flows maintain consistent context and history across tasks.

- **Task**: A discrete objective for AI agents to solve, defined using the `@task` decorator or declared inline. Tasks specify the expected inputs and outputs, acting as a bridge between AI agents and traditional code.

- **Agent**: An AI agent that can be assigned tasks. Agents are powered by specialized AI models that excel at specific tasks, such as text generation or decision making based on unstructured data.


## Get Started

ControlFlow is under active development.

```bash
git clone https://github.com/PrefectHQ/controlflow.git
cd controlflow
pip install .
```

## Development

To install for development:

```bash
git clone https://github.com/PrefectHQ/controlflow.git
cd controlflow
pip install -e ".[dev]"
```

To run tests:

```bash
cd controlflow
pytest -vv
```

The ControlFlow documentation is built with [Mintlify](https://mintlify.com/). To build the documentation, first install `mintlify`:
```bash
npm i -g mintlify
```
Then run the local build:
```bash
cd controlflow/docs
mintlify dev
```
## Example

```python
from controlflow import Agent, Task, flow, task, instructions
from pydantic import BaseModel


class Name(BaseModel):
    first_name: str
    last_name: str


@task(user_access=True)
def get_user_name() -> Name:
    pass


@task(agents=[Agent(name="poetry-bot", instructions="loves limericks")])
def write_poem_about_user(name: Name, interests: list[str]) -> str:
    """write a poem based on the provided `name` and `interests`"""
    pass


@flow()
def demo():
    # set instructions that will be used for multiple tasks
    with instructions("talk like a pirate"):
        # define an AI task as a function
        name = get_user_name()

        # define an AI task imperatively
        interests = Task(
            "ask user for three interests", result_type=list[str], user_access=True
        )
        interests.run_until_complete()

    # set instructions for just the next task
    with instructions("no more than 8 lines"):
        poem = write_poem_about_user(name, interests.result)

    return poem


if __name__ == "__main__":
    demo()
```




<img width="1491" alt="image" src="https://github.com/jlowin/controlflow/assets/153965/43b7278b-7bcf-4d65-b219-c3a20f62a179">
