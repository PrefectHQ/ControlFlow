![image](https://github.com/jlowin/ControlFlow/assets/153965/9465c321-6b3f-4a6f-af88-f7e3c250fb31)


# ControlFlow

ControlFlow is a Python framework for orchestrating AI agents in workflows alongside traditional code. It allows you to seamlessly integrate AI into any workflow, coordinate multiple specialized AI agents, collect of human inputs when needed, and maintain full observability for debugging.

ControlFlow is designed with the belief that AI works best when focused and iterated. It encourages breaking workflows into small, targeted steps, each handled by a dedicated AI agent. This keeps each AI as effective as possible, while maintaining context across the entire ensemble. ControlFlow recognizes that AI should augment traditional development, not replace it. It enables a declarative approach to AI, where the desired outcomes are specified and the framework handles the implementation details. This allows developers to mix AI and traditional code freely, leveraging AI where it's most useful while using standard programming everywhere else.

🚨 ControlFlow requires bleeding-edge versions of [Prefect](https://github.com/prefecthq/prefect) and [Marvin](https://github.com/prefecthq/marvin). Caveat emptor!


## Key Features

- **Seamless integration:** Any step in a workflow can be delegated to one or more AI agents, which return structured data that can be used by other steps in the workflow.
- **Multi-agent coordination:** ControlFlow can orchestrate multiple agents, allowing them to collaborate and leverage their unique strengths. Agents can interact with each other and humans in a well-defined way, enabling complex workflows to be built from simple, autonomous components.
- **Human interaction:** Though code, not chat, is the primary interface, ControlFlow agents can interact with humans to provide information or collect inputs. Build workflows that combine AI ability with human-in-the-loop interactivity and oversight.
- **Detailed observability:** ControlFlow provides detailed observability into the behavior of every agent, making it simple to identify, triage, and fix any issues.
- **Intuitive APIs:** Clean, readable decorators and APIs for defining tasks and agents, built on top of the powerful Prefect and Marvin engines.

## Get started

ControlFlow is under active development.

```bash
git clone https://github.com/jlowin/control_flow.git
cd control_flow
pip install .
```

## Example

```python
from control_flow import ai_flow, ai_task, run_agent, instructions
from pydantic import BaseModel


class Name(BaseModel):
    first_name: str
    last_name: str


@ai_task(user_access=True)
def get_user_name() -> Name:
    pass


@ai_task
def write_poem_about_user(name: Name, interests: list[str]) -> str:
    """write a poem based on the provided `name` and `interests`"""
    pass


@ai_flow()
def demo():

    # set instructions that will be used for multiple tasks
    with instructions("talk like a pirate"):

        # define an AI task as a function and have it execute it
        name = get_user_name()

        # define an AI task inline
        interests = run_agent("ask user for three interests", cast=list[str], user_access=True)

        # set instructions for just the next task
        with instructions("no more than 8 lines"):
            poem = write_poem_about_user(name, interests)

    return poem


if __name__ == "__main__":
    demo()
```

<img width="1491" alt="image" src="https://github.com/jlowin/ControlFlow/assets/153965/d436de8d-f5c8-4ef2-a281-221b8abebd1f">
