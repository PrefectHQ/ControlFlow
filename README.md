![image](https://github.com/jlowin/ControlFlow/assets/153965/9465c321-6b3f-4a6f-af88-f7e3c250fb31)

# ControlFlow

ControlFlow is a Python framework for orchestrating AI agents in workflows alongside traditional code. It allows you to declaratively define AI tasks, assign them to agents, and seamlessly integrate them into larger workflows. By providing a structured way to coordinate multiple AI agents, ControlFlow enables you to build sophisticated applications that leverage the power of AI while maintaining the control and flexibility of traditional programming.

At its core, ControlFlow is built on the idea of agent orchestration. It provides a way to break down complex workflows into smaller, focused tasks that can be assigned to specialized AI agents. These agents can work autonomously on their assigned tasks, while the framework ensures smooth coordination and information sharing between them. This approach allows each agent to excel at its specific role, while the overall workflow benefits from their combined capabilities.

🚨 ControlFlow requires bleeding-edge versions of [Prefect](https://github.com/prefecthq/prefect) and [Marvin](https://github.com/prefecthq/marvin). Caveat emptor!

## Key Concepts

- **Flow**: A container for an AI-enhanced workflow that maintains consistent context and history. Flows are defined with the `@ai_flow` decorator.
- **Task**: A discrete objective for AI agents to solve. Tasks can be defined with the `@ai_task` decorator or declared inline.
- **Agent**: Agents encapsulate the logic for applying an AI assistant to one or more tasks.
- **Controller**: Controllers are responsible for coordinating agents, delegating tasks, and managing the overall execution of the workflow.

Users typically don't interact directly with agents or controllers. Instead, they define tasks and flows, which are then executed by the controller. The controller is responsible for managing the agents and ensuring that the workflow is executed correctly.

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
from control_flow import ai_flow, ai_task, run_ai_task, instructions
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
        interests = run_ai_task("ask user for three interests", cast=list[str], user_access=True)

        # set instructions for just the next task
        with instructions("no more than 8 lines"):
            poem = write_poem_about_user(name, interests)

    return poem

if __name__ == "__main__":
    demo()
```

<img width="1491" alt="image" src="https://github.com/jlowin/ControlFlow/assets/153965/d436de8d-f5c8-4ef2-a281-221b8abebd1f">