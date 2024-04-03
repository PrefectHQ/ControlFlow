![image](https://github.com/jlowin/ControlFlow/assets/153965/2d779a8e-4400-4b04-ad30-60af56db7674)

# ControlFlow

ControlFlow is a framework for integrating AI agents into traditional workflows. It allows for agents that can be precisely controlled, observed, and debugged, while retaining the autonomy and flexibility that make LLMs so powerful. ControlFlow agents are designed to be invoked programmatically, though they are capable of interacting with humans and other agents as well.

ControlFlow is built with [Marvin](https://github.com/prefecthq/marvin) and [Prefect](https://github.com/prefecthq/prefect).

## Example

```python
from control_flow import ai_flow, ai_task, run_ai, instructions
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
        interests = run_ai("ask user for three interests", result_type=list[str], user_access=True)

        # set instructions for just the next task
        with instructions("no more than 8 lines"):
            poem = write_poem_about_user(name, interests)

    return poem


demo()


```

<img width="1491" alt="image" src="https://github.com/jlowin/ControlFlow/assets/153965/d436de8d-f5c8-4ef2-a281-221b8abebd1f">
