from control_flow import Agent, Task, ai_flow, ai_task, instructions
from pydantic import BaseModel


class Name(BaseModel):
    first_name: str
    last_name: str


@ai_task(user_access=True)
def get_user_name() -> Name:
    pass


@ai_task(agents=[Agent(name="poetry-bot", instructions="loves limericks")])
def write_poem_about_user(name: Name, interests: list[str]) -> str:
    """write a poem based on the provided `name` and `interests`"""
    pass


@ai_flow()
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
