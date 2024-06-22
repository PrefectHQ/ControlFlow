from controlflow import Agent, Task, flow, instructions, task
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
        interests.run()

    # set instructions for just the next task
    with instructions("no more than 8 lines"):
        poem = write_poem_about_user(name, interests.result)

    return poem


if __name__ == "__main__":
    demo()
