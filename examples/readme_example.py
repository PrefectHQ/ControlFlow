from control_flow import ai_flow, ai_task, instructions, run_ai
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
        interests = run_ai(
            "ask user for three interests", cast=list[str], user_access=True
        )

        # set instructions for just the next task
        with instructions("no more than 8 lines"):
            poem = write_poem_about_user(name, interests)

    return poem


if __name__ == "__main__":
    demo()
