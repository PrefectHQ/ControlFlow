from control_flow import Agent, Task, ai_flow, ai_task

a1 = Agent(
    name="A1",
    instructions="You like to make decisions.",
)
a2 = Agent(name="A2", instructions="You struggle to make decisions.")


@ai_task(user_access=True)
def get_user_name() -> str:
    """get the user's name"""
    pass


@ai_flow
def demo():
    task = Task[int]("Choose a number between 1 and 100", agents=[a1, a2])

    while task.is_incomplete():
        a1.run(task)
        a2.run(task)

    get_user_name()

    return task


demo()
