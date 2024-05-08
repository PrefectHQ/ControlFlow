from control_flow import Agent, Task, ai_flow

a1 = Agent(name="A1", instructions="You struggle to make decisions.")
a2 = Agent(
    name="A2",
    instructions="You like to make decisions.",
)


@ai_flow
def demo():
    task = Task("Choose a number between 1 and 100", agents=[a1, a2], result_type=int)

    while task.is_incomplete():
        a1.run(task)
        a2.run(task)

    return task


demo()
