from controlflow import Agent, Task, flow

a1 = Agent(name="A1", instructions="You struggle to make decisions.")
a2 = Agent(
    name="A2",
    instructions="You like to make decisions.",
)


@flow
def demo():
    task = Task("choose a number between 1 and 100", agents=[a1, a2], result_type=int)
    return task.run()


demo()
