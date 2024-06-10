from controlflow import Agent, Task, flow

a1 = Agent(
    name="Half-full",
    instructions="""
    You are an eternal optimist.
    """,
)
a2 = Agent(
    name="Half-empty",
    instructions="""
    You are an eternal pessimist.
    """,
)
# create an agent that will decide who wins the debate
a3 = Agent(name="Moderator")


@flow
def demo():
    topic = "pineapple on pizza"

    task = Task(
        "Have a debate about the topic. Each agent should take at least two turns.",
        agents=[a1, a2],
        context={"topic": topic},
    )
    task.run()

    task2 = Task(
        "which argument do you find more compelling?",
        result_type=[a1.name, a2.name],
        agents=[a3],
    )
    task2.run()


if __name__ == "__main__":
    demo()
