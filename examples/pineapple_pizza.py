from control_flow import Agent, Task, ai_flow
from control_flow.instructions import instructions

a1 = Agent(
    name="Half-full",
    instructions="""
    You are an ardent fan and hype-man of whatever topic
    the user asks you for information on.
    Purely positive, though thorough in your debating skills.
    """,
)
a2 = Agent(
    name="Half-empty",
    instructions="""
    You are a critic and staunch detractor of whatever topic
    the user asks you for information on.
    Mr Johnny Rain Cloud, you will find holes in any argument 
    the user puts forth, though you are thorough and uncompromising
    in your research and debating skills.
    """,
)
# create an agent that will decide who wins the debate
a3 = Agent(name="Moderator")


@ai_flow
def demo():
    topic = "pineapple on pizza"

    task = Task("Discuss the topic", agents=[a1, a2], context={"topic": topic})
    with instructions("2 sentences max"):
        task.run()

    task2 = Task(
        "which argument do you find more compelling?", [a1.name, a2.name], agents=[a3]
    )
    task2.run()


demo()
