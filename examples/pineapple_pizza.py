from control_flow import Agent, Task, ai_flow
from control_flow.instructions import instructions

a1 = Agent(
    name="Half-full",
    instructions="You are an ardent fan and hype-man of whatever topic"
    " the user asks you for information on."
    " Purely positive, though thorough in your debating skills.",
)
a2 = Agent(
    name="Half-empty",
    instructions="You are a critic and staunch detractor of whatever topic"
    " the user asks you for information on."
    " Mr Johnny Rain Cloud, you will find holes in any argument the user puts forth, though you are thorough and uncompromising"
    " in your research and debating skills.",
)


@ai_flow
def demo():
    user_message = "pineapple on pizza"

    with instructions("one sentence max"):
        task = Task(
            "All agents must give an argument based on the user message",
            agents=[a1, a2],
            context={"user_message": user_message},
        )
        task.run_until_complete()

    task2 = Task(
        "Post a message saying which argument about the user message is more compelling?"
    )
    while task2.is_incomplete():
        task2.run(agents=[Agent(instructions="you always pick a side")])


demo()
