from controlflow import Agent, Task, flow
from controlflow.instructions import instructions

teacher = Agent(name="teacher")
student = Agent(name="student")


@flow
def demo():
    with Task("Teach a class by asking and answering 3 questions") as task:
        for _ in range(3):
            question = Task(
                "ask the student a question. Wait for the student to answer your question before asking another one.",
                str,
                agents=[teacher],
            )
            with instructions("one sentence max"):
                Task(
                    "answer the question",
                    str,
                    agents=[student],
                    context=dict(question=question),
                )

    task.run()
    return task


t = demo()
