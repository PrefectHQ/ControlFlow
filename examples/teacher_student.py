from controlflow import Agent, Task, flow
from controlflow.instructions import instructions

teacher = Agent(name="Teacher")
student = Agent(name="Student")


@flow
def demo():
    with Task("Teach a class by asking and answering 3 questions", agents=[teacher]):
        for _ in range(3):
            question = Task(
                "Ask the student a question.", result_type=str, agents=[teacher]
            )

            with instructions("One sentence max"):
                answer = Task(
                    "Answer the question.",
                    agents=[student],
                    context=dict(question=question),
                )

            grade = Task(
                "Assess the answer.",
                result_type=["pass", "fail"],
                agents=[teacher],
                context=dict(answer=answer),
            )

            # run each qa session, one at a time
            grade.run()


t = demo()
