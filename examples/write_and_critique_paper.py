from controlflow import Agent, Task

writer = Agent(name="writer")
editor = Agent(name="editor", instructions="you always find at least one problem")
critic = Agent(name="critic")


# ai tasks:
# - automatically supply context from kwargs
# - automatically wrap sub tasks in parent
# - automatically iterate over sub tasks if they are all completed but the parent isn't?


def write_paper(topic: str) -> str:
    """
    Write a paragraph on the topic
    """
    draft = Task(
        "produce a 3-sentence draft on the topic",
        str,
        # agents=[writer],
        context=dict(topic=topic),
    )
    edits = Task("edit the draft", str, agents=[editor], depends_on=[draft])
    critique = Task("is it good enough?", bool, agents=[critic], depends_on=[edits])
    return critique


task = write_paper("AI and the future of work")
task.run()
