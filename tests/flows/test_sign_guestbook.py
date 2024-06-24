import pytest
from controlflow import Agent, Task, flow

# define agents

a = Agent(name="a")
b = Agent(name="b")
c = Agent(name="c")


# define tools

GUESTBOOK = []


def sign(name):
    """sign your name in the guestbook"""
    GUESTBOOK.append(name)


def view_guestbook():
    """view the guestbook"""
    return GUESTBOOK


# define flow


@flow
def guestbook_flow():
    task = Task(
        """
        Add your name to the list using the `sign` tool. All agents must
        sign their names for the task to be complete. You can read the sign to
        see if that has happened yet. You can not sign for another agent.
        """,
        agents=[a, b, c],
        tools=[sign, view_guestbook],
    )
    task.run()


# run test


@pytest.mark.skip(reason="Skipping test for now")
def test():
    guestbook_flow()
    assert GUESTBOOK == ["a", "b", "c"]
