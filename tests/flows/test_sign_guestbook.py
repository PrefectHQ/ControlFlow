from control_flow import Assistant, run_ai_task
from control_flow.core.flow import ai_flow

# define assistants

a = Assistant(name="a")
b = Assistant(name="b")
c = Assistant(name="c")


# define tools

GUESTBOOK = []


def sign(name):
    """sign your name in the guestbook"""
    GUESTBOOK.append(name)


def view_guestbook():
    """view the guestbook"""
    return GUESTBOOK


# define flow


@ai_flow
def guestbook_flow():
    run_ai_task(
        """
        Add your name to the list using the `sign` tool. All assistants must
        sign their names for the task to be complete. You can read the sign to
        see if that has happened yet. You can not sign for another assistant.
        """,
        assistants=[a, b, c],
        tools=[sign, view_guestbook],
    )


# run test


def test():
    guestbook_flow()
    assert GUESTBOOK == ["a", "b", "c"]
