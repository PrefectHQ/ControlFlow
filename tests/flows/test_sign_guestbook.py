from controlflow import Agent, flow, run_ai

# define assistants

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
    run_ai(
        """
        Add your name to the list using the `sign` tool. All assistants must
        sign their names for the task to be complete. You can read the sign to
        see if that has happened yet. You can not sign for another assistant.
        """,
        agents=[a, b, c],
        tools=[sign, view_guestbook],
    )


# run test


def test():
    guestbook_flow()
    assert GUESTBOOK == ["a", "b", "c"]
