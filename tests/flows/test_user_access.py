import pytest
from control_flow import Agent, flow, run_ai

# define assistants

user_agent = Agent(name="user-agent", user_access=True)
non_user_agent = Agent(name="non-user-agent", user_access=False)


def test_no_user_access_fails():
    @flow
    def user_access_flow():
        run_ai(
            "This task requires human user access. Inform the user that today is a good day.",
            agents=[non_user_agent],
        )

    with pytest.raises(ValueError):
        user_access_flow()


def test_user_access_agent_succeeds():
    @flow
    def user_access_flow():
        run_ai(
            "This task requires human user access. Inform the user that today is a good day.",
            agents=[user_agent],
        )

    assert user_access_flow()


def test_user_access_task_succeeds():
    @flow
    def user_access_flow():
        run_ai(
            "This task requires human user access. Inform the user that today is a good day.",
            agents=[non_user_agent],
            user_access=True,
        )

    assert user_access_flow()


def test_user_access_agent_and_task_succeeds():
    @flow
    def user_access_flow():
        run_ai(
            "This task requires human user access. Inform the user that today is a good day.",
            agents=[user_agent],
            user_access=True,
        )

    assert user_access_flow()
