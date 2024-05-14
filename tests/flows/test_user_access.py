import pytest
from controlflow import Agent, Task, flow

pytest.skip("Skipping the entire file", allow_module_level=True)

# define assistants
user_agent = Agent(name="user-agent", user_access=True)
non_user_agent = Agent(name="non-user-agent", user_access=False)


def test_no_user_access_fails():
    @flow
    def user_access_flow():
        task = Task(
            "This task requires human user access. Inform the user that today is a good day.",
            agents=[non_user_agent],
        )
        task.run()

    with pytest.raises(ValueError):
        user_access_flow()


def test_user_access_agent_succeeds():
    @flow
    def user_access_flow():
        task = Task(
            "This task requires human user access. Inform the user that today is a good day.",
            agents=[user_agent],
        )
        task.run()

    assert user_access_flow()


def test_user_access_task_succeeds():
    @flow
    def user_access_flow():
        task = Task(
            "This task requires human user access. Inform the user that today is a good day.",
            agents=[non_user_agent],
            user_access=True,
        )
        task.run()

    assert user_access_flow()


def test_user_access_agent_and_task_succeeds():
    @flow
    def user_access_flow():
        task = Task(
            "This task requires human user access. Inform the user that today is a good day.",
            agents=[user_agent],
            user_access=True,
        )
        task.run()

    assert user_access_flow()
