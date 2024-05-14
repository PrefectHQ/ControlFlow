from unittest.mock import AsyncMock, Mock, patch

import pytest
from controlflow.utilities.user_access import talk_to_human

# @pytest.fixture(autouse=True)
# def mock_talk_to_human():
#     """Return an empty default handler instead of a print handler to avoid
#     printing assistant output during tests"""

#     def mock_talk_to_human(message: str, get_response: bool) -> str:
#         print(dict(message=message, get_response=get_response))
#         return "Message sent to user."

#     mock_talk_to_human.__doc__ = talk_to_human.__doc__
#     with patch(
#         "controlflow.utilities.user_access.mock_talk_to_human", new=talk_to_human
#     ):
#         yield


@pytest.fixture
def mock_run(monkeypatch):
    """
    This fixture mocks the calls to OpenAI. Use it in a test and assign any desired side effects (like completing a task)
    to the mock object's `.side_effect` attribute.

    For example:

    def test_example(mock_run):
        task = Task(objective="Say hello")

        def side_effect():
            task.mark_complete()

        mock_run.side_effect = side_effect

        task.run()

    """
    MockRun = AsyncMock()
    monkeypatch.setattr("controlflow.core.controller.controller.Run.run_async", MockRun)
    yield MockRun
