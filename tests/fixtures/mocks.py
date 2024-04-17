from unittest.mock import Mock, patch

import pytest
from control_flow.utilities.user_access import talk_to_human


@pytest.fixture(autouse=True)
def mock_talk_to_human():
    """Return an empty default handler instead of a print handler to avoid
    printing assistant output during tests"""

    def mock_talk_to_human(message: str, get_response: bool) -> str:
        print(dict(message=message, get_response=get_response))
        return "Message sent to user."

    mock_talk_to_human.__doc__ = talk_to_human.__doc__
    with patch(
        "control_flow.utilities.user_access.mock_talk_to_human", new=talk_to_human
    ):
        yield
