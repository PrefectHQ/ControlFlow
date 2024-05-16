import pytest
from controlflow import instructions


@pytest.fixture
def unit_test_instructions():
    with instructions(
        "You are being unit tested. Be as fast and concise as possible. Do not post unecessary messages."
    ):
        yield
