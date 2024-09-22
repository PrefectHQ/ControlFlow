import pytest
from prefect.testing.utilities import prefect_test_harness

from .fixtures import *


@pytest.fixture(autouse=True, scope="session")
def prefect_test_fixture():
    """
    Run Prefect against temporary sqlite database
    """
    with prefect_test_harness():
        yield
