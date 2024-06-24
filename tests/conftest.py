import pytest
from controlflow.llm.messages import MessageType
from controlflow.settings import temporary_settings
from prefect.testing.utilities import prefect_test_harness

from .fixtures import *


@pytest.fixture(autouse=True, scope="session")
def temp_controlflow_settings():
    with temporary_settings(max_iterations=10, enable_experimental_tui=False):
        yield


@pytest.fixture(autouse=True)
def reset_settings_after_each_test():
    with temporary_settings():
        yield


@pytest.fixture(autouse=True, scope="session")
def prefect_test_fixture():
    """
    Run Prefect against temporary sqlite database
    """
    with prefect_test_harness():
        yield
