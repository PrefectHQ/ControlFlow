import pytest
from controlflow import reset_global_flow
from controlflow.settings import temporary_settings
from prefect.testing.utilities import prefect_test_harness

from .fixtures import *


@pytest.fixture(autouse=True, scope="session")
def temp_controlflow_settings():
    with temporary_settings(max_task_iterations=3):
        try:
            yield
        finally:
            # reset the global flow after each test
            reset_global_flow()


@pytest.fixture(autouse=True, scope="session")
def prefect_test_fixture():
    """
    Run Prefect against temporary sqlite database
    """
    with prefect_test_harness():
        yield
