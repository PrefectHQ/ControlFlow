import pytest
from prefect.testing.utilities import prefect_test_harness

from controlflow.settings import temporary_settings

from .fixtures import *


@pytest.fixture(autouse=True, scope="session")
def temp_controlflow_settings():
    with temporary_settings(
        enable_print_handler=False,
        task_max_turns=10,
        orchestrator_max_turns=10,
        orchestrator_max_calls_per_turn=10,
    ):
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
