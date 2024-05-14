import pytest
from controlflow.settings import temporary_settings

from .fixtures import *


@pytest.fixture(autouse=True, scope="session")
def temp_settings():
    with temporary_settings(enable_global_flow=False, max_task_iterations=3):
        yield
