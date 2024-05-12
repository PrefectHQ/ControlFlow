import pytest
from controlflow.settings import temporary_settings


@pytest.fixture(autouse=True, scope="session")
def disable_global_flow():
    with temporary_settings(enable_global_flow=False):
        yield
