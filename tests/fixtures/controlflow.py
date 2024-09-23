import pytest

import controlflow
from controlflow.events.history import InMemoryHistory
from controlflow.llm.messages import BaseMessage
from controlflow.settings import temporary_settings
from controlflow.utilities.testing import FakeLLM


@pytest.fixture(autouse=True, scope="session")
def temp_controlflow_settings():
    with temporary_settings(
        pretty_print_agent_events=False,
        log_all_messages=True,
        log_level="DEBUG",
        orchestrator_max_agent_turns=10,
        orchestrator_max_llm_calls=10,
    ):
        yield


@pytest.fixture(autouse=True)
def reset_settings_after_each_test():
    with temporary_settings():
        yield


def temp_controlflow_defaults(monkeypatch):
    # use in-memory history
    monkeypatch.setattr(
        controlflow.defaults,
        "history",
        InMemoryHistory(),
    )


@pytest.fixture(autouse=True)
def reset_defaults_after_each_test(monkeypatch):
    """
    Monkeypatch defaults to themselves, which will automatically reset them after every test
    """
    for k, v in controlflow.defaults.__dict__.items():
        monkeypatch.setattr(controlflow.defaults, k, v)
    yield


@pytest.fixture()
def fake_llm() -> FakeLLM:
    return FakeLLM(responses=[])


@pytest.fixture()
def default_fake_llm(fake_llm) -> FakeLLM:
    controlflow.defaults.model = fake_llm
    return fake_llm
