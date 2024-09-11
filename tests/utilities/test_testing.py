import controlflow
from controlflow.llm.messages import AIMessage
from controlflow.utilities.testing import record_events


def test_record_events_empty():
    with record_events() as events:
        pass
    assert events == []


def test_record_task_events(default_fake_llm):
    task = controlflow.Task("say hello", id="12345")

    response = AIMessage(
        id="run-2af8bb73-661f-4ec3-92ff-d7d8e3074926",
        name="Marvin",
        role="ai",
        content="",
        tool_calls=[
            {
                "name": "mark_task_12345_successful",
                "args": {"result": "Hello!"},
                "id": "call_ZEPdV8mCgeBe5UHjKzm6e3pe",
                "type": "tool_call",
            }
        ],
    )

    default_fake_llm.set_responses([response])
    with record_events() as events:
        task.run()

    assert events[0].event == "orchestrator-message"

    assert events[1].event == "agent-message"
    assert response == events[1].ai_message

    assert events[3].event == "tool-result"
    assert events[3].tool_call == {
        "name": "mark_task_12345_successful",
        "args": {"result": "Hello!"},
        "id": "call_ZEPdV8mCgeBe5UHjKzm6e3pe",
        "type": "tool_call",
    }
    assert events[3].tool_result.model_dump() == dict(
        tool_call_id="call_ZEPdV8mCgeBe5UHjKzm6e3pe",
        str_result='Task #12345 ("say hello") marked successful.',
        is_error=False,
    )
