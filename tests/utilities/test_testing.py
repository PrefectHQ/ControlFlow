import datetime

import controlflow
from controlflow.llm.messages import AIMessage, ToolMessage
from controlflow.utilities.testing import record_messages


def test_record_messages_empty():
    with record_messages() as messages:
        pass
    assert messages == []


def test_record_task_messages(default_fake_llm):
    task = controlflow.Task("say hello")

    response = AIMessage(
        agent=dict(name="Marvin"),
        id="run-2af8bb73-661f-4ec3-92ff-d7d8e3074926",
        timestamp=datetime.datetime(
            2024, 6, 23, 17, 12, 24, 91830, tzinfo=datetime.timezone.utc
        ),
        role="ai",
        content="",
        name="Marvin",
        tool_calls=[
            {
                "name": f"mark_task_{task.id}_successful",
                "args": {"result": "Hello!"},
                "id": "call_ZEPdV8mCgeBe5UHjKzm6e3pe",
            }
        ],
        is_delta=False,
    )

    default_fake_llm.set_responses([response])
    with record_messages() as rec_messages:
        task.run()

    assert rec_messages[0].content == response.content
    assert rec_messages[0].id == response.id
    assert rec_messages[0].tool_calls == response.tool_calls

    expected_tool_message = ToolMessage(
        agent=dict(name="Marvin"),
        id="cb84bb8f3e0f4245bbf5eefeee9272b2",
        timestamp=datetime.datetime(
            2024, 6, 23, 17, 12, 24, 187384, tzinfo=datetime.timezone.utc
        ),
        role="tool",
        content=f'Task {task.id} ("say hello") marked successful by Marvin.',
        name="Marvin",
        tool_call_id="call_ZEPdV8mCgeBe5UHjKzm6e3pe",
        tool_call={
            "name": f"mark_task_{task.id}_successful",
            "args": {"result": "Hello!"},
            "id": "call_ZEPdV8mCgeBe5UHjKzm6e3pe",
        },
        tool_metadata={"ignore_result": True},
    )
    assert rec_messages[1].content == expected_tool_message.content
    assert rec_messages[1].tool_call_id == expected_tool_message.tool_call_id
    assert rec_messages[1].tool_call == expected_tool_message.tool_call
