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
        content="",
        response_metadata={"finish_reason": "tool_calls"},
        name="Marvin",
        id="run-0e56d606-f1fd-408d-a270-aa81c5f71c9c",
        timestamp=datetime.datetime(
            2024, 6, 20, 15, 11, 29, 321853, tzinfo=datetime.timezone.utc
        ),
        tool_calls=[
            {
                "name": f"mark_task_{task.id}_successful",
                "args": {"result": "Hello!"},
                "id": "call_Fdo3MmYA0Bg7hn5vviTrfMuJ",
            }
        ],
    )

    default_fake_llm.responses = [response]
    with record_messages() as rec_messages:
        task.run()

    assert rec_messages[0].content == response.content
    assert rec_messages[0].id == response.id
    assert rec_messages[0].tool_calls == response.tool_calls

    expected_tool_message = ToolMessage(
        content=f'Task {task.id} ("say hello") marked successful by Marvin.',
        id="3a5a2ddaef764b6986e97d79e8d0418f",
        timestamp=datetime.datetime(
            2024, 6, 20, 15, 11, 29, 508201, tzinfo=datetime.timezone.utc
        ),
        tool_call_id="call_Fdo3MmYA0Bg7hn5vviTrfMuJ",
        tool_call={
            "name": f"mark_task_{task.id}_successful",
            "args": {"result": "Hello!"},
            "id": "call_Fdo3MmYA0Bg7hn5vviTrfMuJ",
        },
        tool_result=f'Task {task.id} ("say hello") marked successful by Marvin.',
    )

    assert rec_messages[1].content == expected_tool_message.content
    assert rec_messages[1].tool_call_id == expected_tool_message.tool_call_id
    assert rec_messages[1].tool_call == expected_tool_message.tool_call


def test_record_task_messages_removes_extra_information(default_fake_llm):
    task = controlflow.Task("say hello")

    # note the hardcoded IDs in the kwargs/chunks, which will be removed
    default_fake_llm.responses = [
        AIMessage(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_Fdo3MmYA0Bg7hn5vviTrfMuJ",
                        "function": {
                            "arguments": '{"result":"Hello!"}',
                            "name": f"mark_task_{task.id}_successful",
                        },
                        "type": "function",
                    }
                ]
            },
            response_metadata={"finish_reason": "tool_calls"},
            name="Marvin",
            id="run-0e56d606-f1fd-408d-a270-aa81c5f71c9c",
            timestamp=datetime.datetime(
                2024, 6, 20, 15, 11, 29, 321853, tzinfo=datetime.timezone.utc
            ),
            tool_calls=[
                {
                    "name": f"mark_task_{task.id}_successful",
                    "args": {"result": "Hello!"},
                    "id": "call_Fdo3MmYA0Bg7hn5vviTrfMuJ",
                }
            ],
            tool_call_chunks=[
                {
                    "name": f"mark_task_{task.id}_successful",
                    "args": '{"result":"Hello!"}',
                    "id": "call_Fdo3MmYA0Bg7hn5vviTrfMuJ",
                    "index": 0,
                }
            ],
        )
    ]

    with record_messages() as rec_messages:
        task.run()

    assert rec_messages[0].additional_kwargs == {}
    assert rec_messages[0].tool_call_chunks == []
