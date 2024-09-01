from unittest.mock import MagicMock

from langchain_community.tools import DuckDuckGoSearchRun

import controlflow


def test_ddg_tool(default_fake_llm, monkeypatch):
    events = [
        {
            "content": "",
            "additional_kwargs": {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_zL53FKAth0u3NSglQyW4hSzu",
                        "function": {
                            "arguments": '{"query": "top business headlines today"}',
                            "name": "duckduckgo_search",
                        },
                        "type": "function",
                    },
                    {
                        "index": 1,
                        "id": "call_USrYJBHxxCpqJxUospzHQZfo",
                        "function": {
                            "arguments": '{"query": "today\'s top business news"}',
                            "name": "duckduckgo_search",
                        },
                        "type": "function",
                    },
                ]
            },
            "response_metadata": {
                "finish_reason": "tool_calls",
                "model_name": "gpt-4o-2024-05-13",
                "system_fingerprint": "fp_dd932ca5d1",
            },
            "type": "ai",
            "name": "Marvin",
            "id": "run-2e65c19b-c631-42f3-a54d-1e4b7298971a",
            "example": False,
            "tool_calls": [
                {
                    "name": "duckduckgo_search",
                    "args": {"query": "top business headlines today"},
                    "id": "call_zL53FKAth0u3NSglQyW4hSzu",
                    "type": "tool_call",
                }
            ],
            "invalid_tool_calls": [],
            "usage_metadata": None,
            "tool_call_chunks": [
                {
                    "name": "duckduckgo_search",
                    "args": '{"query": "top business headlines today"}',
                    "id": "call_zL53FKAth0u3NSglQyW4hSzu",
                    "index": 0,
                    "type": "tool_call_chunk",
                }
            ],
        }
    ]
    default_fake_llm.set_responses(events)
    tool = DuckDuckGoSearchRun()
    mock_run = MagicMock(return_value=["headline 1", "headline 2"])
    tool.__dict__.update(dict(_run=mock_run))
    task = controlflow.Task(
        "Retrieve and summarize today's two top business headlines",
        tools=[tool],
        # agent=summarizer,
        result_type=list[str],
    )
    task.run(max_turns=1, max_calls_per_turn=1)
    mock_run.assert_called_once()
