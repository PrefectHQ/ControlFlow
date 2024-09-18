from unittest.mock import MagicMock

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel

import controlflow
from controlflow.events.events import AIMessage, ToolCall


class LCBaseToolInput(BaseModel):
    x: int


class LCBaseTool(BaseTool):
    name: str = "TestTool"
    description: str = "A test tool"
    args_schema: type[BaseModel] = LCBaseToolInput

    def _run(self, x: int) -> str:
        return str(x)


def test_lc_base_tool(default_fake_llm, monkeypatch):
    events = [
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    id="abc",
                    name="TestTool",
                    args={"x": 3},
                )
            ],
        )
    ]
    default_fake_llm.set_responses(events)
    tool = LCBaseTool()
    mock_run = MagicMock(return_value="3")
    tool.__dict__.update(dict(_run=mock_run))
    task = controlflow.Task(
        "Use the tool",
        tools=[tool],
        result_type=str,
    )
    task.run(max_agent_turns=1)
    mock_run.assert_called()


def test_ddg_tool(default_fake_llm, monkeypatch):
    events = [
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    id="abc",
                    name="duckduckgo_search",
                    args={"query": "top business headlines"},
                )
            ],
        )
    ]
    default_fake_llm.set_responses(events)
    tool = DuckDuckGoSearchRun()
    mock_run = MagicMock(return_value=["headline 1", "headline 2"])
    tool.__dict__.update(dict(_run=mock_run))
    task = controlflow.Task(
        "Retrieve and summarize today's two top business headlines",
        tools=[tool],
        result_type=list[str],
    )
    task.run(max_agent_turns=1)
    mock_run.assert_called()
