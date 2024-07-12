import controlflow
from langchain_community.tools import DuckDuckGoSearchRun


def test_ddg_tool():
    task = controlflow.Task(
        "Retrieve and summarize today's two top business headlines",
        tools=[DuckDuckGoSearchRun()],
        # agent=summarizer,
        result_type=list[str],
    )
    task.run()
    assert task.is_successful()
