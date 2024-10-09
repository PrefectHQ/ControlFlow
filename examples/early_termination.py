from pydantic import BaseModel

import controlflow as cf
from controlflow.orchestration.conditions import AnyComplete, MaxLLMCalls


class ResearchPoint(BaseModel):
    topic: str
    key_findings: list[str]


@cf.flow
def research_workflow(topics: list[str]):
    if len(topics) < 2:
        raise ValueError("At least two topics are required")

    research_tasks = [
        cf.Task(f"Research {topic}", result_type=ResearchPoint) for topic in topics
    ]

    # Run tasks until either two topics are researched or 15 LLM calls are made
    results = cf.run_tasks(
        research_tasks,
        instructions="Research only one topic at a time.",
        run_until=(
            AnyComplete(
                min_complete=2
            )  # stop after two tasks (if there are more than two topics)
            | MaxLLMCalls(15)  # or stop after 15 LLM calls, whichever comes first
        ),
    )

    completed_research = [r for r in results if isinstance(r, ResearchPoint)]
    return completed_research


if __name__ == "__main__":
    # Example usage
    topics = [
        "Artificial Intelligence",
        "Quantum Computing",
        "Biotechnology",
        "Renewable Energy",
    ]
    results = research_workflow(topics)

    print(f"Completed research on {len(results)} topics:")
    for research in results:
        print(f"\nTopic: {research.topic}")
        print("Key Findings:")
        for finding in research.key_findings:
            print(f"- {finding}")
