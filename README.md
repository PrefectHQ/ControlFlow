![ControlFlow Banner](https://github.com/PrefectHQ/ControlFlow/blob/main/docs/assets/brand/controlflow_banner.png)

# ControlFlow

**ControlFlow is a Python framework for building agentic AI workflows.**

ControlFlow provides a structured, developer-focused framework for defining workflows and delegating work to LLMs, without sacrificing control or transparency:

- Create discrete, observable [tasks](https://controlflow.ai/concepts/tasks) for an AI to work on.
- Assign one or more specialized AI [agents](https://controlflow.ai/concepts/agents) to each task.
- Combine tasks into a [flow](https://controlflow.ai/concepts/flows) to orchestrate more complex behaviors.
## Example

The simplest ControlFlow workflow has one task, a default agent, and automatic thread management:

```python
import controlflow as cf

result = cf.run("Write a short poem about artificial intelligence")

print(result)
```
**Result:**
```
In circuits and code, a mind does bloom,
With algorithms weaving through the gloom.
A spark of thought in silicon's embrace,
Artificial intelligence finds its place.
```
## Why ControlFlow?

ControlFlow addresses the challenges of building AI-powered applications that are both powerful and predictable:

- 🧩 [**Task-Centric Architecture**](https://controlflow.ai/concepts/tasks): Break complex AI workflows into manageable, observable steps.
- 🔒 [**Structured Results**](https://controlflow.ai/patterns/task-results): Bridge the gap between AI and traditional software with type-safe, validated outputs.
- 🤖 [**Specialized Agents**](https://controlflow.ai/concepts/agents): Deploy task-specific AI agents for efficient problem-solving.
- 🎛️ [**Flexible Control**](https://controlflow.ai/patterns/instructions): Continuously tune the balance of control and autonomy in your workflows.
- 🕹️ [**Multi-Agent Orchestration**](https://controlflow.ai/concepts/flows): Coordinate multiple AI agents within a single workflow or task.
- 🔍 [**Native Observability**](https://github.com/PrefectHQ/prefect): Monitor and debug your AI workflows with full Prefect 3.0 support.
- 🔗 **Ecosystem Integration**: Seamlessly work with your existing code, tools, and the broader AI ecosystem.


## Installation

Install ControlFlow with `pip`:

```bash
pip install controlflow
```

Next, configure your LLM provider. ControlFlow's default provider is OpenAI, which requires the `OPENAI_API_KEY` environment variable:

```
export OPENAI_API_KEY=your-api-key
```

To use a different LLM provider, [see the LLM configuration docs](https://controlflow.ai/guides/configure-llms).


## Workflow Example

Here's a more involved example that showcases user interaction, a multi-step workflow, and structured outputs:

```python
import controlflow as cf
from pydantic import BaseModel


class ResearchProposal(BaseModel):
    title: str
    abstract: str
    key_points: list[str]


@cf.flow
def research_proposal_flow():

    # Task 1: Get the research topic from the user
    user_input = cf.Task(
        "Work with the user to choose a research topic",
        interactive=True,
    )
    
    # Task 2: Generate a structured research proposal
    proposal = cf.run(
        "Generate a structured research proposal",
        result_type=ResearchProposal,
        depends_on=[user_input]
    )
    
    return proposal


result = research_proposal_flow()

print(result.model_dump_json(indent=2))
```
<details>
<summary><i>Click to see results</i></summary>
</br>

>**Conversation:**
> ```text
> Agent: Hello! I'm here to help you choose a research topic. Do you have 
> any particular area of interest or field you would like to explore? 
> If you have any specific ideas or requirements, please share them as well.
> 
> User: Yes, I'm interested in LLM agentic workflows
> ```
> 
> **Proposal:**
> ```json
> {
>     "title": "AI Agentic Workflows: Enhancing Efficiency and Automation",
>     "abstract": "This research proposal aims to explore the development and implementation of AI agentic workflows to enhance efficiency and automation in various domains. AI agents, equipped with advanced capabilities, can perform complex tasks, make decisions, and interact with other agents or humans to achieve specific goals. This research will investigate the underlying technologies, methodologies, and applications of AI agentic workflows, evaluate their effectiveness, and propose improvements to optimize their performance.",
>     "key_points": [
>         "Introduction: Definition and significance of AI agentic workflows, Historical context and evolution of AI in workflows",
>         "Technological Foundations: AI technologies enabling agentic workflows (e.g., machine learning, natural language processing), Software and hardware requirements for implementing AI workflows",
>         "Methodologies: Design principles for creating effective AI agents, Workflow orchestration and management techniques, Interaction protocols between AI agents and human operators",
>         "Applications: Case studies of AI agentic workflows in various industries (e.g., healthcare, finance, manufacturing), Benefits and challenges observed in real-world implementations",
>         "Evaluation and Metrics: Criteria for assessing the performance of AI agentic workflows, Metrics for measuring efficiency, accuracy, and user satisfaction",
>         "Proposed Improvements: Innovations to enhance the capabilities of AI agents, Strategies for addressing limitations and overcoming challenges",
>         "Conclusion: Summary of key findings, Future research directions and potential impact on industry and society"
>     ]
> }
> ```
</details>

In this example, ControlFlow is automatically managing a `flow`, or a shared context for a series of tasks. You can switch between standard Python functions and agentic tasks at any time, making it easy to incrementally build out complex workflows. 

## Learn More

To dive deeper into ControlFlow:

- [Read the full documentation](https://controlflow.ai)
- [Explore example projects](https://controlflow.ai/examples)
- [Join our community on Slack](https://prefect.io/slack)
