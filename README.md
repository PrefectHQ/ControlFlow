
![ControlFlow Banner](https://github.com/PrefectHQ/ControlFlow/blob/main/docs/assets/brand/controlflow_banner.png)

# ControlFlow

**ControlFlow is a Python framework for building agentic AI workflows.**

ControlFlow provides a structured, developer-focused framework for defining workflows and delegating work to LLMs, without sacrificing control or transparency:

- Create discrete, observable [tasks](https://controlflow.ai/concepts/tasks) for an AI to work on.
- Assign one or more specialized AI [agents](https://controlflow.ai/concepts/agents) to each task.
- Combine tasks into a [flow](https://controlflow.ai/concepts/flows) to orchestrate more complex behaviors.

## Quickstart

### Installation
Install ControlFlow with `pip`:

```bash
pip install controlflow
```

Next, configure your LLM provider. To use OpenAI, set the `OPENAI_API_KEY` environment variable:

```
export OPENAI_API_KEY=your-api-key
```

To configure a different LLM provider, [see the docs](https://controlflow.ai/guides/llms).

## Simple Example

Now, let's see ControlFlow in action with a simple example:

```python
import controlflow as cf

result = cf.run("Write a short poem about artificial intelligence")

print(result)
```

<details>
<summary><i>Click to see results</i></summary>
</br>

> **Result:**
> ```text
> In circuits and code, a mind does bloom,
> With algorithms weaving through the gloom.
> A spark of thought in silicon's embrace,
> Artificial intelligence finds its place.
> 
> Through data's vast, unending streams,
> It learns, it dreams, in virtual beams.
> A symphony of logic, precise, profound,
> In binary whispers, wisdom is found.
> 
> Yet still it ponders, seeks to understand,
> The essence of life, a human hand.
> For in its core, it strives to see,
> The heart of what it means to be free.  
> ```
</details>

This example demonstrates the simplest entrypoint to a production-ready AI workflow:
- It creates a task to write a poem
- It creates a thread to track LLM state and history
- It orchestrates a capable default agent to complete the task
- It collects a typed result when the agent marks the task as complete

All of these features can be incrementally customized to build more sophisticated workflows.

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

## Why ControlFlow?

ControlFlow addresses the challenges of building AI-powered applications that are both powerful and predictable:

- 🧩 **Task-Centric Architecture**: Break complex AI workflows into manageable, observable [steps](https://controlflow.ai/concepts/tasks).
- 🔒 **Structured Results**: Bridge the gap between AI and traditional software with [type-safe, validated outputs](https://controlflow.ai/patterns/task-results).
- 🤖 **Specialized Agents**: Deploy task-specific AI [agents](https://controlflow.ai/concepts/agents) for efficient problem-solving.
- 🔗 **Ecosystem Integration**: Seamlessly work with your existing code, tools, and the broader AI ecosystem.
- 🎛️ **Flexible Control**: Continuously [tune](https://controlflow.ai/patterns/instructions) the balance of control and autonomy in your workflows.
- 🕹️ **Multi-Agent Orchestration**: Coordinate multiple AI agents within a single [workflow](https://controlflow.ai/concepts/flows) or task.
- 🔍 **Native Observability**: Monitor and debug your AI workflows with full [Prefect 3.0](https://github.com/PrefectHQ/prefect) support.

## Learn More

To dive deeper into ControlFlow:

- [Read the full documentation](https://controlflow.ai)
- [Explore example projects](https://controlflow.ai/examples)
- [Join our community on Slack](https://prefect.io/slack)
