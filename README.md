
![ControlFlow Banner](/docs/assets/brand/controlflow_banner.png)

_🚨🚧 Please note that ControlFlow is under active development ahead of its initial public release!🚧🚨_

# ControlFlow

**ControlFlow is a Python framework for building agentic AI workflows.**

ControlFlow provides a structured, developer-focused framework for defining workflows and delegating work to LLMs, without sacrificing control or transparency:

- Create discrete, observable [tasks](https://controlflow.ai/concepts/tasks) for an AI to solve.
- Assign one or more specialized AI [agents](https://controlflow.ai/concepts/agents) to each task.
- Combine tasks into a [flow](https://controlflow.ai/concepts/flows) to orchestrate more complex behaviors.

This task-centric approach allows you to harness the power of AI for complex workflows while maintaining fine-grained control. By defining clear objectives and constraints for each task, you can balance AI autonomy with precise oversight, letting you build sophisticated AI-powered applications with confidence.


## Installation

Install ControlFlow with `pip`:

```bash
pip install controlflow
```

You'll also need to configure your LLM provider. ControlFlow's default provider is OpenAI, which requires an API key via the `OPENAI_API_KEY` environment variable:
```
export OPENAI_API_KEY=your-api-key
```
You can also configure a [different LLM provider](https://controlflow.ai/guides/llms).

## Example

```python
import controlflow as cf
from pydantic import BaseModel


# create an agent to write a research report
author = cf.Agent(
    name="Deep Thought",
    instructions="Use a formal tone and clear language",
)


class ResearchTopic(BaseModel):
    title: str
    keywords: list[str]


@cf.flow
def research_workflow() -> str:
    # Task 1: the default agent will work with the user to choose a topic
    topic = cf.Task(
        "Work with the user to come up with a research topic",
        result_type=ResearchTopic,
        user_access=True,
    )

    # Task 2: the default agent will create an outline based on the topic
    outline = cf.Task("Create an outline", context=dict(topic=topic))
    
    # Task 3: the author agent will write a first draft 
    draft = cf.Task(
        "Write a first draft", 
        context=dict(outline=outline),
        agents=[author]
    )
    
    return draft


# run the workflow
result = research_workflow()
print(result)
```

ControlFlow is built on Prefect 3.0, so you can follow your flow's execution in the Prefect UI:

<img width="1427" alt="Prefect UI showing ControlFlow execution" src="https://github.com/PrefectHQ/ControlFlow/assets/153965/2dfdfb43-3afa-4709-9ec3-c66840084087">

## Why ControlFlow?

ControlFlow is designed to address the challenges of building AI-powered applications that are both powerful and predictable:

### 🧩 Task-Centric Architecture

Break complex AI workflows into manageable, observable steps. This approach ensures that AI agents operate within well-defined boundaries, making it easier to reason about and manage complex workflows.

```python
topic = cf.Task("Generate a research topic", result_type=ResearchTopic)
outline = cf.Task("Create an outline", context=dict(topic=topic))
draft = cf.Task("Write a first draft", context=dict(outline=outline))
```

### 🔒 Structured Results

Bridge the gap between AI and traditional software with type-safe outputs. By using Pydantic models, you ensure that AI-generated content always matches your application's requirements.

```python
class ResearchTopic(BaseModel):
    title: str
    keywords: list[str]

topic_task = cf.Task("Generate a topic", result_type=ResearchTopic)
```

### 🤖 Specialized Agents

Deploy task-specific AI agents for efficient problem-solving. Agents can have their own instructions, tools, and even be backed by different LLM models.

```python
researcher = cf.Agent(name="Researcher", instructions="Conduct thorough research")
writer = cf.Agent(name="Writer", instructions="Write clear, concise content")

topic_task = cf.Task("Research topic", agents=[researcher])
draft_task = cf.Task("Write draft", agents=[writer])
```

### 🔗 Ecosystem Integration

Seamlessly work with your existing code, tools, and the broader AI ecosystem. ControlFlow supports a wide range of LangChain models and tools, making it easy to incorporate cutting-edge AI capabilities.

```python
from langchain.tools import WikipediaQueryRun

research_task = cf.Task("Research topic", tools=[WikipediaQueryRun()])
```

### 🎛️ Flexible Control

Continuously tune the balance of control and autonomy in your agentic workflows. Adjust the scope and oversight of your tasks dynamically throughout the process.

```python
with cf.instructions("Be creative"):
    brainstorm_task.run()

with cf.instructions("Follow APA style strictly"):
    formatting_task.run()
```

### 🕹️ Multi-Agent Orchestration

Coordinate multiple AI agents within a single workflow - or a single task. This allows you to create complex, multi-step AI processes that leverage the strengths of different models and approaches.

```python
@cf.flow
def research_paper():
    topic = cf.Task("Choose topic", agents=[researcher])
    outline = cf.Task("Create outline", agents=[researcher, writer])
    draft = cf.Task("Write draft", agents=[writer])
    return draft
```

### 🔍 Native Observability and Debugging

Built on Prefect 3.0, ControlFlow allows you to combine agentic and traditional workflows and monitor them all in one place. This observability is crucial for debugging, optimizing performance, and ensuring that your AI applications function as intended.

```python
@cf.flow(retries=2)
def enhance_data():
    data = etl_pipeline()
    enhanced_data = cf.Task("Add topics to data", context=dict(data=data))
    return enhanced_data
```

ControlFlow empowers you to build AI workflows with confidence, maintaining control and visibility throughout the process. It offers a powerful and flexible framework for creating AI-powered applications that are transparent, maintainable, and aligned with software engineering best practices.

## Development

To install for development:

```bash
git clone https://github.com/PrefectHQ/ControlFlow.git
cd controlflow
pip install -e ".[dev]"
```

To run tests:

```bash
cd controlflow
pytest -vv
```

The ControlFlow documentation is built with [Mintlify](https://mintlify.com/). To build the documentation, first install `mintlify`:
```bash
npm i -g mintlify
```
Then run the local build:
```bash
cd controlflow/docs
mintlify dev
```
