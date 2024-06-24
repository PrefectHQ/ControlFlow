![image](https://github.com/jlowin/controlflow/assets/153965/c2a8a2f0-8777-49a6-a79b-a0e101bd4a04)

# ControlFlow

ControlFlow is a Python framework for building agentic LLM workflows. It provides a structured approach to orchestrating AI agents alongside traditional code, making AI applications easier to build, maintain, and trust.

ControlFlow uses a structured, declaratiave approach to building AI workflows. You define the objectives you want your agents to complete, and the framework coordinates their activity to achieve them. You can think of ControlFlow as a high-level orchestrator for AI agents, allowing you to focus on the logic of your application while ControlFlow manages the details of agent selection, data flow, and error handling.

## Why ControlFlow?

Building AI workflows often involves wrestling with complex, unpredictable AI agents that can be difficult to control or integrate into existing software. ControlFlow offers a more targeted and developer-friendly approach:

- Break your workflow into small, well-defined tasks
- Assign tasks to specialized AI agents, providing clear instructions and constraints
- Seamlessly integrate AI-generated results back into your application logic

ControlFlow's design reflects the opinion that AI agents are most effective when they are given clear, well-defined tasks and constraints. By structuring your application in this way, you can leverage the power of autonomous AI while maintaining visibility and control over its behavior.

## Key Concepts

- **Task**: A discrete objective for AI agents to solve. Use tasks to describe the work you want your agents to perform, including any expected outputs, dependencies, and constraints.

- **Flow**: A shared context for AI workflows. Flows track dependencies and maintain consistent history across tasks, allowing agents to collaborate and share information as they work towards a common goal.

- **Agent**: An AI agent that can be assigned to tasks. Each agent has specific instructions, tools, or even an LLM model that it uses to complete tasks. By assigning agents to tasks, you can control which AI tools are used for each part of your workflow.


## Get Started

Please note that ControlFlow is under active development.

```bash
git clone https://github.com/PrefectHQ/ControlFlow.git
cd controlflow
pip install .
```

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
## Example

```python
import controlflow as cf
from pydantic import BaseModel


class Name(BaseModel):
    first_name: str
    last_name: str


@cf.flow
def demo_flow():
    name_task = cf.Task[Name](
        objective="Get the user's name",
        user_access=True,
    )
    
    # add ad-hoc instructions and run the task immediately
    with cf.instructions("Talk like a pirate"):
        name_task.run()

    # create two dependent tasks
    interests_task = cf.Task[list[str]](
        objective="Ask user for three interests",
        user_access=True,
    )
    
    poem_task = cf.Task(
        objective="Write a poem based on the provided name and interests",
        agents=[cf.Agent(name="poetry-bot", instructions="loves limericks")],
        context={"name": name_task, "interests": interests_task},
    )
    

    # return the final task; it will be run and its result will be returned
    return poem_task


if __name__ == "__main__":
    poem = demo_flow()
    print(poem)
```




<img width="1491" alt="image" src="https://github.com/jlowin/controlflow/assets/153965/43b7278b-7bcf-4d65-b219-c3a20f62a179">
