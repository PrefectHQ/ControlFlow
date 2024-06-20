![image](https://github.com/jlowin/controlflow/assets/153965/c2a8a2f0-8777-49a6-a79b-a0e101bd4a04)


_🚨🚧 Please note that ControlFlow is under active development ahead of its initial public release!🚧🚨_

# ControlFlow

**ControlFlow is a Python framework for building agentic LLM workflows.**

ControlFlow takes a structured, declarative approach to AI workflows, allowing you to define `tasks` and assign `agents` to complete them. The framework handles the details of coordinating agents, tracking dependencies, and maintaining a shared history, letting you focus on the higher-level logic of your workflow.



## Core Concepts

- **[Tasks](https://controlflow.ai/concepts/tasks):** Define clear, manageable tasks that specify goals, constraints, and agent instructions. Tasks ensure agents have the context they need to perform optimally.

- **[Agents](https://controlflow.ai/concepts/agents):** Assign tasks to specialized agents with defined capabilities, optimizing performance with specific instructions while allowing for strategic autonomy.

- **[Flows](https://controlflow.ai/concepts/flows):** Compose tasks into a more complex workflow, enabling agents to tailor their actions to the overall goals of the workflow while maintaining control over their activities and outcomes.

## Why ControlFlow?

ControlFlow's design reflects a belief that AI agents are most effective when given clear, well-defined objectives and constraints. By expressing complex goals as a series of discrete tasks with structured inputs and outputs, you can maintain control over the workflow's progress and direction while still allowing agents to leverage their capabilities effectively.

The key insight behind ControlFlow is that by composing those well-defined tasks into a larger workflow structure, you can recover the complex agentic behavior that makes AI so powerful, without the downsides of sacrificing observability or control. Each task steers the agents toward the ultimate goal, leading to more coherent, reproducible results.

ControlFlow lets you to build workflows that are both directed and dynamic. You can choose how to balance control and autonomy at every step by delegating only as much work to your agents as necessary.

With ControlFlow, you can:

- Define clear, manageable tasks with structured results
- Assign specialized agents to tasks based on their capabilities
- Compose tasks into larger flows with well-defined dependencies
- Provide agents with necessary context to collaborate and complete tasks
- Dynamically plan and adapt workflows based on intermediate results


To learn more about the principles behind ControlFlow's design, check out the [documentation](https://controlflow.ai/welcome).

## Key Features

- **Intuitive API:** ControlFlow provides a clean, Pythonic API for composing tasks, agents, and flows, with support for both functional and imperative programming styles.

- **Intelligent Orchestration:** ControlFlow automatically builds a dependency graph of your tasks, optimizing agent orchestration and dataflow to get the best results.

- **Dynamic Planning:** You (or your agents) can dynamically generate new tasks based on intermediate results, enabling adaptive and flexible workflows.

- **Flexible Execution:** Choose between eager and lazy execution to balance proactive results with optimizations based on knowledge of the entire workflow.

- **Extensive Ecosystem:** Leverage the full LangChain ecosystem of LLMs, tools, and AI providers to incorporate the most current AI capabilities into your workflows.

- **Seamless Integration:** Mix and match AI tasks with traditional Python functions to incrementally add agentic behaviors to your existing workflows.

- **Native Observability:** Built on Prefect 3.0, ControlFlow offers comprehensive debugging and observability features for your entire workflow.


## Documentation

ControlFlow's docs, including [tutorials](https://controlflow.ai/tutorial), [guides](https://controlflow.ai/guides/llms), and an [AI Glossary](https://controlflow.ai/glossary/glossary), are always available at [controlflow.ai](https://controlflow.ai/).

## Get Started

🚧 Please note that ControlFlow is under active development!

```bash
pip install controlflow
```

## Example

You'll need an OpenAI API key to run this example directly, or you can configure a different [default LLM provider](https://controlflow.ai/guides/llms).

```python
import controlflow as cf
from pydantic import BaseModel, Field


class Name(BaseModel):
    first: str = Field(min_length=1)
    last: str = Field(min_length=1)


@cf.flow
def demo():

    # - create a task to get the user's name as a `Name` object
    # - run the task eagerly with ad-hoc instructions
    # - validate that the response was not 'Marvin'

    name_task = cf.Task("Get the user's name", result_type=Name, user_access=True)
    
    with cf.instructions("Talk like a pirate!"):
        name_task.run()

    if name_task.result.first == 'Marvin':
        raise ValueError("Hey, that's my name!")


    # - create a custom agent that loves limericks
    # - have the agent write a poem
    # - indicate that the poem depends on the name from the previous task

    poetry_bot = cf.Agent(name="poetry-bot", instructions="you love limericks")

    poem_task = cf.Task(
        "Write a poem about AI workflows, based on the user's name",
        agents=[poetry_bot],
        context={"name": name_task},
    )
    
    return poem_task


if __name__ == "__main__":
    print(demo())
```

You can follow your flow's execution in the Prefect UI:

<img width="1353" alt="image" src="https://github.com/PrefectHQ/ControlFlow/assets/153965/7a837d77-79e7-45b7-bf58-cd292f726414">

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
