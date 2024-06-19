![image](https://github.com/jlowin/controlflow/assets/153965/c2a8a2f0-8777-49a6-a79b-a0e101bd4a04)

# ControlFlow

**ControlFlow is a Python framework for building agentic LLM workflows.**

ControlFlow takes a structured, declarative approach to AI workflows, allowing you to define `tasks` and assign `agents` to complete them. The framework handles the details of coordinating agents, tracking dependencies, and maintaining a shared history, letting you focus on the higher-level logic of your workflow.


## Why ControlFlow?

The goal of this framework is to let you build AI workflows with confidence. 

ControlFlow's design reflects the opinion that AI agents are most effective when given clear, well-defined tasks and constraints. By breaking down complex goals into manageable tasks that can be composed into a larger workflow, we can harness the power of AI while maintaining precise control over its overall direction and outcomes.

At the core of every agentic workflow is a loop that repeatedly invokes an LLM to make progress towards a goal. At every step of this loop, ControlFlow lets you continuously tune how much autonomy your agents have. This allows you to strategically balance control and autonomy throughout the workflow, ensuring that the AI's actions align closely with your objectives while still leveraging its creative capabilities. For some tasks, you may provide specific, constrained instructions; for others, you can give the AI more open-ended goals.

ControlFlow provides the tools and abstractions to help you find this balance, letting you build agentic workflows tailored to your specific needs and objectives. You can delegate only as much - or as little - work to your agents as you need, while maintaining full visibility and control over the entire process. 

With ControlFlow, you can:

- Split your workflow into well-defined [tasks](https://controlflow.ai/concepts/tasks), each with clear objectives and constraints
- Assign tasks to specialized AI [agents](https://controlflow.ai/concepts/agents), each with its own capabilities, instructions, and LLM model
- Seamlessly integrate AI-generated results back into your [application](https://controlflow.ai/concepts/flows), including non-agentic workflow logic
  
To learn more about the principles behind ControlFlow's design, check out the [documentation](https://controlflow.ai/welcome).

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
## Documentation

ControlFlow's docs are available at [controlflow.ai](https://controlflow.ai/).

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
