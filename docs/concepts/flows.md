# Flows

In the ControlFlow framework, a `Flow` represents a container for an AI-enhanced workflow. It serves as the top-level object that encapsulates tasks, agents, tools, and context, providing a structured environment for AI-powered applications.

## The Role of Flows

Flows play a crucial role in organizing and managing the execution of AI-powered workflows. They provide a high-level abstraction for defining the overall structure and dependencies of tasks, agents, and tools, allowing developers to focus on the desired outcomes rather than the low-level details of agent coordination and communication.

Key aspects of flows include:

- **Task Management**: Flows contain a collection of tasks that define the discrete objectives and goals of the workflow. Tasks can be added to a flow explicitly or implicitly through the use of the `@ai_task` decorator or the `Task` class.

- **Agent Coordination**: Flows manage the assignment and orchestration of agents to tasks. By default, flows are initialized with a default agent, but custom agents can be specified to handle specific tasks or parts of the workflow.

- **Tool Management**: Flows provide a centralized place to define and manage tools that are available to agents throughout the workflow. Tools can be functions or callable objects that agents can use to perform specific actions or computations.

- **Context Sharing**: Flows maintain a consistent context across tasks and agents, allowing for seamless sharing of information and state throughout the workflow. The flow's context can be accessed and modified by tasks and agents, enabling dynamic and adaptive behavior.

## Creating a Flow

To create a flow, you can use the `@flow` decorator on a Python function. The decorated function becomes the entry point for the AI-powered workflow.

```python
from controlflow import flow

@flow
def my_flow():
    # Define tasks, agents, and tools here
    ...
```

Alternatively, you can create a flow object directly using the `Flow` class:

```python
from controlflow import Flow

flow = Flow()
```

## Flow Properties

Flows have several key properties that define their behavior and capabilities:

- `thread` (Thread): The thread associated with the flow, which stores the conversation history and context.
- `tools` (list[AssistantTool | Callable]): A list of tools that are available to all agents in the flow.
- `agents` (list[Agent]): The default agents for the flow, which are used for tasks that do not specify agents explicitly.
- `context` (dict): Additional context or information that is shared across tasks and agents in the flow.

## Running a Flow

To run a flow, you can simply call the decorated function:

```python
@flow
def my_flow():
    # Define tasks, agents, and tools here
    ...

my_flow()
```

When a flow is run, it executes the defined tasks, assigning agents and tools as needed. The flow manages the context across agents.



## Conclusion

Flows are a fundamental concept in the ControlFlow framework, providing a structured and flexible way to define, organize, and execute AI-powered workflows. By encapsulating tasks, agents, tools, and context within a flow, developers can create complex and dynamic applications that leverage the power of AI while maintaining a clear and maintainable structure.

Flows abstract away the low-level details of agent coordination and communication, allowing developers to focus on defining the desired outcomes and objectives of their workflows. With the `@flow` decorator and the `Flow` class, creating and running AI-powered workflows becomes a straightforward and intuitive process.