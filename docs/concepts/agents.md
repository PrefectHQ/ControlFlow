# Agents

In the ControlFlow framework, an `Agent` represents an AI entity capable of performing tasks and interacting with other agents to achieve desired outcomes. Agents are powered by specialized AI models that excel at specific tasks, such as text generation, decision-making based on unstructured data, or engaging in interactive conversations.

## The Role of Agents in ControlFlow

Agents play a crucial role in the execution of tasks within the ControlFlow framework. When a task is defined and added to a workflow, it is assigned to one or more agents responsible for completing the task based on the provided objectives, instructions, and context.

ControlFlow treats agents as autonomous entities with their own knowledge, capabilities, and tools. By assigning tasks to agents and allowing them to collaborate and communicate with each other, ControlFlow enables the creation of complex AI-powered workflows that can adapt to different scenarios and requirements.

## Defining Agents

To create an agent in ControlFlow, you can use the `Agent` class, which provides a flexible way to define an agent's properties and capabilities.

```python
from controlflow import Agent

writer_agent = Agent(
    name="WriterAgent",
    description="An AI agent specializing in creative writing tasks.",
    tools=[generate_text, summarize_text],
    user_access=False
)
```

In this example, we define an agent named "WriterAgent" with a description of its specialization. We also specify the tools available to the agent, which are functions or callable objects that the agent can use to perform specific actions or computations. The `user_access` parameter indicates whether the agent is allowed to interact directly with human users.

## Agent Properties

Agents have several key properties that define their characteristics and capabilities:

- `name` (str): The name of the agent, used for identification and communication purposes.
- `description` (str, optional): A brief description of the agent's specialization or role.
- `tools` (list[AssistantTool | Callable], optional): A list of tools or functions available to the agent for performing tasks.
- `user_access` (bool, optional): Indicates whether the agent is allowed to interact directly with human users.

## Assigning Tasks to Agents

When defining a task using the `Task` class or the `@ai_task` decorator, you can specify the agents responsible for completing the task by setting the `agents` parameter.

```python
from controlflow import Task

write_story_task = Task(
    objective="Write a short story about a mysterious artifact.",
    result_type=str,
    agents=[writer_agent, editor_agent]
)
```

In this example, we assign the "write_story_task" to two agents: "writer_agent" and "editor_agent". These agents will collaborate to complete the task based on their individual capabilities and tools.

## Agent Execution and Communication

During the execution of a workflow, agents assigned to tasks take turns performing actions and communicating with each other to progress towards completing the tasks. The `run()` method of a task automatically handles the selection and iteration of agents until the task is complete.

Agents can communicate with each other by posting messages within the context of a task. These messages are visible to all agents involved in the task and can be used to share information, provide updates, or request assistance.

```python
from controlflow import Flow

with Flow():
    story_task = Task(
        objective="Write a short story and provide feedback.",
        result_type=str,
        agents=[writer_agent, editor_agent]
    )
    result = story_task.run()
```

In this example, the "writer_agent" and "editor_agent" will take turns working on the "story_task". They can communicate with each other by posting messages within the task's context, allowing them to collaborate and provide feedback until the task is complete.

## Agent Tools and User Access

Agents can be equipped with tools, which are functions or callable objects that provide additional capabilities or actions that the agent can perform. These tools can be used by the agent during task execution to perform specific computations, access external resources, or interact with other systems.

The `user_access` property of an agent determines whether the agent is allowed to interact directly with human users. If `user_access` is set to `True`, the agent can use special tools, such as `talk_to_human()`, to send messages to and receive input from human users. This feature is useful for tasks that require human feedback or intervention.


## Conclusion

Agents are a fundamental concept in the ControlFlow framework, representing the AI entities responsible for executing tasks and collaborating to achieve desired outcomes. By defining agents with specific capabilities, tools, and user access permissions, and assigning them to tasks within a workflow, you can create powerful and adaptable AI-powered applications.

ControlFlow provides a flexible and intuitive way to orchestrate the interaction between agents and tasks, enabling developers to focus on defining the objectives and dependencies of their workflows while the framework handles the complexities of agent coordination and communication.