# Tasks

In ControlFlow, a `Task` is the fundamental unit of work that represents a specific objective or goal within an AI-powered workflow. Tasks are the primary means of defining and structuring the desired outcomes of an application, acting as a bridge between the AI agents and the application logic.

## Modeling Application State with Tasks

One of the key roles of tasks in ControlFlow is to model the internal state of an AI-powered application. By defining tasks with clear objectives, dependencies, and result types, developers can create a structured representation of the application's desired outcomes and the steps required to achieve them.

Each task contributes to the overall state of the application, either by producing a specific result or by performing an action that affects the behavior of other tasks or agents. This allows developers to build complex workflows where the results of one task can be used as inputs for subsequent tasks, creating a dynamic and responsive application state.

## The Philosophy of Declarative Task-Based Workflows

ControlFlow embraces a declarative, task-based approach to defining AI workflows. Instead of focusing on directly controlling the AI agents' behavior, which can be unpredictable and difficult to manage, ControlFlow encourages developers to define clear, discrete tasks that specify what needs to be accomplished.

By defining tasks with specific objectives, inputs, outputs, and dependencies, developers can create a structured workflow that guides the AI agents towards the desired outcomes. This declarative approach allows for more predictable and manageable AI integrations, as the agents are dispatched to complete well-defined tasks rather than being controlled through complex prompts that attempt to steer their behavior.

The task-based workflow also promotes modularity and reusability. Tasks can be easily composed, reordered, and reused across different workflows, enabling developers to build complex AI applications by combining smaller, self-contained units of work.

## Defining Tasks

In ControlFlow, tasks can be defined using the `Task` class or the `@task` decorator.

### Using the `Task` Class

The `Task` class provides a flexible way to define tasks by specifying various properties and requirements, such as the objective, instructions, assigned agents, context, dependencies, and more.

```python
from controlflow import Task

interests = Task(
    objective="Ask user for three interests",
    result_type=list[str],
    user_access=True,
    instructions="Politely ask the user to provide three of their interests or hobbies."
)
```

### Using the `@task` Decorator

The `@task` decorator offers a convenient way to define and execute tasks using familiar Python functions. The decorator automatically infers the task's objective from the function name, instructions from its docstring, context from the function arguments, and result type from the return annotation. Various additional arguments can be passed to the decorator.

```python
from controlflow import task

@task(user_access=True)
def get_user_name() -> str:
    "Politely ask the user for their name."
    pass
```

When a decorator-based task is called, it automatically invokes the `run()` method, executing the task and returning its result (or raising an exception if the task fails).

## Task Properties

Tasks have several key properties that define their behavior and requirements:

- `objective` (str): A brief description of the task's goal or desired outcome.
- `instructions` (str, optional): Detailed instructions or guidelines for completing the task.
- `agents` (list[Agent], optional): The AI agents assigned to work on the task.
- `context` (dict, optional): Additional context or information required for the task.
- `result_type` (type, optional): The expected type of the task's result.
- `tools` (list[AssistantTool | Callable], optional): Tools or functions available to the agents for completing the task.
- `user_access` (bool, optional): Indicates whether the task requires human user interaction.

## Task Execution and Results

Tasks can be executed using the `run()` method, which intelligently selects the appropriate agents and iterates until the task is complete. The `run_once()` method allows for more fine-grained control, executing a single step of the task with a selected agent.

The `result` property of a task holds the outcome or output of the task execution. By specifying a clear `result_type`, developers can ensure that the task's result is structured and can be easily integrated into the application logic. This makes it possible to create complex workflows where the results of one task can be used as inputs for subsequent tasks.

It's important to note that not all tasks require a result. In some cases, a task may be designed to perform an action or produce a side effect without returning a specific value. For example, a task could be used to prompt an agent to say something on the internal thread, which could be useful for later tasks or agents. In such cases, the `result_type` can be set to `None`.

## Task Dependencies and Composition

Tasks can have dependencies on other tasks, which must be completed before the dependent task can be executed. Dependencies can be specified explicitly using the `depends_on` property or implicitly by providing tasks as values in the `context` dictionary.

Tasks can also be composed hierarchically using subtasks. Subtasks are tasks that are part of a larger, parent task. They can be added to a parent task using the `subtasks` property or by creating tasks within a context manager (e.g., `with Task():`). Parent tasks cannot be completed until all their subtasks are finished, although subtasks can be skipped using a special `skip` tool.

## Talking to Humans

ControlFlow provides a built-in mechanism for tasks to interact with human users. By setting the `user_access` property to `True`, a task can indicate that it requires human input or feedback to be completed.

When a task with `user_access=True` is executed, the AI agents assigned to the task will be given access to a special `talk_to_human` tool. This tool allows the agents to send messages to the user and receive their responses, enabling a conversation between the AI and the human.

Here's an example of a task that interacts with a human user:

```python
@task(user_access=True)
def get_user_feedback(product_name: str) -> str:
    """
    Ask the user for their feedback on a specific product.
    
    Example conversation:
    AI: What do you think about the new iPhone?
    Human: I think it's a great phone with impressive features, but it's a bit expensive.
    AI: Thank you for your feedback!
    """
    pass
```

In this example, the AI agent will use the `talk_to_human` tool to ask the user for their feedback on the specified product. The agent can then process the user's response and store it in the task's `result` property, making it available for use in subsequent tasks or other parts of the application.

By leveraging the `user_access` property and the `talk_to_human` tool, developers can create AI-powered workflows that seamlessly integrate human input and feedback, enabling more interactive and user-centric applications.