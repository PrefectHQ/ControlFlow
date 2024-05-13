# Tasks

In ControlFlow, a `Task` is the fundamental unit of work that represents a specific objective or goal within an AI-powered workflow. Tasks are the primary means of defining and structuring the desired outcomes of an application, acting as a bridge between the AI agents and the application logic.

## The Philosophy of Task-Centric Workflows

ControlFlow takes a unique approach to AI-powered workflows by placing tasks at the center of the design process. Instead of focusing on directly controlling the AI agents' behavior, which can be unpredictable and difficult to manage, ControlFlow encourages developers to define clear, discrete tasks that specify what needs to be accomplished.

By defining tasks with specific objectives, inputs, outputs, and dependencies, developers can create a structured workflow that guides the AI agents towards the desired outcomes. This task-centric approach allows for more predictable and manageable AI integrations, as the agents are dispatched to complete well-defined tasks rather than being controlled through a complex script that attempts to respond to their stochastic behavior.

## Defining Tasks

In ControlFlow, tasks are typically defined using the `Task` class, which provides a flexible and expressive way to specify task properties and requirements. However, for convenience, ControlFlow also offers the `@ai_task` decorator, which can be used to define tasks using Python functions.

### Using the `Task` Class

The `Task` class is the standard way to define tasks in ControlFlow. It allows you to specify various properties and requirements for a task, such as its objective, instructions, assigned agents, context, dependencies, and more.

```python
from controlflow import Task

interests = Task(
    objective="Ask user for three interests",
    result_type=list[str],
    user_access=True,
    instructions="Politely ask the user to provide three of their interests or hobbies."
)
```

### Using the `@ai_task` Decorator

The `@ai_task` decorator provides a convenient way to define tasks using Python functions. The decorator accepts many of the same arguments as the `Task` class, and it automatically infers the task's objective, context, and result type from the function definition.

```python
from controlflow import ai_task

@ai_task(user_access=True)
def get_user_name() -> str:
    "Politely ask the user for their name."
    pass
```

When a decorator-based task is called, it automatically invokes the `run()` method, executing the task and returning its result.

## Task Properties

Tasks have several key properties that define their behavior and requirements:

- `objective` (str): A brief description of the task's goal or desired outcome.
- `instructions` (str, optional): Detailed instructions or guidelines for completing the task.
- `agents` (list[Agent], optional): The AI agents assigned to work on the task.
- `context` (dict, optional): Additional context or information required for the task.
- `subtasks` (list[Task], optional): A list of subtasks that are part of the main task.
- `depends_on` (list[Task], optional): Tasks that must be completed before this task can be executed.
- `result_type` (type, optional): The expected type of the task's result.
- `tools` (list[AssistantTool | Callable], optional): Tools or functions available to the agents for completing the task.
- `user_access` (bool, optional): Indicates whether the task requires human user interaction.

## Task Execution and Results

Tasks can be executed using the `run()` method, which intelligently selects the appropriate agents and iterates until the task is complete. The `run_once()` method allows for more fine-grained control, executing a single step of the task with a selected agent.

The `result` property of a task holds the outcome or output of the task execution. By specifying a clear `result_type`, developers can ensure that the task's result is structured and can be easily integrated into the application logic. This makes it possible to create complex workflows where the results of one task can be used as inputs for subsequent tasks.

## Task Dependencies and Subtasks

Tasks can have dependencies on other tasks, which must be completed before the dependent task can be executed. Dependencies can be specified explicitly using the `depends_on` property or implicitly by providing tasks as values in the `context` dictionary.

Subtasks are tasks that are part of a larger, parent task. They can be added to a parent task using the `add_subtask()` method or by creating tasks within a context manager (e.g., `with Task():`). Parent tasks cannot be completed until all their subtasks are finished, although subtasks can be skipped using a special `skip` tool.

## Modeling Application State with Tasks

In ControlFlow, tasks are used to model the internal state of an AI-powered application. By defining tasks with clear objectives, dependencies, and result types, developers can create a structured representation of the application's desired outcomes and the steps required to achieve them.

This task-centric approach allows for a more modular and manageable integration of AI capabilities into traditional software development workflows. By focusing on defining what needs to be done rather than attempting to control the AI's behavior directly, ControlFlow enables developers to create robust, scalable, and maintainable AI-powered applications.
