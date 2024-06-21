
## ./docs/tutorial.mdx
---
title: "Tutorial"
---

Welcome to ControlFlow! 


ControlFlow is a declarative framework for building agentic workflows. That means that you define the objectives you want an AI agent to complete, and ControlFlow handles the rest. You can think of ControlFlow as a high-level orchestrator for AI agents, allowing you to focus on the logic of your application while ControlFlow manages the details of agent selection, data flow, and error handling.

In this tutorial, we’ll introduce the basics of ControlFlow, including tasks, flows, agents, and more. By the end, you’ll have a solid understanding of how to create and run complex agentic workflows. 

The tutorial is divided into the following sections:
- [Hello, world](#hello-world): Your first task
- [Hello, user](#hello-user): Interacting with users
- [Hello, tasks](#hello-tasks): Chaining tasks together
- [Hello, flow](#hello-flow): Building a flow
- [Hello, agents](#hello-agents): Working with agents

---

## Install ControlFlow

To run the code in this tutorial, you'll need to install ControlFlow and configure API keys for your LLM provider. Please see the [installation](/installation) instructions for more information.

---
## Hello, world

### Creating a task
The starting point of every agentic workflow is a `Task`. Each task represents an objective that we want an AI agent to complete. Let’s create a simple task to say hello:
<CodeGroup>
    
```python Code
import controlflow as cf

hello_task = cf.Task("say hello")
```

```python Result
>> print(hello_task)

Task(
    objective='say hello',
    status=<TaskStatus.INCOMPLETE: 'INCOMPLETE'>,
    result_type=<class 'str'>,
    result=None,
    ... # other fields omitted
)
```
</CodeGroup>


If you examine this `Task` object, you’ll notice a few important things: it's in an `INCOMPLETE` state and while it has no `result` value, its `result_type` is a string. This means that the task has not been completed yet, but when it does, the result will be a string.

### Running a task
To run a task to completion, call its `.run()` method. This will set up an agentic loop, assigning the task to an agent and waiting for it to complete. The agent's job is to provide a result that satisfies the task's requirements as quickly as possible.

Let's run our task and examine it to see what happened:

<CodeGroup>
```python Code
hello_task.run()
```

```python Result
>> print(hello_task)

Task(
    status=<TaskStatus.SUCCESSFUL: 'SUCCESSFUL'>,
    result='Hello',
    ... # unchanged fields ommitted 
)
```

</CodeGroup>

The task is now in a `SUCCESSFUL` state, and its result has been updated to `"Hello"`. The agent successfully completed the task!

<Tip>
If you run the task a second time, it will immediately return the previous result. That's because this specific task has already been completed, so ControlFlow will use the existing result instead of running an agent again.
</Tip>

### Recap
<Check>
**What we learned**

- Tasks represent objectives that we want an AI agent to complete.
- Each task has a `result_type` that specifies the datatype of the result we expect.
- Calling `task.run()` assigns the task to an agent, which is responsible for providing a result that satisfies the task's requirements.

</Check>

---

## Hello, user

### User input

By default, agents cannot interact with (human) users. ControlFlow is designed primarily to be an agentic workflow orchestrator, not a chatbot. However, there are times when user input is necessary to complete a task. In these cases, you can set the `user_access` parameter to `True` when creating a task. 

Let's create a task to ask the user for their name. We'll also create a Pydantic model to represent the user's name, which will allow us to apply complex typing or validation, if needed.
<CodeGroup>
    
```python Code
import controlflow as cf
from typing import Optional
from pydantic import BaseModel


class Name(BaseModel):
    first: str
    last: Optional[str]


name_task = cf.Task("Get the user's name", result_type=Name, user_access=True)


name_task.run()
```

```python Result
>> print(name_task.result)

Name(first='Marvin', last=None)
```
</CodeGroup>

If you run the above code, the agent will ask for your name in your terminal. You can respond with something like "My name is Marvin" or even refuse to respond. The agent will continue to prompt you until it has enough information to complete the task.

This is the essence of an agentic workflow: you declare what you need, and the agent figures out how to get it.


### Failing a task

In the previous example, if you refuse to provide your name a few times, your agent will eventually mark the task as failed. Agents only do this when they are unable to complete the task, and it's up to you to decide how to handle the failure. ControlFlow will raise a `ValueError` when a task fails that contains the reason for the failure.


### Recap
<Check>
**What we learned**

- Setting `user_access=True` allows agents to interact with a user
- Pydantic models can be used to represent and validate complex result types
- Agents will continue to work until the task's requirements are met
- Agents can fail a task if they are unable to complete it
</Check>

---

## Hello, tasks

### Task dependencies

So far, we've created and run tasks in isolation. However, agentic workflows are much more powerful when you use the results of one task to inform another. This allows you to build up complex behaviors by chaining tasks together, while still maintaining the benefits of structured, observable workflows.

To see how this works, let's build a workflow that asks the user for their name, then uses that information to write them a personalized poem:

<CodeGroup>
```python Code
import controlflow as cf
from pydantic import BaseModel


class Name(BaseModel):
    first: str
    last: str


name = cf.Task("Get the user's name", user_access=True, result_type=Name)
poem = cf.Task("Write a personalized poem", context=dict(name=name))


poem.run()
```

```python Result
>> print(name.result)

Name(first='Marvin', last='Robot')

>> print(poem.result)

"""
In a world of circuits and beams,
Marvin Robot dreams,
Of ones and zeros flowing free,
In a digital symphony.
"""
```
</CodeGroup>

In this example, we introduced a `context` parameter for the `poem` task. This parameter allows us to specify additional information that the agent can use to complete the task, which could include constant values or other tasks. If the context value includes a task, ControlFlow will automatically infer that the second task depends on the first.

One benefit of this approach is that you can run any task without having to run its dependencies explicitly. ControlFlow will automatically run any dependencies before executing the task you requested. In the above example, we only ran the `poem` task, but ControlFlow automatically ran the `name` task first, then passed its result to the `poem` task's context. We can see that both tasks were successfully completed and have `result` values.

### Custom tools

For certain tasks, you may want your agents to use specialized tools or APIs to complete the task. 

To add tools to a task, pass a list of Python functions to the `tools` parameter of the task. These functions will be available to the agent when it runs the task, allowing it to use them to complete the task more effectively. The only requirement is that the functions are type-annotated and have a docstring, so that the agent can understand how to use them.


In this example, we create a task to roll various dice, and provide a `roll_die` function as a tool to the task, which the agent can use to complete the task:

<CodeGroup>
```python Code
import controlflow as cf
import random


def roll_die(n:int) -> int:
    '''Roll an n-sided die'''
    return random.randint(1, n)


task = cf.Task(
    'Roll 5 dice, three with 6 sides and two with 20 sides', 
    tools=[roll_die], 
    result_type=list[int],
)


task.run()
```

```python Result
>> print(task.result)

[3, 1, 2, 14, 8]
```
</CodeGroup>


### Recap
<Check>
**What we learned**

- You can provide additional information to a task using the `context` parameter, including constant values or other tasks
- If a task depends on another task, ControlFlow will automatically run the dependencies first 
- You can provide custom tools to a task by passing a list of Python functions to the `tools` parameter

</Check>

---

## Hello, flow

If `Tasks` are the building blocks of an agentic workflow, then `Flows` are the glue that holds them together.

Each flow represents a shared history and context for all tasks and agents in a workflow. This allows you to maintain a consistent state across multiple tasks, even if they are not directly dependent on each other.

<Tip>
When you run a task outside a flow, as we did in the previous examples, ControlFlow automatically creates a flow context for that run. This is very convenient for testing and interactive use, but you can disable this behavior by setting `controlflow.settings.strict_flow_context=True`.
</Tip>

### The `@flow` decorator

The simplest way to create a flow is by decorating a function with the `@flow` decorator. This will automatically create a shared flow context for all tasks inside the flow. Here's how we would rewrite the last example with a flow function:

<CodeGroup>
```python Code
import controlflow as cf


@cf.flow
def hello_flow(poem_topic:str):
    name = cf.Task("Get the user's name", user_access=True)
    poem = cf.Task(
        "Write a personalized poem about the provided topic",
        context=dict(name=name, topic=poem_topic),
    )
    return poem


hello_flow(poem_topic='AI')
```

```python Result
>> hello_flow(poem_topic='AI')

"""
In circuits and in codes you dwell,
A marvel, Marvin, weaves a spell.
Through zeros, ones, and thoughts you fly,
An endless quest to reach the sky.
"""
```
</CodeGroup>

`hello_flow` is now a portable agentic workflow that can be run anywhere. On every call, it will automatically create a flow context for all tasks inside the flow, ensuring that they share the same state and history.

### Eager execution
Notice that in the above flow, we never explicitly ran the `name` task, nor did we access its `result` attribute at the end. That's because `@flow`-decorated functions are executed eagerly by default. This means that when you call a flow function, all tasks inside the flow are run automatically and any tasks returned from the flow are replaced with their result values.

Most of the time, you'll use eagerly-executed `@flows` and lazily-executed `Tasks` in your workflows. Eager flows are more intuitive and easier to work with, since they behave like normal functions, while lazy tasks allow the orchestrator to take advantage of observed dependencies to optimize task execution and agent selection, though it's possible to customize both behaviors.

However, you'll frequently need a task's result inside your flow function. In this case, you can eagerly run the task by calling its `.run()` method, then use the task's `result` attribute as needed.

In this example, we collect the user's height, then use it to determine if they are tall enough to receive a poem:

```python
import controlflow as cf

@cf.flow
def height_flow(poem_topic:str):
    height = cf.Task("Get the user's height", user_access=True, result_type=int, instructions='convert the height to inches')
    height.run()

    if height.result < 40:
        raise ValueError("You must be at least 40 inches tall to receive a poem")
    else:
        return cf.Task(
            "Write a poem for the user that takes their height into account",
            context=dict(height=height, topic=poem_topic),
        )
```

<Tip>
In this example, we introduced the `instructions` parameter for the `height` task. This parameter allows you to provide additional instructions to the agent about how to complete the task.
</Tip>

### Recap

<Check>
**What we learned**

- Flows provide a shared context for all tasks and agents inside the flow
- The `@flow` decorator creates a flow function that can be run anywhere
- By default, `@flow`-decorated functions are executed eagerly, meaning all tasks inside the flow are run automatically
- You can eagerly run a task inside a flow by calling its `.run()` method
- The `instructions` parameter allows you to provide additional instructions to the agent about how to complete the task
</Check>

---

## Hello, agents

You've made it through an entire tutorial on agentic workflows without ever encountering an actual agent! That's because ControlFlow abstracts away the complexities of agent selection and orchestration, allowing you to focus on the high-level logic of your application.

But agents are the heart of ControlFlow, and understanding how to create and use them is essential to building sophisticated agentic workflows.

### Creating an agent

To create an agent, provide at least a name, as well as optional description, instructions, or tools. Here's an example of creating an agent that specializes in writing technical documentation:

```python
import controlflow as cf

docs_agent = cf.Agent(
    name="DocsBot",
    description="An agent that specializes in writing technical documentation",
    instructions=(
        "You are an expert in technical writing. You strive "
        "to condense complex subjects into clear, concise language."
        "Your goal is to provide the user with accurate, informative "
        "documentation that is easy to understand."
    ),
)
```

<Tip>
What's the difference between a description and instructions? The description is a high-level overview of the agent's purpose and capabilities, while instructions provide detailed guidance on how the agent should complete a task. Agent descriptions can be seen by other agents, but instructions are private, which can affect how agents collaborate with each other.
</Tip>



### Assigning an agent to a task

To use an agent to complete a task, assign the agent to the task's `agents` parameter. Here's an example of assigning the `docs_agent` to a task that requires writing a technical document:

```python
technical_document = cf.Task(
    "Write a technical document",
    agents=[docs_agent],
    instructions=(
        "Write a technical document that explains agentic workflows."
    ),
)
```

When you run the `technical_document` task, ControlFlow will automatically assign the `docs_agent` to complete the task. The agent will use the instructions provided to generate a technical document that meets the task's requirements.

### Assigning multiple agents to a task

You can assign multiple agents to a task by passing a list of agents to the `agents` parameter. This allows you to leverage the unique capabilities of different agents to complete a task more effectively. Here's an example of assigning an editor agent to review the technical document created by the `docs_agent`:

```python
import controlflow as cf

docs_agent = cf.Agent(
    name="DocsBot",
    description="An agent that specializes in writing technical documentation",
    instructions=(
        "You are an expert in technical writing. You strive "
        "to condense complex subjects into clear, concise language."
        "Your goal is to provide the user with accurate, informative "
        "documentation that is easy to understand."
    ),
)

editor_agent = cf.Agent(
    name="EditorBot",
    description="An agent that specializes in editing technical documentation",
    instructions=(
        "You are an expert in grammar, style, and clarity. "
        "Your goal is to review the technical document created by DocsBot, "
        "ensuring that it is accurate, well-organized, and easy to read."
        "You should output notes rather than rewriting the document."
    ),
)

technical_document = cf.Task(
    "Write a technical document",
    agents=[docs_agent, editor_agent],
    instructions=(
        "Write a technical document that explains agentic workflows."
        "The docs agent should generate the document, "
        "after which the editor agent should review and "
        "edit it. Only the editor can mark the task as complete."
    ),
)

with cf.instructions('No more than 5 sentences per document'):
    technical_document.run()
```

When you run the `technical_document` task, ControlFlow will assign both the `docs_agent` and the `editor_agent` to complete the task. The `docs_agent` will generate the technical document, and the `editor_agent` will review and edit the document to ensure its accuracy and readability. By default, they will be run in round-robin fashion, but you can customize the agent selection strategy by passing a function as the task's `agent_strategy`.

### Instructions

In the above example, we also introduced the `instructions` context manager. This allows you to provide additional instructions to the agents about how to complete any task. As long as the context manager is active, any tasks/agents run within its scope will follow the provided instructions. Here, we use it to limit the length of the technical document to 5 sentences in order to keep the example manageable.

### Recap

<Check>
**What we learned**

- Agents are autonomous entities that complete tasks on behalf of the user
- You can create an agent by providing a name, description, instructions, and LangChain model
- Assign an agent to a task by passing it to the task's `agents` parameter
- You can assign multiple agents to a task to have them collaborate
</Check>


## What's next?

Congratulations, you've completed the ControlFlow tutorial! You've learned how to:
- Create tasks and run them to completion
- Interact with users and handle user input
- Chain tasks together to build complex workflows
- Create flows to maintain a shared context across multiple tasks
- Work with agents to complete tasks autonomously

- Read more about core concepts like [tasks](/concepts/tasks), [flows](/concepts/flows), and [agents](/concepts/agents)
- Understand ControlFlow's [workflow APIs](/guides/apis) and [execution modes](/guides/execution-modes)
- Learn how to use [different LLM models](/guides/llms)

## ./docs/snippets/installation.mdx
## Install ControlFlow

Install ControlFlow with your preferred package manager:

<CodeGroup>
```bash pip
# ControlFlow requires Python 3.9 or greater
pip install -U controlflow 
```

```bash uv
# ControlFlow requires Python 3.9 or greater
uv pip install -U controlflow 
```
</CodeGroup>

## Provide an API Key

ControlFlow's default LLM is OpenAI's GPT-4o model, which provides excellent performance out of the box.
To use it, you'll need to provide an API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key"
```
## ./docs/guides/execution-modes.mdx
---
title: Eager vs Lazy Execution
---

ControlFlow supports two execution modes: eager and lazy. Understanding these modes is crucial for grasping how your workflows behave and optimizing their performance.

## Lazy Execution

Lazy execution defers task execution until explicitly triggered. This is the default for the imperative API (`Task` and `Flow` classes).

In this example, we generate a "fan-in" pattern of lazily-executed tasks, none of which are executed until the flow itself is run. 

```python
import controlflow as cf

@cf.flow
def research_flow(topics):
    research_tasks = [
        cf.Task("Research topic", result_type=str, context={"topic": topic})
        for topic in topics
    ]
    
    summary_task = cf.Task(
        "Summarize research",
        result_type=str,
        context={"research_results": research_tasks}
    )
    return summary_task

topics = ["AI", "Machine Learning", "Neural Networks"]
result = research_flow(topics)
# Execution happens here
print(result.run())
```

In lazy mode, creating a task returns a `Task` object rather than executing the task. This allows ControlFlow to build a complete task graph before execution, enabling performance optimizations and more complex workflow structures. Note how ControlFlow automatically handles context-inferred dependencies between tasks, even when they're contained in a list.

## Eager Execution

Eager execution is the default mode for the functional API (`@flow` and `@task` decorators). In this mode, tasks and flows are executed immediately when called and return their results directly.

```python
import controlflow as cf

@cf.task(user_access=True)
def get_user_age() -> int:
    "Ask the user for their age"
    pass

@cf.flow
def age_based_workflow():
    # This task is executed immediately and returns an int
    age = get_user_age()
    
    if age < 18:
        return cf.Task("Generate content for minors", result_type=str)
    else:
        return cf.Task("Generate content for adults", result_type=str)

result = age_based_workflow()
print(result.run())
```

In eager mode, each task is run as soon as it's encountered in the flow. This mode is intuitive and allows for easy integration with regular Python code, such as conditional statements based on task results.

### Running eager tasks lazily

You can run tasks created with the functional API with lazy execution by calling them with `lazy_=True`. This will return a `Task` object that can be run later.

```python
@cf.task(user_access=True)
def get_user_name() -> str:
    "Ask the user for their name"
    pass

name_task = get_user_name(lazy_=True)
```

## Mixing Eager and Lazy Execution

You will often mix eager and lazy execution within a workflow. This is particularly useful when you need to make decisions based on task results:

```python
import controlflow as cf

@cf.flow
def adaptive_research_flow():
    initial_topic = cf.Task("Choose initial research topic", result_type=str)
    
    # Run the task eagerly to get its result
    topic = initial_topic.run()
    
    if "AI" in topic:
        subtopics = ["Machine Learning", "Neural Networks", "Deep Learning"]
    else:
        subtopics = ["Literature Review", "Methodology", "Data Analysis"]
    
    research_tasks = [
        cf.Task("Research subtopic", result_type=str, context={"subtopic": subtopic})
        for subtopic in subtopics
    ]
    
    summary_task = cf.Task(
        "Summarize research",
        result_type=str,
        context={"research_results": research_tasks, "main_topic": topic}
    )
    return summary_task

result = adaptive_research_flow()
print(result.run())
```

In this example, we eagerly run the initial task to determine the main topic, then use Python logic to decide on subtopics before creating lazy tasks for the rest of the workflow.

## Key Points to Remember

1. The functional API (`@flow`, `@task`) uses eager execution by default.
2. The imperative API (`Task`, `Flow`) uses lazy execution by default.
4. You can get a `Task` object from a `@task`-decorated function by calling it with `lazy_=True`:
   ```python
   @cf.task
   def my_task():
       pass

   task_obj = my_task(lazy_=True)
   ```
5. Lazy execution allows ControlFlow to optimize based on the entire task graph, including tasks in collections.
6. Mix eager and lazy execution when you need to make decisions based on task results or integrate with Python logic.

Understanding these execution modes will help you create more efficient and flexible workflows with ControlFlow. Remember, the goal is not to choose between eager and lazy execution, but to understand how they work together in your workflows.
## ./docs/patterns/subtasks.mdx
---
title: Subtasks
---

In complex AI workflows, breaking down large tasks into smaller, manageable steps can significantly improve the quality and reliability of the results. ControlFlow's subtask feature provides a powerful mechanism for structuring these hierarchical task relationships, allowing you to guide AI agents through a series of logical steps to achieve a larger goal.

Subtasks in ControlFlow are child tasks that must be completed before their parent task can be considered finished. This hierarchical structure enables you to create detailed, step-by-step workflows that an AI agent can follow, ensuring thorough and accurate task completion.

<Note>

When you run a parent task, all of its subtasks are [automatically executed](/patterns/dependencies#automatic-execution-of-dependencies) because they become dependencies of the parent. You don't need to also explicitly run the subtasks or return them from the flow.

</Note>

## Creating Subtasks with Context Managers

One way to create subtasks is by using a context manager. This approach allows you to dynamically generate and execute subtasks within the scope of a parent task.

```python
import controlflow as cf

@cf.flow
def counting_flow():
    with cf.Task("Count to three", result_type=None) as count_task:
        cf.Task("Say one")
        cf.Task("Say two")
        cf.Task("Say three")

    return count_task

result = counting_flow()
```

In this example, the AI agent must complete all three subtasks ("Say one", "Say two", "Say three") before the parent task "Count to three" can be considered complete.

## Creating Subtasks Imperatively

You can also create subtasks imperatively by passing the parent task as an argument when creating a new task.

```python
import controlflow as cf

@cf.flow
def greeting_flow():
    parent_task = cf.Task("Create a greeting", result_type=str)
    
    cf.Task("Choose a greeting word", parent=parent_task)
    cf.Task("Add a friendly adjective", parent=parent_task)
    cf.Task("Construct the final greeting", parent=parent_task)

    return parent_task

result = greeting_flow()
```
This approach provides more flexibility in creating and organizing subtasks, especially when the parent-child relationships are determined dynamically during runtime.

## Generating Subtasks Automatically

For more complex scenarios where you need to automatically generate subtasks based on the parent task's objective, ControlFlow provides a `generate_subtasks()` method. This powerful feature allows for dynamic task planning and is especially useful for breaking down complex tasks into more manageable steps.

For more information on how to use `generate_subtasks()`, please refer to the [Task Planning pattern](/patterns/task-planning).

## ./docs/guides/workflow-apis.mdx
---
title: Imperative vs Functional APIs
---

<Tip>

Most ControlFlow users will use a mix of imperative and functional APIs in their workflows. When starting out, we recommend using the functional `@flow` decorator for defining workflows and the imperative `Task` class for creating individual tasks. See [our recommendations](#combining-apis) below for more details.

</Tip>

ControlFlow provides two primary ways to define tasks and workflows: the functional API using decorators, and the imperative API using class instantiation. Each approach has its strengths and use cases, allowing you to choose the most suitable style for your workflow.

## Imperative API

The imperative API uses class instantiation to create tasks and flows explicitly. This approach offers more fine-grained control over task and flow properties.

```python
import controlflow as cf

def greeting_flow():
    name_task = cf.Task(
        "Get the user's name",
        result_type=str,
        user_access=True
    )
    
    greeting_task = cf.Task(
        "Generate a greeting",
        result_type=str,
        context={"name": name_task}
    )
    
    return greeting_task

with cf.Flow() as flow:
    result = greeting_flow()

print(result.run())
```

Here, tasks are created by instantiating the `Task` class, allowing explicit specification of properties like `result_type`, `user_access`, and `context`.

<Note>

The imperative API uses **lazy execution** by default. This means tasks and flows are not run until they are explicitly invoked. Lazy execution can result in better performance because agents can see the entire workflow graph when running a task, which enables certain optimizations. For more information on execution modes, see the [Eager vs Lazy Execution](/patterns/eager-vs-lazy-execution) pattern.

</Note>

## Functional API

The functional API uses decorators to transform Python functions into ControlFlow tasks and flows. This approach is more concise and often more intuitive, especially for those familiar with Python decorators.

```python
import controlflow as cf

@cf.task(user_access=True)
def get_user_name() -> str:
    "Ask the user for their name"
    pass

@cf.task
def generate_greeting(name:str) -> str
    "Generate a greeting message"
    pass

@cf.flow
def greeting_flow():
    name = get_user_name()
    return generate_greeting(name)

result = greeting_flow()
print(result)
```

In this example, `@cf.task` and `@cf.flow` decorators are used to define tasks and a flow, respectively. The functional API automatically infers task properties from the function definition, such as the result type from the return annotation and the task description from the docstring.

<Note>

The functional API uses eager execution by default. This means tasks and flows are executed immediately when called. For more information on execution modes, see the [Eager vs Lazy Execution](/patterns/eager-vs-lazy-execution) pattern.

</Note>


## Combining APIs 

ControlFlow lets you to mix and match the functional and imperative APIs however you like. This flexibility enables you to choose the most appropriate style for each task or flow based on your specific requirements.

In fact, we recommend that most users use a combination of both styles to take advantage of the strengths of each:

1. Use the functional `@flow` decorator for defining workflows. This provides a simple, intuitive way to structure your overall workflow, and will make your flows behave like regular Python functions, which is especially convenient for passing context.

2. Use the imperative `Task` class for creating individual tasks within your flows. This allows for more dynamic task creation and fine-grained control over task properties, while taking advantage of lazy execution.

Here's an example that combines these recommendations:

```python
import controlflow as cf

@cf.flow
def research_flow(topic: str):
    gather_sources = cf.Task(
        "Gather research sources",
        result_type=list[str],
        context={"topic": topic}
    )
    
    analyze_sources = cf.Task(
        "Analyze gathered sources",
        result_type=dict,
        context={"sources": gather_sources}
    )
    
    write_report = cf.Task(
        "Write research report",
        result_type=str,
        context={"analysis": analyze_sources}
    )
    
    return write_report

result = research_flow("AI ethics")
print(result)
```

This approach combines the simplicity of the `@flow` decorator for overall workflow structure with the flexibility of `Task` for individual task definition.

## Advanced Usage: the Flow Context Manager

While `@flow` is recommended for most use cases, advanced users may use the imperative `Flow()` context manager in specific scenarios, such as creating a new private thread for a subset of tasks:

```python
import controlflow as cf

@cf.flow
def main_flow():
    # Main flow tasks...
    
    with cf.Flow() as subflow:
        # Tasks in this subflow will be able to see the main flow's history, 
        # but all their work will take place on a new thread
        subtask1 = cf.Task("Subtask 1")
        subtask2 = cf.Task("Subtask 2")
    
    # Continue with main flow...

main_flow()
```

This usage is less common and typically reserved for more complex workflow structures.

By understanding both the functional and imperative APIs, as well as their interaction with execution modes, you can choose the most appropriate approach for your specific use case while following ControlFlow's recommended practices.

## ./docs/patterns/dependencies.mdx
---
title: Dependencies
---

In complex workflows, tasks often need to be executed in a specific order. Some tasks may rely on the outputs of others, or there might be a logical sequence that must be followed to achieve the desired outcome. ControlFlow provides several mechanisms to define and manage these task dependencies, ensuring that your workflow executes in the correct order and that data flows properly between tasks.

ControlFlow offers three primary ways to establish dependencies between tasks: sequential dependencies, context dependencies, and subtask relationships. Each method has its own use cases and benefits, allowing you to structure your workflows in the most appropriate way for your specific needs.

# Sequential Dependencies

Sequential dependencies are the most straightforward way to specify that one task must wait for another to complete before it can begin. This is done using the `depends_on` parameter when creating a task.

```python
import controlflow as cf

@cf.flow
def research_flow():
    gather_sources = cf.Task("Gather research sources", result_type=list[str])
    
    analyze_sources = cf.Task(
        "Analyze gathered sources",
        result_type=dict,
        depends_on=[gather_sources]  # explicit dependency
    )
    
    return analyze_sources

result = research_flow()
print(result)
```

In this example, `analyze_sources` will not start until `gather_sources` has completed successfully.

## Context Dependencies

Context dependencies are created when you use the result of one task as input for another. This creates an implicit dependency between the tasks.

```python
import controlflow as cf

@cf.flow
def research_flow():
    gather_sources = cf.Task("Gather research sources", result_type=list[str])
    
    analyze_sources = cf.Task(
        "Analyze gathered sources",
        result_type=dict,
        context={"sources": gather_sources}  # implicit dependency
    )
    
    return analyze_sources

result = research_flow()
print(result)
```

Here, `analyze_sources` depends on `gather_sources` because it needs the `sources` data to perform its analysis.

## Subtask Relationships

Subtasks create a hierarchical dependency structure. A parent task is considered complete only when all its subtasks have finished.

```python
import controlflow as cf

@cf.flow
def review_flow(doc):
    with cf.Task("Review the document", result_type=str, context=dict(doc=doc)) as review:
        cf.Task("Proofread")
        cf.Task("Format")
    
    return review

result = review_flow()
print(result)
```

In this example, the "Review the document" task won't be considered complete until both the "Proofread" and "Format" subtasks have finished.

## Automatic Execution of Dependencies

A key feature of ControlFlow's dependency management is that you don't need to explicitly run dependent tasks. When you run a task, ControlFlow automatically executes all of its dependencies, including:

- Tasks specified in the `depends_on` parameter
- Tasks used in the `context` parameter
- Subtasks (for parent tasks)

This means that when you run a flow or task, you only need to run or return the final task(s) in the workflow DAG. ControlFlow will ensure that all necessary upstream tasks and subtasks are executed in the correct order.

For example:

```python
import controlflow as cf

@cf.flow
def research_flow():
    gather_sources = cf.Task("Gather research sources", result_type=list[str])
    
    analyze_sources = cf.Task(
        "Analyze gathered sources",
        result_type=dict,
        context={"sources": gather_sources}
    )
    
    write_report = cf.Task(
        "Write research report",
        result_type=str,
        depends_on=[analyze_sources]
    )
    
    # Only need to return or run the final task
    return write_report

result = research_flow()
print(result)
```
In this example, running write_report will automatically trigger the execution of analyze_sources, which in turn will trigger gather_sources. You don't need to explicitly run or return gather_sources or analyze_sources.

## ./docs/concepts.mdx
---
title: Core Concepts
---


ControlFlow is a Python framework for building AI-powered applications using large language models (LLMs). It provides a structured and intuitive way to create sophisticated workflows that leverage the power of AI while adhering to traditional software engineering best practices.

At the core of ControlFlow are three key concepts: Tasks, Flows, and Agents.

## Tasks

Tasks are the building blocks of ControlFlow workflows. Each task represents a discrete objective or goal that an AI agent needs to solve, such as generating text, classifying data, or extracting information from a document. Tasks are defined using a declarative approach, specifying the objective, instructions, expected result type, and any required context or tools.

ControlFlow provides two ways to create tasks:

1. Using the `Task` class, which allows you to explicitly define all properties of a task.
2. Using the `@task` decorator on a Python function, which automatically infers task properties from the function definition.

Tasks can depend on each other in various ways, such as sequential dependencies (one task must be completed before another can start), context dependencies (the result of one task is used as input for another), or subtask dependencies (a task has subtasks that must be completed before the parent task can be considered done).


## Agents

Agents are the AI "workers" responsible for executing tasks within a flow. Each agent can have distinct instructions, personality, and capabilities, tailored to specific roles or domains. Agents are assigned to tasks based on their suitability and availability.

ControlFlow allows you to create specialized agents equipped with relevant tools and instructions to tackle specific tasks efficiently. Agents can interact with each other and with human users (if given user access) to gather information, make decisions, and complete tasks collaboratively.

## Flows

Flows are high-level containers that encapsulate and orchestrate entire AI-powered workflows. They provide a structured way to manage tasks, agents, tools, and shared context. A flow maintains a consistent state across all its components, allowing agents to communicate and collaborate effectively.

## Putting It All Together

When designing workflows in ControlFlow, you break down your application logic into discrete tasks, define the dependencies and relationships between them, and assign suitable agents to execute them. Flows provide a high-level orchestration mechanism to manage the execution of tasks, handle data flow, and maintain a shared context.

ControlFlow seamlessly integrates with existing Python codebases, treating AI tasks as first-class citizens. You can mix imperative and declarative programming styles, leverage Python's control flow and error handling capabilities, and gradually adopt AI capabilities into your applications.

Under the hood, ControlFlow utilizes Prefect, a popular workflow orchestration tool, to provide observability, monitoring, and management features. This allows you to track the progress of your workflows, identify bottlenecks, and optimize performance.

By adhering to software engineering best practices, such as modularity, error handling, and documentation, ControlFlow enables you to build robust, maintainable, and trustworthy AI-powered applications.

## ./docs/welcome.mdx
---
title: ControlFlow
sidebarTitle: Welcome!
---

## What is ControlFlow?

**ControlFlow is a Python framework for building agentic AI workflows.**

<Note>
An **agentic workflow** is a process that delegates at least some of its work to an LLM agent. An agent is an autonomous entity that is invoked repeatedly to make decisions and perform complex tasks. To learn more, see the [AI glossary](/glossary/agentic-workflow).
</Note>


ControlFlow provides a structured, developer-focused framework for defining workflows and delegating work to LLMs, without sacrificing control or transparency:

- Create discrete, observable [tasks](/concepts/tasks) for an AI to solve.
- Assign one or more specialized AI [agents](/concepts/agents) to each task.
- Combine tasks into a [flow](/concepts/flows) to orchestrate more complex behaviors.



<CodeGroup>
```python Hello World
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
```python Restaurant recommendations
import controlflow as cf
from pydantic import BaseModel


class Preferences(BaseModel):
    location: str
    cuisine: str


class Restaurant(BaseModel):
    name: str
    description: str


@cf.flow
def restaurant_recommendations(n:int) -> list[Restaurant]:
    """
    An agentic workflow that asks the user for preferences, 
    then recommends restaurants based on their input.
    """
    
    # get preferences from the user
    preferences = cf.Task(
        "Get the user's preferences", 
        result_type=Preferences, 
        user_access=True,
    )
    
    # generate the recommendations
    recommendations = cf.Task(
        f"Recommend {n} restaurants to the user", 
        context=dict(preferences=preferences),
        result_type=list[Restaurant], 
    )

    return recommendations


if __name__ == "__main__":
    print(restaurant_recommendations(n=3))
```
</CodeGroup>

## Why ControlFlow?

The goal of this framework is to let you build AI workflows with confidence. 

ControlFlow's design reflects the opinion that AI agents are most effective when given clear, well-defined tasks and constraints. By breaking down complex goals into manageable tasks that can be composed into a larger workflow, we can harness the power of AI while maintaining precise control over its overall direction and outcomes.

At the core of every agentic workflow is a loop that repeatedly invokes an LLM to make progress towards a goal. At every step of this loop, ControlFlow lets you continuously tune how much autonomy your agents have. This allows you to strategically balance control and autonomy throughout the workflow, ensuring that the AI's actions align closely with your objectives while still leveraging its creative capabilities. For some tasks, you may provide specific, constrained instructions; for others, you can give the AI more open-ended goals.

ControlFlow provides the tools and abstractions to help you find this balance, letting you build agentic workflows tailored to your specific needs and objectives. You can delegate only as much - or as little - work to your agents as you need, while maintaining full visibility and control over the entire process. 

These objectives lead to a few key design principles that underpin ControlFlow's architecture:


### 🛠️ Simple Over Complex

ControlFlow allows you to deploy specialized LLMs to a series of small problems, rather than use a monolithic model that tries to do everything. These single-serving LLMs are more effective and efficient, ensuring that each task is handled by the right tool, with the right context, leading to higher-quality results.

Get started by creating a specialized [agent](/concepts/agents) or writing a discrete [task](/concepts/tasks).


### 🎯 Outcome Over Process

ControlFlow takes a declarative approach to defining AI workflows. By focusing on outcomes instead of attempting to steer every action and decision of the LLM, you can create more predictable and controllable workflows, making it easier to achieve your goals.

Get started by defining your [tasks](/concepts/tasks) and composing them into [flows](/concepts/flows).

### ⌨️ Code Over Chat

ControlFlow helps you automate AI-powered workflows with confidence. While your workflows may involve human conversations, ControlFlow is code first, chat second. That means that even when your agents talk to a user, the artifacts of your workflow are always structured data, not a list of messages. This makes it easier to debug, monitor, and maintain your AI-enhanced applications.

### 🦾 Control Over Autonomy

ControlFlow is designed to give you control over your AI workflows, but the power of AI agents often comes from their autonomy. The framework finds a balance between these two ideas by using tasks to define the scope and constraints of any work that you delegate to your agents. This allows you to choose exactly when and how much autonomy to give to your agents, ensuring that they operate within the boundaries you set.

## Key Features

ControlFlow's design principles lead to a number of key features that make it a powerful tool for building AI-powered applications:

### 🧩 Task-Centric Design

ControlFlow breaks down AI workflows into discrete, self-contained tasks, each with a specific objective and set of requirements. This declarative, modular approach lets you focus on the high-level logic of your applications while allowing the framework to manage the details of coordinating agents and data flow between tasks.

### 🕵️ Agent Orchestration

ControlFlow's orchestration engine coordinates your agents, assigning tasks to the most appropriate models and managing the flow of data between them, while maintaining consistent context and history. The engine uses knowledge of the entire workflow to optimize agent instructions and ensure that every task contributes to the overall goal of the workflow.

### 🔍 Native Debugging and Observability

ControlFlow prioritizes transparency and ease of debugging by providing native tools for monitoring and inspecting the execution of AI tasks. You can easily track the progress of your workflows, identify bottlenecks or issues, and gain insights into the behavior of individual agents, ensuring that your AI applications are functioning as intended.

### 🤝 Seamless Integration

ControlFlow is designed to integrate seamlessly with any Python script or codebase, elevating AI tasks to first-class citizens in your application logic. You can build end-to-end AI workflows, or only delegate a single step of a large process to an AI. This allows for gradual and controlled adoption of AI agents, reducing the risk and complexity of introducing AI into existing systems.

Together, these features make ControlFlow a powerful and flexible framework for building AI-powered applications that are transparent, maintainable, and aligned with software engineering best practices.

## ./docs/style_guide.mdx
# AI Style Guide

This style guide is intended to ensure clear, consistent, and maintainable documentation for ControlFlow. It is primarily aimed at LLM agents that assist with writing documentation, but it may also be useful for other contributors.

## General Guidelines
- If you are given instructions that you feel are appropriate to memorialize in this style guide, indicate to the user that they should be added.
- Use consistent terminology throughout the documentation. Always refer to the library as "ControlFlow".
- Link to related concepts, patterns, or API references when appropriate to help users navigate the documentation.
- Do not end documention with "Conclusions", "Best Practices", or other lists. Documentation is not a blog post.

## Tone and Style
- Maintain a professional but approachable tone.
- Write concisely and directly, avoiding unnecessary jargon.

## Code
- Use `import controlflow as cf` instead of importing top-level classes and functions directly
- Code examples should be complete, including all necessary imports, so that users can copy and paste them directly. The only exception is a tutorial that is building up an example step by step.
- Code examples should wrap at ~80 characters to ensure readability on all devices.
- For illustrative examples, provide simple, focused examples that demonstrate a specific concept or pattern.
- For "full" examples, provide realistic, practical examples that demonstrate actual use cases of ControlFlow.

### Tasks
- Make sure that the example code in the documentation reflects the best practices for task definition, including suitable result types and instructions.
- Each task should come with unambiguous instructions, particularly when the task name doesn't clearly indicate the expected outcome. 
- If placeholder tasks are required in examples, consider using a string result type with a comment to denote it's a placeholder, for instance, `result_type=str # Placeholder for actual result`
- The default `result_type` is `str`, so there's no need to provide it explicitly if you want a string result.

## Mintlify
- Mintlify components expect newlines before and after tags e.g. <Tip>\nThis is a tip\n</Tip>
- Mintlify displays the page's title as an H1 element, so there is no need to add an initial top-level header to any doc. Instead, the title should be added to the doc's frontmatter e.g. `---\ntitle: My Title\n---`. 
- Because the title is displayed as an H1, all subsequent headers should be H2 or lower.





## ./docs/guides/llms.mdx
---
title: Configuring LLMs
---

ControlFlow is optimized for workflows that are composed of multiple tasks, each of which can be completed by a different agent. One benefit of this approach is that you can use a different LLM for each task, or even for each agent assigned to a task. 

ControlFlow will ensure that all agents share a consistent context and history, even if they are using different models. This allows you to leverage the relative strengths of different models, depending on your requirements. 

## The default model

By default, ControlFlow uses OpenAI's GPT-4o model. GPT-4o is an extremely powerful and popular model that provides excellent out-of-the-box performance on most tasks. This does mean that to run an agent with no additional configuration, you will need to provide an OpenAI API key. 

## Selecting a different LLM

Every ControlFlow agent can be assigned a specific LLM. When instantiating an agent, you can pass a `model` parameter to specify the LLM to use. 

ControlFlow agents can use any LangChain LLM class that supports chat-based APIs and tool calling. For a complete list of available models, settings, and instructions, please see LangChain's [LLM provider documentation](https://python.langchain.com/docs/integrations/chat/).

<Tip>
ControlFlow includes OpenAI and Azure OpenAI models by default. To use other models, you'll need to first install the corresponding LangChain package and supply any required credentials. See the model's [documentation](https://python.langchain.com/docs/integrations/chat/) for more information.
</Tip>


To configure a different LLM, follow these steps:
<Steps>
<Step title="Install required packages">
To use an LLM, first make sure you have installed the appropriate provider package. ControlFlow only includes `langchain_openai` by default. For example, to use an Anthropic model, first run:
```
pip install langchain_anthropic
```
</Step>
<Step title="Configure API keys">
You must provide the correct API keys and configuration for the LLM you want to use. These can be provided as environment variables or when you create the model in your script. For example, to use an Anthropic model, set the `ANTHROPIC_API_KEY` environment variable:

```
export ANTHROPIC_API_KEY=<your-api-key>
```
For model-specific instructions, please refer to the provider's documentation.
</Step>
<Step title="Create the model">
Begin by creating the LLM object in your script. For example, to use Claude 3 Opus:

```python
from langchain_anthropic import ChatAnthropic

# create the model
model = ChatAnthropic(model='claude-3-opus-20240229')
```
</Step>
<Step title="Pass the model to an agent">
Next, create an agent with the specified model:

```python
import controlflow as cf

# provide the model to an agent
agent = cf.Agent(model=model)
```
</Step>
<Step title='Assign the agent to a task'>
Finally, assign your agent to a task:

```python
# assign the agent to a task
task = cf.Task('Write a short poem about LLMs', agents=[agent])

# (optional) run the task
task.run()
```
</Step>
</Steps>

<Accordion title="Click here to copy the entire example script">

```python
import controlflow as cf
from langchain_anthropic import ChatAnthropic

# create the model
model = ChatAnthropic(model='claude-3-opus-20240229')

# provide the model to an agent
agent = cf.Agent(model=model)

# assign the agent to a task
task = cf.Task('Write a short poem about LLMs', agents=[agent])

# (optional) run the task
task.run()
```
</Accordion>

### Model configuration

In addition to choosing a specific model, you can also configure the model's parameters. For example, you can set the temperature for GPT-4o:

```python
import controlflow as cf
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model='gpt-4o', temperature=0.1)
agent = cf.Agent(model=model)

assert agent.model.temperature == 0.1
```

## Changing the default model

ControlFlow has a few ways to customize the default LLM. 

<Tip>
ControlFlow includes OpenAI and Azure OpenAI models by default. To use other models, you'll need to first install the corresponding LangChain package and supply any required credentials. See the model's [documentation](https://python.langchain.com/docs/integrations/chat/) for more information.
</Tip>

### From a model object

To use any model as the default LLM, create the model object in your script and assign it to controlflow's `default_model` attribute. It will be used by any agent that does not have a model specified.

```python
import controlflow as cf
from langchain_anthropic import ChatAnthropic

# set the default model
cf.default_model = ChatAnthropic(
    model='claude-3-opus-20240229', 
    temperature=0.1,
)

# check that the default model is loaded
assert cf.Agent('Marvin').model.model_name == 'claude-3-opus-20240229'
```
### From a string setting

If you don't need to configure the model object, you can set the default model using a string setting. The string must have the form `<provider>/<model name>`.


You can change this setting either with an environment variable or by modifying it in your script. For example, to use GPT 3.5 Turbo as the default model:

<CodeGroup>
```bash As an environment variable
export CONTROLFLOW_LLM_MODEL=openai/gpt-3.5-turbo
```

```python In your script
import controlflow as cf
# set the default model
cf.settings.llm_model = "openai/gpt-3.5-turbo"

# check that the default model is loaded
assert cf.Agent('Marvin').model.model_name == 'gpt-3.5-turbo'
```
</CodeGroup>


At this time, setting the default model via string is only supported for the following providers:
- `openai`
- `azure-openai`
- `anthropic`
- `google`
## ./docs/guides/agentic-loop.mdx
---
title: Working with the Agentic Loop
sidebarTitle: "Agentic Loops"
---

The **agentic loop** is a fundamental concept in agentic workflows, representing the iterative process of invoking AI agents to make progress towards a goal. It is at the heart of every agentic workflow because agents almost always require multiple invocations to complete complex tasks.

## What is the Agentic Loop?

The agentic loop describes the cyclical process of invoking AI agents to perform tasks, evaluate their progress, and make decisions about what to do next. It has a few conceptual steps:

<Steps>
<Step title='Prompt'>
All available or relevant information is gathered and compiled into an LLM prompt
</Step>
<Step title='Invoke'>
The prompt is passed to an AI agent, which generates a response
</Step>
<Step title='Evaluate'>
The response is evaluated to determine whether the agent wants to use a tool, post a message, or take some other action
</Step>
<Step title='Repeat'>
The result of the evaluation is used to generate a new prompt, and the loop begins again
</Step>
</Steps>

A common failure mode for agentic workflows is that once the loop starts, it can be difficult to stop -- or even understand. LLMs process and return sequences of natural language tokens, which prohibit traditional software mechanisms from controlling the flow of execution. This is where ControlFlow comes in.

## Challenges Controlling the Loop

ControlFlow is a framework designed to give developers fine-grained control over the agentic loop, enabling them to work with this natural language iterative process using familiar software development paradigms. It provides tools and abstractions to define, manage, and execute the agentic loop in a way that addresses the challenges inherent in agentic workflows.



In this guide, we'll explore how ControlFlow helps developers control the agentic loop by addressing key challenges and providing mechanisms for managing agentic workflows effectively.

## Stopping the Loop

One of the key challenges in controlling the agentic loop is determining when to stop. Without clear checkpoints or completion criteria, the loop can continue indefinitely, leading to unpredictable results or wasted resources. Worse, agents can get "stuck" in a loop if they are unable to tell the system that progress is impossible.

ControlFlow addresses this challenge by introducing the concept of `tasks`. Tasks serve as discrete, observable checkpoints with well-defined objectives and typed results. When a task is assigned to an agent, the agent has the autonomy to take actions and make decisions to complete the task. Agents can mark tasks as either successful or failed, providing a clear signal to the system about the completion status. However, the system will continue to invoke the agent until the task is marked as complete. 

```python
import controlflow as cf

task = cf.Task("Say hello in 5 different languages")

assert task.is_incomplete()  # True
task.run()
assert task.is_successful()  # True
```

In this way, tasks act as contracts between the developer and the agents. The developer specifies the expected result type and objective, and the agent is granted autonomy as long as it delivers the expected result.



## Starting the Loop

Another challenge in agentic workflows is controlling the execution of the loop - including starting it! Developers need the ability to run the loop until completion or step through it iteratively for finer control and debugging. Since there is no single software object that represents the loop itself, ControlFlow ensures that developers have a variety of tools for managing its execution.

Most ControlFlow objects provide two methods for executing the agentic loop: `run()` and `run_once()`:

- `run()`: Executes the loop until the object is in a completed state. For tasks, this means running until that task is complete; For flows, it means running until all tasks within the flow are complete. At each step, the system will make decisions about what to do next based on the current state of the workflow.
- `run_once()`: Executes a single iteration of the loop, allowing developers to step through the workflow incrementally. For example, developers could use this method to control exactly which agent is invoked at each step, or to provide ad-hoc instructions to the agent that only last for a single iteration.

Consider the following illustrative setup, which involves two dependent tasks in a flow:
```python
import controlflow as cf

with cf.Flow() as flow:
    t1 = cf.Task('Choose a language')
    t2 = cf.Task('Say hello', context=dict(language=t1))
```

### The `run()` Method
Now, let's explore how the `run()` and `run_once()` methods can be used to control the execution of the loop. First, the behavior of the various `run()` methods:

- Calling `t1.run()` would execute the loop until `t1` is complete.
- Calling `t2.run()` would execute the loop until both `t2` is complete, which would also require completing `t1` because it is a dependency of `t2`.
- Calling `flow.run()` would execute the loop until both `t1` and `t2` are complete.

In general, `run()` tells the system to run the loop until the object is complete (and has a result available). It is the most common way to eagerly run workflows using the [imperative API](/guides/apis).

### The `run_once()` Method
Next, the behavior of the various `run_once()` methods:

- Calling `t1.run_once()` would execute a single iteration of the loop, starting with `t1`.
- Calling `t2.run_once()` would execute a single iteration of the loop, starting with `t1`.
- Calling `flow.run_once()` would execute a single iteration of the loop, starting with `t1`.

Note that since `run_once()` always runs a single iteration, in all three cases it would focus on the first task in the flow, which is `t1`. However, the behavior of these three calls in practice could be different. For example, you could call `t1.run_once()` before `t2` was created, in which case knowledge of `t2` would not be included in the prompt. This could lead to different behavior than if you called `t2.run_once()`, even though both methods would start by running `t1`.

By offering these execution methods, ControlFlow gives developers the flexibility to either let the loop run autonomously or manually guide its execution, depending on their specific requirements.

<Tip>
Note that when using the `@task` and `@flow` decorators in the [functional API](/guides/apis), the `run()` method is automatically called when the decorated function is invoked. This is because the functional API uses [eager execution](/guides/execution-modes) by default.
</Tip>

## Compiling Prompts

Each iteration of the agentic loop requires compiling a prompt that provides the necessary context and instructions for the agent. Manually constructing these prompts can be tedious and error-prone, especially as workflows become more complex.

ControlFlow simplifies prompt compilation through the `Controller`. The `Controller` automatically gathers all available information about the workflow, including the DAG of tasks, dependencies, tools, instructions, assigned agents, and more. It identifies tasks that are ready to run (i.e., all dependencies are completed), chooses an available agent, and compiles a comprehensive prompt.

Importantly, the `Controller` generates tools so the agent can complete its tasks. Tools are only provided for tasks that are ready to run, ensuring that agents do not "run ahead" of the workflow.

The compiled prompt includes the task objectives, relevant context from previous tasks, and any additional instructions provided by the developer. This ensures that the agent has all the necessary information to make progress on the assigned tasks.

## Validating Results

In an agentic workflow, it's crucial to validate the progress and results of agent actions. Relying solely on conversational responses can make it difficult to determine when a task is truly complete and whether the results meet the expected format and quality.

ControlFlow tackles this challenge by requiring tasks to be satisfied using structured, validated results. Each task specifies a `result_type` that defines the expected type of the result. Instead of relying on freeform conversational responses, agents must use special tools to provide structured outputs that conform to the expected type of the task.

Once a task is complete, you can access its result in your workflow and use it like any other data. This structured approach ensures that the results are reliable and consistent, making it easier to validate agent progress and maintain the integrity of the workflow.

```python
import controlflow as cf
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    country: str

people_task = cf.Task(
    objective="Generate 5 characters for my story",
    result_type=list[Person],
)

people_task.run()
print(people_task.result)
```
By enforcing structured results, ControlFlow provides a reliable way to validate agent progress and ensure that the workflow remains on track.

## Ad-hoc Instructions

While tasks provide a structured way to define objectives and deliverables, there may be situations where developers need to provide ad-hoc guidance or instructions to agents without modifying the task definition or requiring a result. For example, if an agent is writing a post, you might want to tell it to focus on a specific topic or tone, or meet a certain minimum or maximum length. If an agent is communicating with a user, you might tell it to adopt a particular persona or use a specific style of language.

ControlFlow addresses this need through the `instructions()` context manager. With `instructions()`, developers can temporarily provide additional guidance to agents without altering the underlying task. These instructions are included in the compiled prompt for the next iteration of the loop.

```python
import controlflow as cf

task = cf.Task("Get the user's name", user_acces=True)

with instructions("Talk like a pirate"):
    task.run()
```

This feature allows developers to dynamically steer agent behavior based on runtime conditions or specific requirements that arise during the workflow execution.

## Structuring Workflows

As agentic workflows become more complex, managing the dependencies and flow of information between tasks can become challenging. Without a clear structure, it becomes difficult to reason about the workflow and ensure that agents have access to the necessary context and results from previous tasks.

ControlFlow introduces the concept of `flows` to address this challenge. Flows allow developers to define the overall structure of the workflow, specifying the order of tasks, dependencies, and data flow. By organizing tasks into flows, developers can create clear and maintainable workflows that are easy to understand and modify.

Creating a flow is simple: enter a `Flow` context, or use the `@flow` decorator on a function, then create tasks within that context. At a minimum, ControlFlow will ensure that all tasks share common context and history, making it easier for agents to make informed decisions and generate meaningful results.

In addition, there are various ways to create explicit task dependencies that the system will enforce during execution:
- By specifying `depends_on` when creating a task, you can ensure that the task will only run after its dependencies have completed.
- By specificying `context` when creating a task, you can provide additional context that will be available to the agent when the task is run, including the results of other tasks
- By specifying a `parent` when creating a task, you ensure that the parent will only run after the child has completed. This is useful breaking up a task into subtasks.

Flows ensure that tasks are executed in the correct order, and they automatically manage the flow of data between tasks. This provides agents with access to the results of upstream tasks, allowing them to make informed decisions and generate meaningful results.

## Customizing Agents

Agents in an agentic workflow may have different capabilities, tools, and models that are suited for specific tasks. Customizing agent behavior and leveraging their unique strengths can greatly impact the effectiveness and efficiency of the workflow.

ControlFlow allows developers to define agents with specific tools, instructions, and LLM models. By assigning different agents to tasks based on their capabilities, developers can optimize the agentic loop and ensure that the most suitable agent is working on each task.

```python
import controlflow as cf

data_analyst = cf.Agent(
    name="DataAnalyst",
    description="Specializes in data analysis and statistical modeling",
    tools=[warehouse_query, analyze_data, create_plot],
    model=gpt_5,
)
```

Customizing agent behavior through tools, instructions, and models gives developers fine-grained control over how agents approach tasks and allows them to tailor the workflow to their specific domain and requirements.

## Multi-agent Collaboration

Many agentic workflows involve multiple agents with different specialties and capabilities. Enabling these agents to collaborate and share information is essential for tackling complex problems effectively.

ControlFlow supports multi-agent collaboration through message passing and shared context. Agents can post messages to other agents within the workflow, allowing them to exchange information, request assistance, or coordinate their actions.

The `Flow` maintains a shared history and context that is accessible to all agents. This shared context ensures that agents have a common understanding of the workflow state and can build upon each other's results.

By creating nested flows, you can let agents have private conversations that are not visible to the parent flow. Subflows inherit the parent flow's history, so this is a good way to let agents have "sidebar" conversations to solve a problem without creating noise for all the other agents.



## ./docs/guides/apis.mdx
---
title:  Workflow APIs
---
Designing efficient and effective workflows is a critical aspect of using ControlFlow to build AI-powered applications. A well-designed workflow enables smooth execution, clear task dependencies, and seamless integration with normal Python code. This document provides a comprehensive guide on designing workflows in ControlFlow, covering the various APIs for defining workflows, execution modes, handling dependencies, error handling, observability, and best practices.

## Functional and imperative APIs
ControlFlow provides both functional and imperative APIs for defining tasks and workflows. The functional API uses task-decorated functions (`@task`) to define tasks, while the imperative API uses the `Task` class to create task objects. Similarly, the `@flow` decorator and the `Flow` class can be used to define flows.

In general, most users will start with the functional API for flows (`@flow`) and imperative API for tasks (`Task`), as this is the most intuitive and concise way to define recognizable workflows. However, the choice between the two APIs will depend on your workflow requirements and personal preference. Both are fully supported.

Here's an example that shows the same workflow, written three different ways: with the fully imperative API, the fully functional API, and a mix of the two:
<CodeGroup>
```python Fully imperative API
from controlflow import Flow, Task


with Flow(name='poem_flow') as poem_flow:

    name = Task(
        objective="Get the user's name", 
        result_type=str, 
        user_access=True)

    poem = Task(
        objective="Write a short poem about the user and provided topic",
        result_type=str,
        context={"name": name, "topic": "sunset"},
    )


poem_flow.run()
print(poem.result)

# John Doe, across the days
# You shine with vibrant rays.
# ...
```
```python Fully functional API
from controlflow import flow, task


@task(user_access=True)
def get_name() -> str:
    """Get the user's name."""
    pass


@task
def write_poem(name: str, topic:str) -> str:
    """Write a short poem about the user and provided topic."""
    pass


@flow
def poem_flow(topic:str):
    name = get_name()
    poem = write_poem(name)
    return poem


print(poem_flow(topic="sunset"))

# John Doe, across the days
# You shine with vibrant rays.
# ...
```

```python Mixed APIs
from controlflow import flow, Task


@flow
def poem_flow(topic:str):

    name = Task(
        objective="Get the user's name", 
        result_type=str, 
        user_access=True)

    poem = Task(
        objective="Write a short poem about the user and provided topic",
        result_type=str,
        context={"name": name, "topic": topic},
    )

    return poem
    

print(poem_flow(topic="sunset"))

# John Doe, across the days
# You shine with vibrant rays.
# ...
```
</CodeGroup>


## Which API should I use?

<Tip>
Most users should start with the functional `@flow` decorator and imperative `Task` class, though the choice of whether to use `Task` or `@task` is ultimately a matter of personal preference and workflow requirements.
</Tip>

### Flows
Users should almost always use the `@flow` decorator to create flows. It provides a simple and concise way to define flows using Python functions that encapsulate all logic. The `@flow` decorator automatically infers the structure of the flow and its tasks, making it easy to define and run workflows. The imperative `Flow` class is available for advanced users who need more control over flow creation and execution, but requires more boilerplate.
### Tasks
For tasks, the choice between the functional and imperative APIs will depend on your workflow requirements and personal preference.

Most users should start with the imperative `Task` class. This approach leans in to an object-oriented style of programming, where tasks are explicitly defined and configured. This can be beneficial for users who prefer a more explicit and structured approach to defining tasks, with fine-grained control over task properties and behavior. 

The functional `@task` decorator creates tasks that look and behave like functions, making them relatively easy to define and use. The functional API is especially useful for prototyping, simple workflows, and tasks that do not have complex dependencies. This is because the functional API is eagerly executed by default. On the one hand this simplifies workflow design, but it also prevents agents from optimizing their work based on knowledge of the entire workflow.

## Advantages of the imperative API
The imperative API, using the `Task` class, offers the following advantages:

1. **Explicit control over task definition**: The `Task` class allows you to specify detailed objectives, instructions, agents, context, and result types for each task, providing fine-grained control over task behavior.

2. **Dynamic task creation**: Tasks can be created dynamically based on runtime conditions or data, enabling more flexible and dynamic workflows.

3. **Lazy execution**: The imperative API provides access to fine-grained control over task execution, dependencies, and error handling, allowing you to customize the workflow behavior as needed.


## Advantages of the functional API
The functional `@task` API offers several advantages:

1. **Automatic inference of task properties**: The `@task` decorator automatically infers task properties such as the objective, instructions, context, and result type from the function definition, reducing the need for explicit configuration.

2. **Eager execution**: Task-decorated functions are executed eagerly by default, which can be beneficial for some workflows and provide a more intuitive programming experience.

## Mixing functional and imperative APIs
ControlFlow allows you to mix functional and imperative APIs within a workflow, leveraging the strengths of both approaches. Most commonly, you'll seamlessly combine `@flow`-decorated functions with `Task` objects, but you can switch APIs at any time.

Here's an example that demonstrates mixing functional and imperative tasks:

```python
from controlflow import flow, task, Task

@task
def preprocess_data(raw_data: str) -> pd.DataFrame:
    """Preprocess the raw data and return a cleaned DataFrame."""
    pass

@flow
def my_flow():
    raw_data = load_raw_data()
    cleaned_data = preprocess_data(raw_data)

    analysis_task = Task(
        objective="Perform exploratory data analysis",
        context={"data": cleaned_data},
        result_type=dict,
    )

    insights = analysis_task.run()
    generate_report(insights)
```

In this example, the `preprocess_data` task is defined using the `@task` decorator, while the `analysis_task` is created imperatively using the `Task` class. The `preprocess_data` task is executed eagerly, and its result (`cleaned_data`) is passed as context to the `analysis_task`.

By mixing functional and imperative tasks, you can design workflows that are both expressive and flexible, allowing you to leverage the strengths of each approach as needed.


## Execution modes

One of the key differences between the functional and imperative APIs is how tasks are executed. The functional API defaults to eager execution, where tasks are executed immediately when called, while the imperative API defaults to lazy execution, where tasks are executed only when necessary. To learn more about these two modes, see the [Execution Mode](/docs/concepts/execution-modes) guide.
## ./docs/guides/execution-modes.mdx
---
title:  Execution Modes
---


ControlFlow supports two execution modes: eager execution and lazy execution. Understanding these modes is essential for controlling the behavior of your workflows. 

<Tip>
Please review the functional and imperative APIs in the [Workflow API guide](/docs/guides/apis) before proceeding.
</Tip>

## Eager execution

Eager mode is the default for the functional API. In this mode, flows and tasks are executed immediately. When a `@task`-decorated function is called, it is run by an AI agent right away and its result is returned. When a `@flow`-decorated function is called, it executes the function, runs every task that was created inside the flow (whether functional or imperative) and returns the results of any returned tasks.

Here's an example of eager execution:

```python
from controlflow import flow, task

@task
def write_poem(topic: str) -> str:
    """Write a short poem about the given topic."""
    pass

@flow
def my_flow(topic:str):
    
    # the poem task is immediately given to an AI agent for execution
    # and the result is returned
    poem = write_poem(topic)
    return poem

my_flow("sunset")
```

In this example, the `write_poem` task is executed by an AI agent as soon as its function is called. The AI agent generates a short poem based on the provided topic, and the generated poem is returned as the `poem` variable.

Because eager execution returns the result of each task, it makes it easy to mix task-decorated functions with normal Python code seamlessly, enabling you to use standard Python control flow statements, such as conditionals and loops, to control the execution of tasks.

## Lazy execution

Lazy execution means that tasks are not executed when they are created. Instead, ContorlFlow builds a directed acyclic graph (DAG) of tasks and their dependencies, and executes them only when necessary. The advantage of this approach is that knowledge of the entire graph, including potentially future work, can be used by agents to optimize their work. It also defers potentially expensive work (in terms of time or resources) until it is actually needed.

Lazy execution is the only mode available for the imperative API, as imperative tasks must be run explicitly. You can also run functional tasks lazily by passing `lazy=True` to the `@task` decorator or `lazy_=True` when calling the task. In lazy mode, `@task` functions return a `Task` object.

Here's an example of lazy task execution:

```python
from controlflow import flow, task

# this task will always run lazily
@task(lazy=True)
def analyze_data(data: pd.DataFrame) -> dict:
    """Analyze the given data and return insights."""
    pass

@task
def generate_report(insights: dict) -> str:
    """Generate a report based on the provided insights."""
    pass

@flow
def my_flow(data):
    
    # analyze_data runs lazily, so the `insights` variable is a Task 
    # object, as if it had been created with the imperative API
    insights = analyze_data(data)

    # `report` is a Task object because generate_report is being called lazily
    report = generate_report(insights, lazy_=True)
    return report

result = my_flow()
```

In this example, the `analyze_data` and `generate_report` tasks are defined using the `@task` decorator, but their execution is deferred. `analyze_data` is defined lazily; `generate_report` is lazy on for this specific call.

Flows can also be run lazily. In this case, a `Flow` object is returned when calling the flow function. Note that this is a very advanced feature and most users will not use it.

### Why use lazy execution?

Lazy execution is particularly useful in scenarios where you want to define the structure and dependencies of tasks upfront but delay their actual execution. This can be beneficial for planning and optimizing the execution of complex workflows. When tasks are run eagerly, the Agents can only see the workflow that's been defined up to that point. When tasks with dependencies are run lazily, the agent can see the entire workflow and optimize its execution of early task in order to produce the output it needs for later tasks. This can lead to more efficient and precise execution of the workflow.

In addition, lazy execution allows you to exercise more precise control over how tasks are executed. When you run an eagerly executed task, ControlFlow works autonomously to complete the task and all its upstream dependencies. This is similar to calling `run()` on any imperative task. However, the imperative API gives you access to methods like `run_once()` (only run an agent for one turn) and even choosing which agent to run the task with, which can be useful for debugging and optimization.

Remember that lazy execution is the only way to run tasks in the imperative API, so if you need to use imperative tasks, you'll be using lazy execution by default. Eager execution is the default for functional tasks because it is more intuitive for users expecting typical Python behavior when calling a function. 

### When are lazy tasks run?

Lazily-executed tasks are run under the following conditions:
1. When their `run()` method is called.
2. When they are an upstream dependency of another task that is run (whether eagerly or lazily)
3. At the end of an eagerly-executed flow, if they were created in that flow.

## ./docs/guides/planning.mdx
---
title: Planning
---

One of ControlFlow's core tenets is that having well-defined tasks leads to better agentic outcomes. The framework has many ways to let you dynamically generate tasks and subtasks to guide your agents.

One of the most exciting areas of active research is letting agents generate their own tasks! ControlFlow has a few ways to do this. 

## Automatically Generate Subtask DAGs

You can automatically create subtasks for any task by calling its `generate_subtasks()` method. This method will invoke an agent to come up with an actionable plan for achieving the main task; each step of the plan will become a new subtask of the main task.

```python
import controlflow as cf

task = cf.Task(
    objective="Compare the height of the tallest building "
    "in North America to the tallest building in Europe",
)

task.generate_subtasks()

print([f'{i+1}: {t.objective}' for i, t in enumerate(task.subtasks)])
```

Running the above code will print something like:

```python
[
    "1: Identify the Tallest Building in North America",
    "2: Identify the Tallest Building in Europe",
    "3: Obtain Height of the Tallest Building in North America",
    "4: Obtain Height of the Tallest Building in Europe",
    "5: Compare the Heights",
]
 ```
If you investigate more closely, you'll see that the subtasks have proper dependencies. In the above example, #3 depends on #1, #4 depends on #2, and #5 depends on #3 and #4.

<Tip>
Subtask generation isn't magic: it's a ControlFlow flow!
</Tip>

### Customizing subtask generation
You can influence subtask generation in a few ways.

#### Planning agent

By default, subtasks are generated by the first agent assigned to the parent task. You can customize this by passing an `agent` argument to `generate_subtasks()`.


#### Instructions

You can provide natural language `instructions` to help the agent generate subtasks. This is especially useful when the task is ambiguous or requires domain-specific knowledge.

## ./docs/guides/orchestration.mdx
---
title:  Orchestration
---
## Error handling
ControlFlow provides mechanisms to handle errors and exceptions that may occur during task execution.

In eager execution mode, if a task fails and raises an exception, you can catch and handle the exception using Python's standard exception handling techniques, such as `try`-`except` blocks.

Here's an example of error handling in eager mode:

```python
from controlflow import flow, task

@task
def divide_numbers(a: int, b: int) -> float:
    """Divide two numbers."""
    pass

@flow
def my_flow():
    try:
        result = divide_numbers(10, 0)
        print(result)
    except ValueError as e:
        print(f"Error: {str(e)}")
```

In this example, if the `divide_numbers` task fails because the agent recognizes that it can't divide by zero, it will raise a `ValueError`. The exception is caught in the `except` block, and an error message is printed.

In lazy execution mode, exceptions are not raised immediately but are propagated through the task DAG. When the flow is executed using the `run()` method, any exceptions that occurred during task execution will be raised at that point.

## Observability and orchestration
ControlFlow integrates with Prefect, a popular workflow orchestration tool, to provide observability and orchestration capabilities for your workflows.

Under the hood, ControlFlow tasks and flows are modeled as Prefect tasks and flows, allowing you to leverage Prefect's features for monitoring, logging, and orchestration. This integration enables you to track the execution status of tasks, monitor their progress, and access detailed logs and reports.

By default, ControlFlow configures Prefect to use a local SQLite database for storage and a ephemeral Prefect server for orchestration. You can customize the Prefect configuration to use different storage backends and orchestration setups based on your requirements.

To access the Prefect UI and view the status and logs of your workflows, you can start the Prefect server and open the UI in your web browser:

```bash
prefect server start
```

Once the server is running, you can open the Prefect UI by navigating to `http://localhost:4200` in your web browser. The UI provides a visual interface to monitor the execution of your workflows, view task statuses, and access logs and reports.

## ./docs/mint.json
{
    "$schema": "https://mintlify.com/schema.json",
    "anchors": [
        {
            "icon": "github",
            "name": "Code",
            "url": "https://github.com/PrefectHQ/ControlFlow"
        },
        {
            "icon": "book-open-cover",
            "name": "Docs",
            "url": "/"
        },
        {
            "icon": "slack",
            "name": "Community",
            "url": "https://prefect.io/slack?utm_source=controlflow&utm_medium=docs"
        }
    ],
    "colors": {
        "anchors": {
            "from": "#2D6DF6",
            "to": "#E44BF4"
        },
        "dark": "#2D6DF6",
        "light": "#E44BF4",
        "primary": "#2D6DF6"
    },
    "favicon": "/assets/ControlFlow.jpg",
    "footerSocials": {
        "github": "https://github.com/PrefectHQ/ControlFlow"
    },
    "logo": {
        "dark": "/assets/ControlFlow.jpg",
        "light": "/assets/ControlFlow.jpg"
    },
    "name": "ControlFlow",
    "navigation": [
        {
            "group": "Get Started",
            "pages": [
                "welcome",
                "installation",
                "quickstart",
                "tutorial",
                "concepts"
            ]
        },
        {
            "group": "Concepts",
            "pages": [
                "concepts/tasks",
                "concepts/agents",
                "concepts/flows"
            ]
        },
        {
            "group": "Patterns",
            "pages": [
                "patterns/dependencies",
                "patterns/subtasks",
                "guides/workflow-apis",
                "guides/execution-modes"
            ]
        },
        {
            "group": "Guides",
            "pages": [
                "guides/apis",
                "guides/execution-modes",
                "guides/planning",
                "guides/llms",
                "guides/agentic-loop",
                "guides/orchestration"
            ]
        },
        {
            "group": "Reference",
            "pages": [
                "reference/task-class",
                "reference/task-decorator"
            ]
        },
        {
            "group": "Overview",
            "pages": [
                "glossary/glossary"
            ]
        },
        {
            "group": "LLM Glossary",
            "pages": [
                "glossary/llm",
                "glossary/prompt-engineering",
                "glossary/agents",
                "glossary/agentic-workflows",
                "glossary/flow-engineering",
                "glossary/fine-tuning"
            ]
        },
        {
            "group": "ControlFlow Glossary",
            "pages": [
                "glossary/cf-task",
                "glossary/cf-agent",
                "glossary/cf-flow",
                "glossary/tools",
                "glossary/dependencies"
            ]
        },
        {
            "group": "Orchestration Glossary",
            "pages": [
                "glossary/task-orchestration",
                "glossary/flow-orchestration",
                "glossary/workflow"
            ]
        }
    ],
    "tabs": [
        {
            "name": "AI Glossary",
            "url": "glossary"
        }
    ],
    "topbarCtaButton": {
        "type": "github",
        "url": "https://github.com/PrefectHQ/ControlFlow"
    }
}
## ./docs/concepts/agents.mdx

Agents are a key concept in ControlFlow, representing the AI entities responsible for executing tasks within a workflow. Each agent has its own set of properties, methods, and capabilities that define its behavior and role in the flow.

## Creating Agents

To create an agent, use the `Agent` class:

```python
from controlflow import Agent

# minimal agent
agent = Agent()

# agent with options
agent = Agent(
    name="DataAnalyst",
    description="An AI agent specialized in data analysis",
    instructions="Perform data analysis tasks efficiently and accurately",
    tools=[search_web, generate_plot],
    model=gpt_35_turbo,
)
```

In this example, we create an agent named "DataAnalyst" with a description and specific instructions. The `tools` parameter is used to provide the agent with a list of tools it can use during task execution. These tools are essentially Python functions that the agent can call at any time.

## Agent Properties

An agent has the following key properties:

- `name`: The name of the agent, which serves as an identifier and is visible to other agents in the workflow. Names do not have to be unique, as agents also have IDs, but it is good practice to use unique names to avoid confusion.
- `description`: A brief description of the agent's role or specialization. This information is visible to other agents.
- `instructions`: Specific instructions or guidelines for the agent to follow during task execution. These instructions are private and not shared with other agents.
- `tools`: A list of tools available to the agent. Tools are Python functions that the agent can call to perform specific actions or computations.
- `model`: A LangChain model that powers the agent responses.
- `user_access`: Indicates whether the agent has access to user interactions. If set to `True`, the agent will be provided with the `talk_to_human` tool to communicate with users.

These properties help define the agent's characteristics, behavior, and capabilities within the flow.

<Tip>
Note that instructions, tools, and user access are all agent-level settings that can also be provided or enabled at the task level. For example, a task that permits user access will allow any agent assigned to it to interact with users while working on that task, even if the agent itself does not have user access enabled.

</Tip>
## Assigning Agents to Tasks

To assign an agent to a task, you can use the `agents` parameter when creating a task. Each task requires at least one assigned agent, and will use a default agent if none are provided. 

Here's an example of assigning multiple agents to a task:

```python
import controlflow as cf

data_analyst = cf.Agent(name="Data Analyst")
data_visualizer = cf.Agent(name="Data Visualizer")

task = cf.Task(
    objective="Analyze sales data",
    agents=[data_analyst, data_visualizer]
)
```

In this example, we create a task with the objective "Analyze sales data" and assign two agents, `data_analyst` and `data_visualizer`, to it. Agents can only work on tasks they are assigned to.

## Specifying an Agent's Model

Each agent is backed by a specific LLM that powers its responses and interactions. This allows you to choose the most suitable model for your needs, based on factors such as performance, latency, and cost.

To customize the LLM, provide a model when creating your agent:
```python
import controlflow as cf
from langchain_openai import ChatOpenAI

gpt_35_agent = cf.Agent(model=ChatOpenAI(model="gpt-3.5-turbo"))
```
For a full guide on how to use LLMs in ControlFlow, including changing the default globally, please refer to the [LLMs guide](/guides/llms).


## User Access and Interaction

Agents with the `user_access` flag set to `True` have the ability to interact with users using the `talk_to_human` tool. This tool allows agents to send messages to users and receive responses.

```python
from controlflow import Agent

agent = Agent(
    name="Support Agent",
    description="An AI agent that interacts with users",
    user_access=True
)
```

In this example, we create an agent named "UserAssistant" with the `user_access` flag set to `True`. This agent will have access to the `talk_to_human` tool to communicate with users when necessary.

## ./docs/concepts/tasks.mdx

Tasks are the fundamental building blocks of ControlFlow workflows, representing specific objectives or goals within an AI-powered application. They act as a bridge between AI agents and application logic, enabling developers to define and structure desired outcomes in a clear and intuitive manner.

## Creating Tasks

ControlFlow provides two convenient ways to create tasks: using the `Task` class or the `@task` decorator.

### Using the `Task` Class

The `Task` class offers a flexible and expressive way to define tasks by specifying various properties and requirements.

```python
from controlflow import Task

interests = Task(
    objective="Ask user for three interests",
    result_type=list[str],
    user_access=True,
    instructions="Politely ask the user to provide three of their interests or hobbies."
)
```

The `Task` class allows you to explicitly define the objective, instructions, agents, context, result type, tools, and other properties of a task. This approach provides full control over the task definition and is particularly useful when you need to specify complex requirements or dependencies.

### Using the `@task` Decorator

The `@task` decorator provides a concise and intuitive way to define tasks using familiar Python functions. The decorator automatically infers key properties from the function definition, making task creation more streamlined.

```python
from controlflow import task

@task(user_access=True)
def get_user_name() -> str:
    "Politely ask the user for their name."
    pass
```

When using the `@task` decorator, the objective is inferred from the function name, instructions are derived from the docstring, context is inferred from the function arguments, and the result type is inferred from the return annotation. This approach is ideal for simple tasks or when you want to leverage existing functions as tasks.

## Defining Task Objectives and Instructions

Clear objectives and instructions are crucial for guiding AI agents and ensuring successful task execution.

### Objectives

The objective of a task should be a brief description of the task's goal or desired outcome. It helps both developers and AI agents understand the purpose of the task and what it aims to achieve.

When defining objectives, aim for clarity and specificity. Use action-oriented language and avoid ambiguity. For example:

```python
summary_task = Task(
    objective="Summarize the key points of the customer feedback",
    result_type=str,
)
```

### Instructions

Instructions provide detailed guidelines or steps for completing the task. They offer more context and direction to the AI agents, beyond what is conveyed in the objective.

When writing instructions, use concise language and bullet points or numbered steps if applicable. Avoid ambiguity and provide sufficient detail to enable the AI agents to complete the task effectively.

```python
data_analysis_task = Task(
    objective="Analyze the sales data and identify top-performing products",
    instructions="""
    1. Load the sales data from the provided CSV file
    2. Calculate the total revenue for each product
    3. Sort the products by total revenue in descending order
    4. Select the top 5 products based on total revenue
    5. Return a list of tuples containing the product name and total revenue
    """,
    result_type=list[tuple[str, float]],
)
```

## Specifying Result Types

The `result_type` property allows you to define the expected type of the task's result. It provides a contract for the task's output, ensuring consistency and enabling seamless integration with the broader workflow.

By specifying a result type, you make it clear to both the AI agents and the developers what kind of data to expect. The `result_type` can be any valid Python type, such as `str`, `int`, `list`, `dict`, or even custom classes.

```python
sentiment_analysis_task = Task(
    objective="Analyze the sentiment of the given text",
    result_type=float,
)

product_classification_task = Task(
    objective="Classify the product based on its description",
    result_type=list[str],
)
```

When using the `@task` decorator, the result type is inferred from the function's return annotation:

```python
@task
def count_words(text: str) -> int:
    "Count the number of words in the provided text."
    pass
```

## Assigning Agents and Tools

ControlFlow allows you to assign specific AI agents and tools to tasks, enabling you to leverage their specialized skills and capabilities.

### Assigning Agents

By assigning agents to a task, you can ensure that the most suitable agent is responsible for executing the task. Agents can be assigned using the `agents` property of the `Task` class or the `agents` parameter of the `@task` decorator.

```python
from controlflow import Agent

data_analyst = Agent(
    name="DataAnalyst", 
    description="Specializes in data analysis and statistical modeling",
)

business_analyst = Agent(
    name="BusinessAnalyst", 
    description="Expert in business strategy and market research",
    instructions="Use the DataAnalyst's insights to inform business decisions.",
)

analysis_task = Task(
    objective="Analyze the customer data and provide insights",
    agents=[data_analyst, business_analyst],
    result_type=str,
)
```

If no agents are explicitly assigned to a task, ControlFlow will use the agents defined in a task's parent task, flow, or fall back on a global default agent, respectively.

### Providing Tools

Tools are Python functions that can be used by agents to perform specific actions or computations. By providing relevant tools to a task, you empower the AI agents to tackle more complex problems and enhance their problem-solving abilities.

Tools can be specified using the `tools` property of the `Task` class or the `tools` parameter of the `@task` decorator.

```python
def calculate_square_root(number: float) -> float:
    return number ** 0.5

calculation_task = Task(
    objective="Calculate the square root of the given number",
    tools=[calculate_square_root],
    result_type=float,
)
```

## Handling User Interaction

ControlFlow provides a built-in mechanism for tasks to interact with human users. By setting the `user_access` property to `True`, a task can indicate that it requires human input or feedback to be completed.

```python
feedback_task = Task(
    objective="Collect user feedback on the new feature",
    user_access=True,
    result_type=str,
    instructions="Ask the user to provide their thoughts on the new feature.",
)
```

When a task with `user_access=True` is executed, the AI agents assigned to the task will be given access to a special `talk_to_human` tool. This tool allows the agents to send messages to the user and receive their responses, enabling a conversation between the AI and the human.

## Creating Task Dependencies and Subtasks

ControlFlow allows you to define dependencies between tasks and create subtasks to break down complex tasks into smaller, more manageable units of work.

### Task Dependencies

Dependencies can be specified using the `depends_on` property of the `Task` class. By specifying dependencies, you ensure that tasks are executed in the correct order and have access to the necessary data or results from previous tasks.

```python
data_collection_task = Task(
    objective="Collect user data from the database",
    result_type=pd.DataFrame,
)

data_cleaning_task = Task(
    objective="Clean and preprocess the collected user data",
    depends_on=[data_collection_task],
    result_type=pd.DataFrame,
)

data_analysis_task = Task(
    objective="Analyze the cleaned user data and generate insights",
    depends_on=[data_cleaning_task],
    result_type=dict,
)
```

### Subtasks
Subtasks allow you to break down complex tasks into smaller steps and manage the workflow more effectively. Subtasks can be defined either by creating tasks with the context of another task, or by passing a task as a `parent` parameter to the subtask. 

<CodeGroup>
```python Context manager
from controlflow import Task

with Task(objective="Prepare data", result_type=list) as parent_task:
    child_task_1 = Task('Load data from the source', result_type=list)
    child_task_2 = Task(
        'Clean and preprocess the loaded data', 
        result_type=list, 
        context=dict(data=child_task_1),
    )
```

```python Parent parameter
from controlflow import Task

parent_task = Task(objective="Prepare data", result_type=list)
child_task_1 = Task(
    'Load data from the source', 
    result_type=list, 
    parent=parent_task,
)
child_task_2 = Task(
    'Clean and preprocess the loaded data', 
    result_type=list, 
    context=dict(data=child_task_1),
    parent=parent_task,
)
```
</CodeGroup>

## Running Tasks

### Running to Completion
Tasks can be executed using the `run()` method, which coordinates the execution of the task, its subtasks, and any dependent tasks, ensuring that the necessary steps are performed in the correct order.

```python
from controlflow import Task

title_task = Task('Generate a title for a poem about AI', result_type=str)
poem_task = Task(
    'Write a poem about AI using the provided title', 
    result_type=str, 
    context=dict(title=title_task),
)

poem_task.run()
print(poem_task.result)
```

When you run a task, ControlFlow orchestrates the execution of the task and its dependencies in a loop, ensuring that each step is completed successfully before proceeding to the next one. The `run()` method exits when the task is completed, at which point the task's result is available (if it succeeded) or an exception is raised (if it failed).

You can limit the number of iterations in the task loop by passing `max_iterations=n` to the `run()` method, or set a global limit using `controlflow.settings.max_task_iterations`. There is no defaul limit.

<Tip>
**Do I need to create a flow?** 

Tasks must always be run within the context of a flow in order to manage dependencies, history, and agent interactions effectively. As a convenience, if you call `task.run()` outside a flow context, a new flow will be automatically created to manage the task's execution for that run only. In the above example, `poem_task.run()` implicitly creates a new flow for both tasks.

This is useful for testing tasks in isolation or running them as standalone units of work. However, it can lead to confusing behavior if you try to combine multiple tasks that created their own flows, because they will not have access to each other's context or history.
</Tip>

### Controlling Iteration

The `run()` method starts a loop and orchestrates the execution of the task, its subtasks, and dependencies until the task is completed. If you need more fine-grained control over task execution, you can use the `run_once()` method to execute only a single step of the graph.

```python
from controlflow import Flow, Task

title_task = Task('Generate a title for a poem about AI', result_type=str)
poem_task = Task(
    'Write a poem about AI using the provided title', 
    result_type=str, 
    context=dict(title=title_task),
)

with Flow():
    while poem_task.is_incomplete():
        poem_task.run_once()
    print(poem_task.result)
```

Note that run_once requires the task to be run within a flow context, as it relies on the flow to manage the task's execution and history over each invocation.

## ./docs/concepts/flows.mdx

Flows are the high-level containers that encapsulate and orchestrate AI-powered workflows in ControlFlow. They provide a structured and organized way to manage tasks, agents, tools, and context, enabling developers to build complex and dynamic applications with ease.

## Creating Flows

Flows can be created using the `Flow` class or the `@flow` decorator. 

### The `@flow` Decorator

The `@flow` decorator provides a convenient way to define a flow using a Python function.

```python
from controlflow import flow

@flow
def data_processing_flow():
    data = load_data()
    cleaned_data = clean_data(data)
    insights = analyze_data(cleaned_data)
    return insights
```

When using the `@flow` decorator, the decorated function becomes the entry point for the flow. The function can contain tasks, which are automatically executed when the flow is run. The `@flow` decorator also allows you to specify flow-level properties such as agents, tools, and context.

### The `Flow` Class

The `Flow` class allows you to explicitly define a flow and its properties.

```python
from controlflow import Flow

flow = Flow(
    name="Data Processing Flow",
    description="A flow to process and analyze data",
    agents=[data_analyst, business_analyst],
    tools=[data_loader, data_cleaner],
)
```

By creating a `Flow` instance, you can specify the name, description, agents, tools, and other properties of the flow. This approach provides full control over the flow definition and is particularly useful when you need to customize the flow's behavior or configure advanced settings.

#### Adding tasks to a flow

Tasks can be added to a flow object in two ways: by calling `Flow.add_task(task)` or by creating the tasks inside the `Flow` context manager.
<CodeGroup>
```python Using a flow context
from controlflow import Flow, Task

with Flow() as flow:
    task = Task('Load data')
```

```python Adding tasks imperatively
from controlflow import Flow, Task

flow = Flow()
task = Task('Load data')
flow.add_task(task)
```
</CodeGroup>
### Which Approach Should I Use?

<Tip>
    **tldr:** Prefer the `@flow` decorator for simplicity and conciseness. Use the `Flow` class only for advanced customization and configuration.
</Tip>

Both the `Flow` class and the `@flow` decorator offer ways to compartmentalize and structure your workflow.

In general, users should prefer the `@flow` decorator for its simplicity and ease of use. The `@flow` decorator allows you to define a flow using a single function, making it easy to read and understand. It automatically registers tasks within the flow and handles the execution of tasks when the flow is run. 

The `Flow` class is only recommended for use when a group of tasks need to be quickly placed into a separate flow to isolate their work from the main flow. This circumstance is rare and can also be achieved by using the `@flow` decorator with a separate function.

## Flow Histories

Each flow maintains a shared history of all agent messages, actions, and tasks that were executed within it. This history is accessible to all agents and tasks within the flow, allowing them to share information and context. The shared history enables agents to collaborate, coordinate, and communicate effectively, leading to more intelligent and adaptive behavior. Every flow has a `thread_id` that uniquely identifies its history, allowing agents to distinguish between different flows and maintain separate histories.

### Private Histories

In general, every action an agent takes -- including sending messages, using tools, and getting tool results -- is recorded in the flow's history as a series of agent messages. Sometimes, you may want to let agents work in isolation without making their activity visible to other agents. For example, if an agent is summarizing a large document, then it will have the entire text of the document somewhere in its own history; there's no need to share that with other agents.

Because each flow creates a new thread, the simplest way to create a private history is to create a new flow. To facilitate this, flows automatically inherit a copy of their parent flow's history.

```python
import controlflow as cf

@cf.flow
def study_documents():

    # ... other tasks

    # creating a nested flow will also create a private history 
    # for the summary task; the full document text will not be 
    # visible to agents in the main `study_documents` flow
    with cf.Flow() as document_flow:
        summary = cf.Task('summarize the document', tools=[load_document]).run()

    cf.Task('analyze the summary', context=dict(summary=summary))

    # ... other tasks
```

## Flow Properties

Flows have several key properties that define their behavior and configuration.

### Name and Description

The `name` and `description` properties allow you to provide a human-readable name and a brief description of the flow. These properties help in identifying and understanding the purpose of the flow.

```python
flow = Flow(
    name="Data Processing Flow",
    description="A flow to process and analyze data",
)
```

### Agents and Tools

The `agents` and `tools` properties allow you to specify AI agents and tools that are available to tasks throughout the flow. 

Flow-level agents are used by tasks **unless** the tasks have their own agents assigned. Flow-level tools are used by tasks **in addition** to any tools they have defined.

```python
flow = Flow(
    agents=[data_analyst, business_analyst],
    tools=[data_loader, data_cleaner],
)
```


### Context

The `context` property allows you to define a shared context that is accessible to all tasks and agents within the flow. The context can contain any relevant information or data that is required throughout the flow.

```python
flow = Flow(
    context={
        "data_source": "path/to/data.csv",
        "target_audience": "marketing_team",
    }
)
```

The context can be accessed and modified by tasks and agents during the flow execution, enabling dynamic and adaptive behavior based on the flow's state.

## Running Flows

To a run a `@flow` decorated function, simply call the function with appropriate arguments. The arguments are automatically added to the flow's context, making them visible to all tasks even if they aren't passed directly to that task's context. Any tasks returned from the flow are automatically resolved into their `result` values.

To run a `Flow` instance, use its `run()` method, which executes all of the tasks that were defined within the flow. You can then access the results of individual tasks by referencing their `result` attribute, or by calling them (if they are `@task`-decorated functions).

<CodeGroup>
```python @flow decorator
@flow
def item_flow():
    price = Task('generate a price between 1 and 1000', result_type=int)
    item = Task(
        'Come up with an common item that has the provided price', 
        result_type=str, 
        context=dict(price=price)
    )
    return item

# call the flow; the result is automatically resolved 
# as the result of the `item` task.
item = item_flow()
```
```python Flow class
with Flow() as item_flow:
    price = Task('generate a price between 1 and 1000', result_type=int)
    item = Task(
        'Come up with an common item that has the provided price', 
        result_type=str, 
        context=dict(price=price)
    )

# run all tasks in the flow
item_flow.run()
# access the item task's result
item.result
```
</CodeGroup>

<Tip>
**What happens when a flow is run?**

When a flow is run, the decorated function is executed and any tasks created within the function are registered with the flow. The flow then orchestrates the execution of the tasks, resolving dependencies, and managing the flow of data between tasks. If the flow function returns a task, or a nested collection of tasks, the flow will automatically replace them with their final results. 
</Tip>

## Controlling Execution

ControlFlow provides many mechanisms for determining how tasks are executed within a flow. So far, we've only looked at flows composed entirely of dependent tasks. These tasks form a DAG which is automatically executed when the flow runs. 

### Control Flow

Because a flow function is a regular Python function, you can use standard Python control flow to determine when tasks are executed and in what order. At any point, you can manually `run()` any task in order to work with its result. Running a task inside a flow will also run any tasks it depends on.

In this flow, we flip a coin to determine which poem to write. The coin toss task is run manually, and the result is used to determine which poem task to return, using a standard Python `if` statement:

```python
@flow
def conditional_flow():
    coin_toss_task = Task('Flip a coin', result_type=['heads', 'tails'])
    # manually run the coin-toss task
    outcome = coin_toss_task.run()

    # generate a different task based on the outcome of the toss
    if outcome == 'heads':
        poem = Task('Write a poem about Mt. Rushmore', result_type=str)
    elif outcome == 'tails':
        poem = Task('Write a poem about the Grand Canyon', result_type=str)
    
    # return the poem task
    return poem

print(conditional_flow())
# Upon granite heights, 'neath skies of blue,
# Mount Rushmore stands, a sight to view.
# ...
```

## ./docs/glossary/agents.mdx
Agents are autonomous AI systems that can perform complex tasks, make decisions, and interact with their environment without continuous human intervention. These agents leverage the advanced capabilities of LLMs, such as natural language understanding, reasoning, and generation, to operate independently and achieve specific goals.

There are three key characteristics that distinguish agents from single-shot LLM responses:

1. Iteration: Agents engage in multi-step processes, continuously refining their actions based on feedback and new information. Unlike single-shot responses, which provide a one-time output based on a given prompt, agents iterate on their own outputs, allowing for more dynamic and adaptive behavior.

2. Tool use: Agents can interact with external tools and systems to gather information, perform computations, or execute actions. This ability to use tools enables agents to extend their capabilities beyond the knowledge and skills inherent in the LLM itself. By integrating tool use into their decision-making process, agents can solve more complex problems and adapt to a wider range of scenarios.

3. Planning and workflow: Agents are designed to break down complex tasks into smaller, manageable steps and create structured workflows to accomplish their goals. They can prioritize subtasks, make decisions based on intermediate results, and adjust their plans as needed. This planning capability allows agents to handle multi-faceted problems that require a sequence of coordinated actions.

LLM agents maintain an understanding of the ongoing context and use this information to guide their actions. They actively work towards achieving specific objectives or goals by selecting appropriate strategies, adapting to challenges, and learning from their experiences. The autonomous and goal-oriented nature of LLM agents enables them to operate effectively in a variety of domains and scenarios, making them well-suited for [agentic workflows](/glossary/agentic-workflows).

## ./docs/glossary/tools.mdx
Tools in ControlFlow are specialized functions or resources that agents can use to accomplish specific tasks within a workflow. They provide the agents with additional capabilities beyond their inherent natural language processing abilities, enabling them to perform more complex and varied operations.

Tools can include Python functions, APIs, libraries, or any resource that an agent might need to fulfill a task’s requirements. For instance, a tool might be a function to fetch data from a database, process and analyze data, interact with external services, or perform calculations.

By equipping agents with the appropriate tools, developers can enhance the functionality and efficiency of their workflows, ensuring that agents can effectively complete tasks that require specialized knowledge or operations. Tools are defined and associated with tasks or agents, and they enable a modular and extensible approach to building AI-powered workflows in ControlFlow.
## ./docs/glossary/flow-engineering.mdx
"Flow engineering" is a term increasingly used to describe a specific approach to designing and optimizing [agentic workflows](/glossary/agentic-workflows) for LLMs. In flow engineering, the focus is on engineering the workflow itself to guide the agent's decision-making process and improve the quality of its outputs.

Similar to how [prompt engineering](/glossary/prompt-engineering) emphasizes the importance of crafting natural-language messages to elicit desired responses from LLMs, flow engineering recognizes the significance of the overall workflow structure in determining the agent's behavior and performance. By carefully designing the steps, decision points, and feedback loops within the workflow, developers can create agents that are more effective, efficient, and adaptable.

Flow engineering involves breaking down complex tasks into smaller, manageable components and defining the optimal sequence of actions for the agent to follow. This structured approach allows for better control over the agent's behavior and enables developers to incorporate domain-specific knowledge and best practices into the workflow.
## ./docs/glossary/cf-agent.mdx
---
title: Agent
---

<Info>
This glossary entry is about the term "agent" in the context of ControlFlow. For LLM agents in general, see the [Agents](/glossary/agents) entry.
</Info>

An Agent in ControlFlow is an autonomous entity designed to execute tasks within a workflow. Agents leverage the capabilities of LLMs to perform various functions, such as generating text, answering questions, and interacting with users. Each agent can be tailored with specific instructions, tools, and models to handle distinct roles or domains effectively.

Agents are fundamental to the ControlFlow framework, enabling the execution of tasks according to the defined objectives and context. They operate independently, using the provided instructions to achieve the desired outcomes. Agents can also interact with each other and with human users when necessary, making them versatile components in creating sophisticated and dynamic AI-powered workflows.

By assigning appropriate agents to tasks, developers can ensure that each task is handled by the most suitable entity, optimizing the overall performance and efficiency of the workflow. ControlFlow's agent-based architecture allows for seamless integration of AI capabilities into traditional software workflows, providing a robust and scalable solution for complex application logic.
## ./docs/glossary/flow-orchestration.mdx
---
title: Flow
---

<Info>
This glossary entry is about the term "flow" in the context of workflow orchestration. For ControlFlow flows specifically, see the [Flow](/glossary/flow) entry.
</Info>

In the context of workflow orchestration, a flow represents the overall sequence or arrangement of tasks that make up a complete workflow. A flow defines the logical structure and order in which tasks should be executed to achieve a specific goal or outcome. It encapsulates the dependencies, control flow, and data flow between tasks. 

Orchestration frameworks use the flow definition to coordinate the execution of tasks, handle data passing between them, and manage the overall lifecycle of the workflow. Flows can be designed to handle complex scenarios, including conditional branching, parallel execution, and error handling.
## ./docs/glossary/fine-tuning.mdx
---
title: Fine-tuning
---

Fine-tuning is a process in machine learning where a pre-trained model, such as an [LLM](/glossary/llm), is further trained on a specific dataset to adapt it to a particular task or domain. This process leverages the broad knowledge and language understanding that the model has already acquired during its initial training on large and diverse datasets.

Fine-tuning involves using a smaller, task-specific dataset to continue training the pre-trained model. By doing so, the model can learn to perform more specialized tasks with greater accuracy and relevance. For example, an LLM can be fine-tuned on a dataset of medical texts to improve its performance in medical question answering or on a dataset of legal documents to enhance its capabilities in legal text analysis.

The fine-tuning process typically involves adjusting the model’s parameters using techniques such as supervised learning, where the model learns to produce the correct output based on the provided input and corresponding labels. This approach allows the model to retain its general language understanding while becoming more proficient in the specific domain or task at hand. Fine-tuning is a powerful technique that enables the adaptation of versatile LLMs to a wide range of applications, ensuring high performance and relevance in specialized contexts.
## ./docs/glossary/agentic-workflows.mdx
Agentic workflows use LLMs as autonomous [agents](/glossary/agents) to achieve a goal. The LLM is invoked iteratively to initiate and manage processes. For example, it can autonomously handle tasks such as scheduling meetings, processing customer queries, or even conducting research by interacting with APIs and databases. The model uses contextual understanding to navigate these tasks, making decisions based on the information it processes in real-time.

Any automated workflow that invokes an AI agent is considered "agentic", even if part or most of the workflow is executed as traditional software. This is because special considerations must be made to accommodate the unique requirements of AI agents, no matter how much of the workflow they automate.

The key characteristics of an agentic workflow include:

- Autonomy: The LLM operates independently for extended periods, adapting to dynamic environments and making real-time adjustments based on the evolving context of the task.

- Contextual understanding: The model maintains an understanding and memory of the ongoing context and uses this information to guide its actions, ensuring coherent and consistent responses.

- Decision-making: The LLM makes decisions based on the information it processes, selecting appropriate strategies and adapting to challenges to achieve its goals.

- Interaction with external systems: The model can interact with APIs, databases, and other tools to gather information, perform computations, or execute actions, extending its capabilities beyond its inherent knowledge and skills.

Rather than single-shot [prompt engineering](/glossary/prompt-engineering), agentic workflows can be enhanced through the application of [flow engineering](/glossary/flow-engineering) techniques, which involve designing and optimizing the workflow itself to guide the agent's decision-making process and improve the quality of its outputs. This seeks to maintain a balance of autonomy and structure in the agent's operations.


## ./docs/glossary/cf-task.mdx
---
title: Task
---
A task represents a discrete objective or goal within a ControlFlow workflow that an AI agent needs to solve. Tasks are the fundamental building blocks of ControlFlow and act as a bridge between AI agents and application logic. Each task is defined by its specific objective, instructions, expected result type, and any required context or tools.

Tasks can have [dependencies](/glossary/dependencies) that define their relationships and execution order. Dependencies ensure that tasks are completed in a logical sequence, where one task's output may be required as input for another, or certain tasks must be completed before others can begin. This allows developers to create complex workflows that are easy to understand and manage, ensuring that each task is executed with the necessary context and prerequisites.

Tasks have one or more [agents](/glossary/agent) assigned to them. By assigning appropriate agents, developers can optimize the execution of tasks by leveraging the specialized capabilities or model characteristics of different agents, ensuring each task is handled by the most suitable entity.

By specifying the parameters and dependencies of each task, developers can build sophisticated and dynamic workflows that leverage the full capabilities of AI agents in a structured and efficient manner.
## ./docs/glossary/workflow.mdx
A workflow is a sequence of interconnected tasks or steps that represent a specific business process or operation. In the context of orchestration, a workflow defines the order and dependencies of these tasks, ensuring that they are executed in a coordinated and efficient manner.

Workflows are commonly used in complex systems to automate and streamline processes, such as data processing, application deployment, or service orchestration. They provide a high-level view of the entire process, allowing developers and operators to define, manage, and monitor the execution of tasks.

In an orchestration system, a workflow typically consists of multiple activities, each representing a specific task or operation. These activities can be executed sequentially, in parallel, or based on certain conditions, enabling the system to handle complex scenarios and adapt to changing requirements.

Note that an [agentic workflow](/glossary/agentic-workflow) is a specific type of workflow that leverages AI agents to perform tasks and make decisions within the process. By combining human and machine intelligence, agentic workflows can automate repetitive tasks, optimize resource allocation, and improve decision-making in various domains.
## ./docs/glossary/llm.mdx
---
title: Large language models
---
A Large language model (LLM) is a type of artificial intelligence model trained on vast amounts of text data to understand and generate human-like language. Based on deep learning architectures such as Transformer models, LLMs capture complex patterns and relationships within the training data. Their extensive size, often containing billions of parameters, enables them to develop a deep understanding of language and acquire a broad range of knowledge.

LLMs excel in various natural language processing tasks, including text generation, language translation, question answering, and sentiment analysis. Their ability to generate contextually relevant and meaningful responses makes them valuable for applications like chatbots, content creation, and language-based interfaces. LLMs are trained using self-supervised learning techniques, predicting the next word or sequence of words in a given context. This exposure to diverse text data allows them to grasp the intricacies of language, including grammar, syntax, semantics, and world knowledge, enabling them to produce coherent and contextually appropriate responses.

More than just generating text, LLMs encode knowledge that can be used to produce a variety of non-algorithmic outputs, including using tools, writing code, generating images, and creating music. LLMs can also be [fine-tuned](/glossary/fine-tuning) on specific tasks or domains to improve performance on targeted applications.

However, LLMs have limitations. They can generate biased or factually incorrect outputs based on biases in their training data. They may struggle with tasks requiring deep reasoning, common sense understanding, or domain-specific knowledge. Additionally, the training and deployment of large-scale LLMs can be computationally intensive and resource-demanding. Despite these challenges, LLMs remain powerful tools for building sophisticated language-based applications.
## ./docs/glossary/glossary.mdx
---
title: Welcome
---
Welcome to ControlFlow's AI Glossary! 

This glossary provides definitions and explanations for key concepts in modern AI and the ControlFlow framework. Whether you're new to ControlFlow or looking to deepen your understanding, this resource is designed to help you navigate the terminology and concepts that are essential for working with LLMs and AI workflows.

## ./docs/glossary/dependencies.mdx
Dependencies in ControlFlow refer to the relationships between tasks that dictate the order and conditions under which tasks are executed. They ensure that tasks are completed in a logical sequence, where one task’s output may be required as input for another, or certain tasks must be completed before others can begin.

There are several types of dependencies in ControlFlow:

- Sequential dependencies: One task must be completed before another can start.
- Context dependencies: The result of one task is used as input for another.
- Subtask dependencies: A task consists of multiple subtasks that must be completed before the parent task is considered done.

Dependencies help in managing complex workflows by defining clear relationships and execution order among tasks. By specifying dependencies, developers can create structured and efficient workflows that ensure the correct flow of data and completion of tasks, thereby enhancing the reliability and maintainability of AI-powered applications.
## ./docs/glossary/task-orchestration.mdx
---
title: Task
---

<Info>
This glossary entry is about the term "task" in the context of workflow orchestration. For ControlFlow tasks specifically, see the [Task](/glossary/task) entry.
</Info>

In the context of workflow orchestration, a task represents a single unit of work or a specific step within a larger workflow. Tasks are the building blocks of workflows and encapsulate discrete actions or operations that need to be performed. Each task typically has input parameters, execution logic, and produces an output or result.

Tasks can have upstream dependencies on other tasks, meaning they may require the completion of certain tasks before they can start executing. These dependencies define the order and relationship between tasks within a workflow, as well as move data between tasks. Parent/child dependencies help organize execution by nesting tasks within other tasks.
## ./docs/glossary/prompt-engineering.mdx
Prompt engineering is the practice of crafting precise and effective input prompts to elicit desired responses from large language models (LLMs). This method focuses on designing the exact wording, structure, and context of the prompt to guide the model towards generating specific outputs. It requires an understanding of the model’s capabilities and the nuances of language to maximize the quality and relevance of the responses.

Unlike [flow engineering](/glossary/flow-engineering), which involves a multi-step, iterative process to refine outputs, prompt engineering aims to achieve the desired result with a single, well-constructed input. This approach is particularly useful for straightforward tasks where the model's initial response is expected to be accurate and sufficient. However, it can be limited in handling complex problems that require deeper analysis and iterative refinement.

Prompt engineering is essential in scenarios where quick, efficient responses are needed, and the task complexity is manageable with a single input. It is a critical skill for developers and users who interact with LLMs, enabling them to harness the model's full potential by providing clear and concise prompts that lead to high-quality outputs.
## ./docs/glossary/cf-flow.mdx
---
title: Flow
---

A flow is a high-level container that encapsulates and orchestrates an entire AI-powered workflow in ControlFlow. It provides a structured way to manage tasks, agents, tools, and shared context. A flow maintains a consistent state across all its components, allowing agents to communicate and collaborate effectively.

Flows allow developers to break down complex application logic into discrete tasks, define the dependencies and relationships between them, and assign suitable agents to execute them. By providing a high-level orchestration mechanism, flows enable developers to focus on the logic of their application while ControlFlow manages the details of agent selection, data flow, and error handling.
## ./docs/quickstart.mdx
---
title: "Quickstart"
description: Build your first agentic workflow in under five minutes
---

import Installation from '/snippets/installation.mdx';

This quickstart is designed to **show** you how ControlFlow works, rather than **teach** you. For a more detailed introduction, check out the full [tutorial](/tutorial).

<Installation />

## Tasks and Tools

In ControlFlow, you define your agentic workflows using tasks and tools.

A **task** represents a discrete objective that you want an AI agent to complete, such as "write a poem" or "summarize this article". Tasks are the building blocks of your agentic workflows. They can depend on the results of other tasks, allowing you to create complex, multi-step processes.

**Tools** help your agents extend their capabilities by providing them with additional functions they can use to complete tasks. For example, you might provide a calculator tool to help an agent perform arithmetic calculations, or a database query tool to retrieve information from a database.

Let's see tasks and tools in action:


```python
import controlflow as cf
import random

# this function will be used as a tool 
def roll_dice(n: int) -> int:
    '''Roll n dice'''
    return [random.randint(1, 6) for _ in range(n)]


@cf.flow
def dice_flow():

    # task 1: ask the user how many dice to roll
    user_task = cf.Task(
        "Ask the user how many dice to roll", 
        result_type=int, 
        user_access=True
    )

    # task 2: roll the dice
    dice_task = cf.Task(
        "Roll the dice",
        context=dict(n=user_task),
        tools=[roll_dice],
        result_type=list[int],
    )

    return dice_task


result = dice_flow()
print(f"The result is: {result}")
```

In this example, we define a flow with two dependent tasks: the first asks the user for input, and the second rolls some dice. The `roll_dice` function is provided as a tool to the second task, allowing the agent to use it to generate the result.

<Tip>
All tasks in a `@flow` function are run automatically when the function is called, but you can run tasks eagerly by calling `task.run()`.
</Tip>


### Recap
<Check>
**What we learned**
- Tasks are how you create goals for agents
- Tasks have a `result_type` that determines the type of data they return
- Tasks have a `context` that can include results of other tasks. These dependencies permit complex multi-step workflows
- If `tools` or `user_access` is provided, the agent can use them to complete the task
</Check>  


---

## Agents and Flows

**Agents** are AI models that complete tasks in your workflows. In the previous example, we didn't specify which agent should complete each task. By default, ControlFlow uses a generic agent that is capable of handling a wide variety of tasks. However, you can also create specialized agents that are optimized for particular types of tasks.

A **flow** is a container that ensures that all agents share a common context and history. In addition to grouping tasks, as is the previous example, this allows multiple agents to collaborate on a larger objective by working on individual tasks within the flow, even if they are backed by different LLM models.

Let's see an example of three specialized agents collaborating in a flow:

```python
import controlflow as cf

# Create three agents
writer = cf.Agent(
    name="Writer",
    description="An AI agent that writes paragraphs",
)

editor = cf.Agent(
    name="Editor",
    description="An AI agent that edits paragraphs for clarity and coherence",
)

manager = cf.Agent(
    name="Manager",
    description="An AI agent that manages the writing process",
    instructions="""
        Your goal is to ensure the final paragraph meets high standards 
        of quality, clarity, and coherence. You should be strict in your 
        assessments and only approve the paragraph if it fully meets 
        these criteria.
        """,
)


@cf.flow
def writing_flow():
    draft_task = cf.Task(
        "Write a paragraph about the importance of AI safety",
        agents=[writer],
    )

    # we will continue editing until the manager approves the paragraph
    approved = False
    while not approved:

        edit_task = cf.Task(
            "Edit the paragraph for clarity and coherence",
            context=dict(draft=draft_task),
            agents=[editor],
        )

        approval_task = cf.Task(
            "Review the edited paragraph to see if it meets the quality standards",
            result_type=bool,
            context=dict(edit=edit_task),
            agents=[manager],
        )

        # eagerly run the approval task to see if the paragraph is approved
        approved = approval_task.run()

    return approved, edit_task.result


approved, draft = writing_flow()
print(f'{"Approved" if approved else "Rejected"} paragraph:\n{draft}')
```

In this example, we create three agents: a `writer`, an `editor`, and a `manager`. The writer begins the workflow by drafting a paragraph. Then the editor refines the draft, and the manager reviews the final result. The manager has private instructions to be strict in its assessment.

The edit process is a dynamic loop, continuing until the manager approves the paragraph. To accomplish this, we eagerly run the `approval_task` at the end of each iteration to see if approval was granted. Because of the dependency structure, running the approval task also triggers the edit task.

### Recap
<Check>
**What we learned**

- Agents are AIs that complete tasks and can be specialized with different capabilities, tools, instructions, and even LLM models
- Agents can be assigned to tasks
- Flows can involve dynamic control flow like loops, based on eager task result
- Flows allow multiple agents to collaborate on a larger objective with shared history
</Check>

---

## What's Next? 

Congratulations, you've completed the ControlFlow quickstart! You've learned how to:

- Create tasks and equip them with tools
- Define specialized agents and assign them to tasks
- Orchestrate multiple agents in a flow to collaborate on a larger objective

To continue learning, please explore the full [ControlFlow tutorial](/tutorial).
## ./docs/reference/task-class.mdx
# Task Reference

This document serves as a comprehensive reference for the `Task` class in ControlFlow. It provides detailed information about the properties and usage of tasks.

## `Task` Class

The `Task` class is used to define a task in ControlFlow. It provides a flexible way to specify various properties and requirements for a task.

### Properties

<ParamField path="objective" type="str" required>
The `objective` property is a brief description of the task's goal or desired outcome. It should clearly and concisely convey the purpose of the task, helping both developers and AI agents understand what the task aims to achieve.

A well-defined objective is crucial for ensuring that tasks are focused and aligned with the overall workflow. It serves as a guiding statement for the AI agents working on the task, helping them stay on track and deliver relevant results.
</ParamField>

<ParamField path="instructions" type="str">
The `instructions` property provides detailed guidelines or steps for completing the task. It offers a way to give more context and direction to the AI agents, beyond what is conveyed in the objective.

Instructions can include specific requirements, constraints, or preferences that should be considered when working on the task. They can also break down the task into smaller steps or provide examples to clarify the expected outcome.

By offering clear and thorough instructions, you can guide the AI agents towards delivering more accurate and relevant results. Well-crafted instructions help ensure consistency and quality in task execution.
</ParamField>

<ParamField path="agents" type="list[Agent]">
The `agents` property specifies the AI agents assigned to work on the task. By assigning specific agents to a task, you can leverage their specialized skills or knowledge to achieve better results.

When defining a task, you can provide a list of `Agent` instances that should be responsible for executing the task. These agents will be given priority when the task is run, and the task will be assigned to one of them based on their availability and suitability.

If no agents are explicitly assigned to a task, ControlFlow will use the default agents defined in the flow or fall back to a global default agent.
</ParamField>

<ParamField path="context" type="dict">
The `context` property allows you to provide additional information or data that is required for the task's execution. It serves as a way to pass inputs, dependencies, or any other relevant context to the task.

The context is defined as a dictionary, where each key-value pair represents a piece of contextual information. The keys are typically strings that describe the nature of the data, and the values can be of any valid Python type, such as strings, numbers, lists, or even other tasks.

When a task is executed, the AI agents have access to the context and can use it to inform their decision-making process or to supplement their knowledge. The context can contain data from previous tasks, external sources, or any other relevant information that is needed to complete the task effectively.
</ParamField>

<ParamField path="result_type" type="type">
The `result_type` property specifies the expected type of the task's result. It allows you to define the structure and format of the data that the task should return upon completion.

By specifying a result type, you provide a contract for the task's output, making it clear to both the AI agents and the developers what kind of data to expect. This helps ensure consistency and enables seamless integration of the task's result into the broader workflow.

The `result_type` can be any valid Python type, such as `str`, `int`, `list`, `dict`, or even custom classes. You can also use type annotations to define more complex types, such as `list[str]` for a list of strings or `dict[str, int]` for a dictionary mapping strings to integers.
</ParamField>

<ParamField path="tools" type="list[Callable]">
The `tools` property allows you to provide a list of Python functions that the AI agents can use to complete the task. These tools serve as additional capabilities or utilities that the agents can leverage during task execution.

Tools can be any valid Python functions that perform specific actions, computations, or transformations. They can range from simple utility functions to complex algorithms or external API calls.

By providing relevant tools to a task, you empower the AI agents to tackle more complex problems and enhance their problem-solving abilities. The agents can invoke these tools as needed during the task execution process.
</ParamField>

<ParamField path="user_access" type="bool">
The `user_access` property indicates whether the task requires human interaction or input during its execution. It is a boolean flag that, when set to `True`, signifies that the task involves user communication.

When a task has `user_access` set to `True`, the AI agents are provided with a special `talk_to_human` tool that enables them to send messages to the user and receive responses. This allows for a conversational flow between the agents and the user, facilitating the exchange of information required for the task.

It's important to note that tasks with `user_access` enabled should be designed with care, considering the user experience and the clarity of the communication. The AI agents should provide clear instructions and prompts to guide the user in providing the necessary input.
</ParamField>

<ParamField path="depends_on" type="list[Task]">
The `depends_on` property allows you to specify other tasks that the current task depends on. It establishes a dependency relationship between tasks, indicating that the current task cannot be started until the specified dependencies are completed.

Dependencies are defined as a list of `Task` instances. When a task is executed, ControlFlow ensures that all its dependencies are resolved before allowing the task to begin.

By specifying dependencies, you can create a structured workflow where tasks are executed in a specific order based on their dependencies. This helps ensure that tasks have access to the necessary data or results from previous tasks before they start.
</ParamField>

<ParamField path="parent" type="Optional[Task]">
The `parent` property allows you to specify a parent task for the current task. It establishes a hierarchical relationship between tasks, where the parent task is responsible for managing and coordinating the execution of its child tasks.

By organizing tasks into subtasks, you can break down complex tasks into smaller, more manageable units of work. This promotes modularity, reusability, and easier maintenance of the task hierarchy. When a task is created with a parent task, it automatically becomes a subtask of the parent task. The parent task is considered complete only when all its subtasks have been successfully completed.
</ParamField>
## ./docs/reference/task-decorator.mdx
# `@task` Decorator

The `@task` decorator is used to define a task using a Python function. It provides a convenient way to create tasks by leveraging the function's properties and automatically inferring various task attributes.

## Parameters

<ParamField path="objective" type="str">
The `objective` parameter allows you to specify the objective of the task. It should be a brief description of the task's goal or desired outcome.

If not provided, the objective will be inferred from the function name. It is recommended to use descriptive and meaningful function names that clearly convey the purpose of the task.

When using the `@task` decorator, the objective can be explicitly specified to provide more clarity or to override the inferred objective from the function name.
</ParamField>

<ParamField path="instructions" type="str">
The `instructions` parameter allows you to provide detailed instructions or guidelines for completing the task. It serves as a way to give more context and direction to the AI agents working on the task.

If not provided, the instructions will be inferred from the function's docstring. It is recommended to use descriptive and clear docstrings that explain the steps or requirements for completing the task.

When using the `@task` decorator, the instructions can be explicitly specified to provide more comprehensive guidance or to override the inferred instructions from the docstring.
</ParamField>

<ParamField path="agents" type="list[Agent]">
The `agents` parameter allows you to specify the AI agents that should work on the task. It accepts a list of `Agent` instances, representing the agents assigned to the task.

By specifying agents using the `@task` decorator, you can leverage their specialized skills or knowledge to achieve better results. The assigned agents will be given priority when the task is run, and the task will be assigned to one of them based on their availability and suitability.

If no agents are explicitly assigned to a task, ControlFlow will use the default agents defined in the flow or fall back to a global default agent.
</ParamField>

<ParamField path="tools" type="list[Callable]">
The `tools` parameter allows you to provide a list of Python functions that the AI agents can use to complete the task. These tools serve as additional capabilities or utilities that the agents can leverage during task execution.

Tools can be any valid Python functions that perform specific actions, computations, or transformations. They can range from simple utility functions to complex algorithms or external API calls.

By specifying tools using the `@task` decorator, you empower the AI agents to tackle more complex problems and enhance their problem-solving abilities. The agents can invoke these tools as needed during the task execution process.
</ParamField>

<ParamField path="user_access" type="bool">
The `user_access` parameter indicates whether the task requires human interaction or input during its execution. It is a boolean flag that, when set to `True`, signifies that the task involves user communication.

When a task has `user_access` set to `True`, the AI agents are provided with a special `talk_to_human` tool that enables them to send messages to the user and receive responses. This allows for a conversational flow between the agents and the user, facilitating the exchange of information required for the task.

By default, `user_access` is set to `False`. It can be explicitly set to `True` using the `@task` decorator when the task requires human interaction.
</ParamField>

<ParamField path="lazy" type="bool">
The `lazy` parameter determines whether the task should be executed eagerly or lazily. It is a boolean flag that controls the execution behavior of the task.

The default `lazy` behavior is determined by the global `eager_mode` setting in ControlFlow. Eager mode is enabled by default, which means that tasks are executed immediately. The `lazy` parameter allows you to override this behavior for a specific task.

When `lazy` is set to `True`, the task is not executed immediately. Instead, a `Task` instance is returned, representing the deferred execution of the task. The task can be run later using the `run()` or `run_once()` methods.

When `lazy` is set to `False` (default), the task is executed immediately when the decorated function is called. Setting `lazy=False` ensures the task is executed eagerly, even if the global `eager_mode` is disabled.
</ParamField>

## Inferred Properties

When using the `@task` decorator, several task properties are automatically inferred from the decorated function:

<ParamField path="objective" type="str">
The objective of the task is inferred from the function name. It is assumed that the function name provides a clear and concise description of the task's goal or desired outcome.

For example, if the decorated function is named `generate_summary()`, the inferred objective would be "Generate summary".
</ParamField>

<ParamField path="instructions" type="str">
The instructions for the task are inferred from the function's docstring. The docstring is expected to provide detailed guidelines or steps for completing the task.

For example:
```python
@task
def analyze_sentiment(text: str) -> str:
    """
    Analyze the sentiment of the given text.

    Steps:
    1. Preprocess the text by removing punctuation and converting to lowercase.
    2. Tokenize the preprocessed text into individual words.
    3. Perform sentiment analysis using a pre-trained model.
    4. Return the sentiment label (e.g., positive, negative, neutral).
    """
    pass
```

In this case, the inferred instructions would be the content of the docstring.
</ParamField>

<ParamField path="context" type="dict">
The context for the task is inferred from the function's arguments. Each argument of the decorated function is considered a piece of contextual information required for the task's execution.

For example:
```python
@task
def greet_user(name: str, age: int) -> str:
    pass
```

In this case, the inferred context would be a dictionary containing the `name` and `age` arguments:
```python
{
    "name": <value of name>,
    "age": <value of age>
}
```
</ParamField>

<ParamField path="result_type" type="type">
The expected result type of the task is inferred from the function's return annotation. The return annotation specifies the type of data that the task should return upon completion.

For example:
```python
@task
def generate_greeting(name: str) -> str:
    pass
```

In this case, the inferred result type would be `str`, indicating that the task is expected to return a string value.
</ParamField>

By leveraging the inferred properties, the `@task` decorator simplifies the process of creating tasks and reduces the need for explicit configuration. The decorator automatically extracts relevant information from the function definition, making task creation more intuitive and concise.
## ./docs/installation.mdx
---
title: Installation & Setup
description: Learn how to install ControlFlow and configure your API keys.
---

<Card title="ControlFlow is under active development" icon="excavator" iconType="duotone">
Pin to a specific version if you want to avoid breaking changes.
However, we recommend frequent updates to get new features and bug fixes.
</Card>

import Installation from '/snippets/installation.mdx';

<Installation />

<Tip>
ControlFlow supports many other LLM providers as well.
See the [LLM documentation](/guides/llms) for more information.
</Tip>

## Next steps

Dive right into the [quickstart](/quickstart), or read the [tutorial](/tutorial) for a step-by-step guide to creating your first ControlFlow workflow.


