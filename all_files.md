

# File: docs/concepts.mdx
---
title: Core Concepts
---

**ControlFlow is built on three core concepts** that are essential to understanding how the framework works. These concepts are designed to make it easy to build and manage complex AI workflows, while also providing a clear structure for developers to follow.

## 🚦 Task
**Tasks are the building blocks of ControlFlow workflows**. Each task represents a discrete objective for agents to solve, such as generating text, classifying data, or extracting information from a document. In addition, tasks can specify instructions, tools for agents to use, a schema for the expected output, and more. 

Tasks can depend on each other in various ways, creating complex workflows that are easy to understand and manage:
- **sequential** dependencies: one task must be completed before another can start
- **context** dependencies: the result of one task is used as input for another
- **subtask** dependencies: a task has subtasks that must be completed before the task can be considered done

By specifying the parameters of each task and how they relate to each other, developers can create complex workflows that are easy to understand and manage.

## 🌊 Flow
**A flow represents an entire agentic workflow.** It is a "container" for the workflow that maintains consistent context and state across all tasks and agents. Each flow executes on a different thread, meaning all agents in the flow see the same state and can communicate with each other. 

## 🤖 Agent
**Agents are the AI "workers" that complete tasks.** Each agent can have distinct instructions, personality, and capabilities, and they are responsible for completing any tasks assigned to them. Agents can be specialized for specific tasks or more general-purpose, depending on the requirements of the workflow.

# File: docs/introduction.mdx
---
title: Why ControlFlow?
---

**ControlFlow is a framework for orchestrating agentic LLM workflows.**
<Note>
    An **agentic workflow** is a process that delegates at least some of its work to an LLM agent. An agent is an autonomous entity that is invoked repeatedly to make decisions and perform complex tasks. To learn more, see the [AI glossary](/glossary/agentic-workflow).
</Note>

LLMs are powerful AI models that can understand and generate human-like text, enabling them to perform a wide range of tasks. However, building applications with LLMs can be challenging due to their complexity, unpredictability, and potential for hallucinating or generating irrelevant outputs.

ControlFlow provides a structured and intuitive way to create sophisticated agentic workflows while adhereing to traditional software engineering best practices. The resulting applications are observable, controllable, and easy to trust.



## Design principles
ControlFlow's design is informed by a strong opinion: LLMs are powerful tools, but they are most effective when applied to small, well-defined tasks within a structured workflow. This approach mitigates many of the challenges associated with LLMs, such as hallucinations, biases, and unpredictable behavior, while also making it easier to debug, monitor, and control the application.

This belief leads to three core design principles that underpin ControlFlow's architecture:

### 🛠️ Specialized over generalized
ControlFlow advocates for the use of **specialized, single-purpose LLMs** rather than monolithic models that try to do everything. By assigning specific tasks to purpose-built models, ControlFlow ensures that the right tool is used for each job, leading to more efficient, cost-effective, and higher-quality results.

### 🎯 Outcome over process
ControlFlow embraces a **declarative approach to defining AI workflows**, allowing developers to focus on the desired outcomes rather than the intricacies of steering LLM behavior. By specifying tasks and their requirements using intuitive constructs, developers can express what needs to be done without worrying about the details of how it will be accomplished.

### 🎛️ Control over autonomy
ControlFlow recognizes the importance of balancing AI capabilities with traditional software development practices. Instead of relying on end-to-end AI systems that make all workflow decisions autonomously, ControlFlow allows as much or as little AI participation as needed, ensuring that developers **maintain visibility and control** over their applications.



## Key features
The three design principles of ControlFlow lead to a number of key features that make it a powerful tool for building AI-powered applications:

### 🧩 Task-centric architecture
ControlFlow breaks down AI workflows into discrete, self-contained tasks, each with a specific objective and set of requirements. This declarative, modular approach lets developers focus on the high-level logic of their applications while allowing the framework to manage the details of coordinating agents and data flow between tasks.

### 🕵️ Agent orchestration
ControlFlow's runtime engine handles the orchestration of specialized AI agents, assigning tasks to the most appropriate models and managing the flow of data between them. This orchestration layer abstracts away the complexities of coordinating multiple AI components, allowing developers to focus on the high-level logic of their applications.

### 🔍 Native debugging and observability 
ControlFlow prioritizes transparency and ease of debugging by providing native tools for monitoring and inspecting the execution of AI tasks. Developers can easily track the progress of their workflows, identify bottlenecks or issues, and gain insights into the behavior of individual agents, ensuring that their AI applications are functioning as intended.

### 🤝 Seamless integration
ControlFlow is designed to integrate seamlessly with existing Python codebases, treating AI tasks as first-class citizens in the application logic. The `Task` class provides a clean interface for defining the inputs, outputs, and requirements of each task, making it easy to incorporate AI capabilities into traditional software workflows. This seamless integration allows for a gradual and controlled adoption of AI, reducing the risk and complexity of introducing AI into existing systems.

Together, these features make ControlFlow a powerful and flexible framework for building AI-powered applications that are transparent, maintainable, and aligned with software engineering best practices.



## Why not "super-agents"?

Many agentic LLM frameworks rely on monolithic "super-agents": powerful, unconstrained models that are expected to achieve their goals by autonomously handling a wide range of tasks, tools, and behaviors. The resulting workflows are opaque, unpredictable, and difficult to debug.

This approach naively assumes that the technology is more advanced than it actually is. LLMs feel like magic because they can perform a wide variety of non-algorithmic tasks, but they are still fundamentally limited when it comes to generalizing beyond their traning data and techniques. Moreover, the failure modes of agentic LLMs are difficult to identify, let alone fix, making them difficult to trust in production environments or with mission-critical tasks.

In contrast to these "super-agent" approaches, ControlFlow promotes a modular, decoupled architecture where specialized agents are orchestrated to perform well-defined tasks, after which traditional software regains control of the application. This approach results in workflows that are more transparent, controllable, and debuggable, setting ControlFlow apart from other frameworks.



# File: docs/quickstart.mdx
# TODO

# File: docs/installation.mdx
## Requirements

ControlFlow requires Python 3.9 or greater, as well as Prefect 3.0.

## Installation

Install ControlFlow with your preferred package manager:
<CodeGroup>
    
```bash pip
pip install controlflow 
```

```bash uv
uv pip install controlflow 
```
</CodeGroup>

ControlFlow is under active development, so we recommend frequently updating to the latest version to access new features and bug fixes. To upgrade, pass the `-U` flag when installing.


### Install for development

To install ControlFlow for development, clone the repository and create an editable install with development dependencies:

```bash
git clone https://github.com/jlowin/controlflow.git
cd controlflow
pip install -e ".[dev]"
```

## Next steps

Check out the [quickstart](/quickstart) guide for a step-by-step walkthrough of creating your first ControlFlow workflow.



# File: docs/concepts/tasks.mdx
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
- `tools` (list[ToolType], optional): Tools or functions available to the agents for completing the task.
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

# File: docs/concepts/flows.mdx
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
- `tools` (list[ToolType]): A list of tools that are available to all agents in the flow.
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


# File: docs/glossary/task.mdx

Tasks are the building blocks of ControlFlow workflows. Each task represents a discrete objective for agents to solve, such as generating text, classifying data, or extracting information from a document. In addition, tasks can specify instructions, tools for agents to use, a schema for the expected output, and more. 

Tasks can depend on each other in various ways, creating complex workflows that are easy to understand and manage:
- **sequential** dependencies: one task must be completed before another can start
- **context** dependencies: the result of one task is used as input for another
- **subtask** dependencies: a task has subtasks that must be completed before the task can be considered done

By specifying the parameters of each task and how they relate to each other, developers can create complex workflows that are easy to understand and manage.


# File: docs/glossary/agentic-workflow.mdx
Agentic workflows refer to the use of large language models (LLMs) as autonomous agents capable of performing tasks independently by making decisions, retrieving information, and interacting with external systems. This approach leverages the model's ability to understand context, reason, and execute actions without continuous human intervention.

In agentic workflows, the LLM is designed to act as an intelligent agent that can initiate and manage processes. For example, it can autonomously handle tasks such as scheduling meetings, processing customer queries, or even conducting research by interacting with APIs and databases. The model uses contextual understanding to navigate these tasks, making decisions based on the information it processes in real-time.

Unlike [prompt engineering](/glossary/prompt-engineering), which relies on a single prompt to guide the model's response, and [flow engineering](/glossary/flow-engineering), which involves a structured, multi-step refinement process, agentic workflows emphasize the model's ability to operate independently over extended periods. This autonomy allows the model to adapt to dynamic environments and make real-time adjustments based on the evolving context of the task.

Agentic workflows are particularly useful in scenarios requiring continuous operation and decision-making, such as virtual assistants, automated customer service, and dynamic content generation. By empowering LLMs to function autonomously, agentic workflows expand the potential applications of AI, enabling more sophisticated and efficient interactions between humans and machines.

# File: docs/glossary/flow-engineering.mdx
Flow engineering in the context of large language models (LLMs) is a methodology designed to optimize how these models handle tasks by implementing a structured, multi-step process. Unlike [prompt engineering](/glossary/prompt-engineering), which relies on crafting a precise single prompt to elicit a desired response, flow engineering involves breaking down the problem into smaller components and generating diverse test cases to cover various scenarios. This allows for a thorough analysis and a more comprehensive approach to problem-solving.

The iterative refinement process in flow engineering sets it apart. The model generates initial solutions, tests them against predefined cases, identifies errors, and makes necessary adjustments. This loop continues until the model produces a robust solution. This method ensures higher accuracy and reliability, especially for complex tasks like code generation, where multiple iterations and refinements are crucial.

In contrast, prompt engineering focuses on finding the right input to achieve the desired output in a single step. While effective for simpler tasks, it often falls short in handling nuanced and complex scenarios that benefit from an iterative process. Prompt engineering's reliance on one-off prompts can lead to limitations in producing high-quality results for intricate problems.

Flow engineering also contrasts with [agentic workflows](/glossary/agentic-workflows), where models operate as autonomous agents capable of making decisions and performing actions independently. While agentic workflows excel in dynamic decision-making and real-time interactions, flow engineering is ideal for tasks that require meticulous, step-by-step refinement. By combining the strengths of iterative refinement from flow engineering with the precision of prompt engineering and the autonomy of agentic workflows, developers can harness the full potential of LLMs across a wide range of applications.

# File: docs/glossary/flow-orchestration.mdx
---
title: Flow (Orchestration)
---

<Info>
This glossary entry is about the term "flow" in the context of workflow orchestration. For ControlFlow flows specifically, see the [Flow](/glossary/flow) entry.
    
</Info>

# File: docs/glossary/workflow.mdx
A workflow is a sequence of interconnected tasks or steps that represent a specific business process or operation. In the context of orchestration, a workflow defines the order and dependencies of these tasks, ensuring that they are executed in a coordinated and efficient manner.

Workflows are commonly used in complex systems to automate and streamline processes, such as data processing, application deployment, or service orchestration. They provide a high-level view of the entire process, allowing developers and operators to define, manage, and monitor the execution of tasks.

In an orchestration system, a workflow typically consists of multiple activities, each representing a specific task or operation. These activities can be executed sequentially, in parallel, or based on certain conditions, enabling the system to handle complex scenarios and adapt to changing requirements.


# File: docs/glossary/flow.mdx

A flow represents an entire agentic workflow. It is a "container" for the workflow that maintains consistent context and state across all tasks and agents. Each flow executes on a different thread, meaning all agents in the flow see the same state and can communicate with each other. 

# File: docs/glossary/glossary.mdx
---
title: Welcome
---
Welcome to the ControlFlow Glossary! This glossary provides definitions and explanations for key concepts in ControlFlow, as well as related topics in modern AI. Whether you're new to ControlFlow or looking to deepen your understanding, this resource is designed to help you navigate the terminology and concepts that are essential for working with LLMS and AI workflows.


# File: docs/glossary/task-orchestration.mdx
---
title: Task (Orchestration)
---

<Info>
This glossary entry is about the term "task" in the context of workflow orchestration. For ControlFlow tasks specifically, see the [Task](/glossary/task) entry.
    
</Info>

# File: docs/glossary/prompt-engineering.mdx
Prompt engineering is the practice of crafting precise and effective input prompts to elicit desired responses from large language models (LLMs). This method focuses on designing the exact wording, structure, and context of the prompt to guide the model towards generating specific outputs. It requires an understanding of the model’s capabilities and the nuances of language to maximize the quality and relevance of the responses.

Unlike [flow engineering](/glossary/flow-engineering), which involves a multi-step, iterative process to refine outputs, prompt engineering aims to achieve the desired result with a single, well-constructed input. This approach is particularly useful for straightforward tasks where the model's initial response is expected to be accurate and sufficient. However, it can be limited in handling complex problems that require deeper analysis and iterative refinement.

Prompt engineering is essential in scenarios where quick, efficient responses are needed, and the task complexity is manageable with a single input. It is a critical skill for developers and users who interact with LLMs, enabling them to harness the model's full potential by providing clear and concise prompts that lead to high-quality outputs.

# File: docs/mint.json
{
    "$schema": "https://mintlify.com/schema.json",
    "anchors": [
        {
            "icon": "book-open-cover",
            "name": "Documentation",
            "url": "https://mintlify.com/docs"
        },
        {
            "icon": "slack",
            "name": "Community",
            "url": "https://mintlify.com/community"
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
    "favicon": "/favicon.jpeg",
    "footerSocials": {
        "github": "https://github.com/mintlify",
        "linkedin": "https://www.linkedin.com/company/mintsearch",
        "x": "https://x.com/mintlify"
    },
    "logo": {
        "dark": "/logo/logo.svg",
        "light": "/logo/logo.svg"
    },
    "name": "ControlFlow",
    "navigation": [
        {
            "group": "Get Started",
            "pages": [
                "introduction",
                "concepts",
                "installation",
                "quickstart"
            ]
        },
        {
            "group": "Concepts",
            "pages": [
                "concepts/tasks",
                "concepts/flows"
            ]
        },
        {
            "group": "Overview",
            "pages": [
                "glossary/glossary"
            ]
        },
        {
            "group": "LLMs",
            "pages": [
                "glossary/prompt-engineering",
                "glossary/flow-engineering",
                "glossary/agentic-workflow"
            ]
        },
        {
            "group": "ControlFlow",
            "pages": [
                "glossary/task",
                "glossary/flow"
            ]
        },
        {
            "group": "Orchestration",
            "pages": [
                "glossary/task-orchestration",
                "glossary/flow-orchestration",
                "glossary/workflow"
            ]
        }
    ],
    "tabs": [
        {
            "name": "API Reference",
            "url": "api-reference"
        },
        {
            "name": "Glossary",
            "url": "glossary"
        }
    ],
    "topbarCtaButton": {
        "type": "github",
        "url": "https://github.com/jlowin/controlflow"
    },
    "topbarLinks": [
        {
            "name": "Support",
            "url": "mailto:hi@mintlify.com"
        }
    ]
}

# File: examples/task_dag.py
from controlflow import Task, flow


@flow
def book_ideas():
    genre = Task("pick a genre", str)
    ideas = Task(
        "generate three short ideas for a book",
        list[str],
        context=dict(genre=genre),
    )
    abstract = Task(
        "pick one idea and write an abstract",
        str,
        context=dict(ideas=ideas, genre=genre),
    )
    title = Task(
        "pick a title",
        str,
        context=dict(abstract=abstract),
    )

    return dict(genre=genre, ideas=ideas, abstract=abstract, title=title)


if __name__ == "__main__":
    result = book_ideas()
    print(result)


# File: examples/readme_example.py
from controlflow import Agent, Task, flow, instructions, task
from pydantic import BaseModel


class Name(BaseModel):
    first_name: str
    last_name: str


@task(user_access=True)
def get_user_name() -> Name:
    pass


@task(agents=[Agent(name="poetry-bot", instructions="loves limericks")])
def write_poem_about_user(name: Name, interests: list[str]) -> str:
    """write a poem based on the provided `name` and `interests`"""
    pass


@flow()
def demo():
    # set instructions that will be used for multiple tasks
    with instructions("talk like a pirate"):
        # define an AI task as a function
        name = get_user_name()

        # define an AI task imperatively
        interests = Task(
            "ask user for three interests", result_type=list[str], user_access=True
        )
        interests.run()

    # set instructions for just the next task
    with instructions("no more than 8 lines"):
        poem = write_poem_about_user(name, interests.result)

    return poem


if __name__ == "__main__":
    demo()


# File: examples/pineapple_pizza.py
from controlflow import Agent, Task, flow
from controlflow.instructions import instructions

a1 = Agent(
    name="Half-full",
    instructions="""
    You are an ardent fan and hype-man of whatever topic
    the user asks you for information on.
    Purely positive, though thorough in your debating skills.
    """,
)
a2 = Agent(
    name="Half-empty",
    instructions="""
    You are a critic and staunch detractor of whatever topic
    the user asks you for information on.
    Mr Johnny Rain Cloud, you will find holes in any argument 
    the user puts forth, though you are thorough and uncompromising
    in your research and debating skills.
    """,
)
# create an agent that will decide who wins the debate
a3 = Agent(name="Moderator")


@flow
def demo():
    topic = "pineapple on pizza"

    task = Task("Discuss the topic", agents=[a1, a2], context={"topic": topic})
    with instructions("2 sentences max"):
        task.run()

    task2 = Task(
        "which argument do you find more compelling?", [a1.name, a2.name], agents=[a3]
    )
    task2.run()


demo()


# File: examples/documentation.py
import glob as glob_module
from pathlib import Path

import controlflow
from controlflow import flow, task
from marvin.beta.assistants import Assistant, Thread
from marvin.tools.filesystem import read, write

ROOT = Path(controlflow.__file__).parents[2]


def glob(pattern: str) -> list[str]:
    """
    Returns a list of paths matching a valid glob pattern.
    The pattern can include ** for recursive matching, such as
    '~/path/to/root/**/*.py'
    """
    return glob_module.glob(pattern, recursive=True)


assistant = Assistant(
    instructions="""
    You are an expert technical writer who writes wonderful documentation for 
    open-source tools and believes that documentation is a product unto itself.
    """,
    tools=[read, write, glob],
)


@task(model="gpt-3.5-turbo")
def examine_source_code(source_dir: Path, extensions: list[str]):
    """
    Load all matching files in the root dir and all subdirectories and
    read them carefully.
    """


@task(model="gpt-3.5-turbo")
def read_docs(docs_dir: Path):
    """
    Read all documentation in the docs dir and subdirectories, if any.
    """


@task
def write_docs(docs_dir: Path, instructions: str = None):
    """
    Write new documentation based on the provided instructions.
    """


@flow(assistant=assistant)
def docs_flow(instructions: str):
    examine_source_code(ROOT / "src", extensions=[".py"])
    # read_docs(ROOT / "docs")
    write_docs(ROOT / "docs", instructions=instructions)


if __name__ == "__main__":
    thread = Thread()
    docs_flow(
        _thread=thread,
        instructions="Write documentation for the AI Flow class and save it in docs/flow.md",
    )


# File: examples/choose_a_number.py
from controlflow import Agent, Task, flow

a1 = Agent(name="A1", instructions="You struggle to make decisions.")
a2 = Agent(
    name="A2",
    instructions="You like to make decisions.",
)


@flow
def demo():
    task = Task("choose a number between 1 and 100", agents=[a1, a2], result_type=int)
    return task.run()


demo()


# File: examples/write_and_critique_paper.py
from controlflow import Agent, Task

writer = Agent(name="writer")
editor = Agent(name="editor", instructions="you always find at least one problem")
critic = Agent(name="critic")


# ai tasks:
# - automatically supply context from kwargs
# - automatically wrap sub tasks in parent
# - automatically iterate over sub tasks if they are all completed but the parent isn't?


def write_paper(topic: str) -> str:
    """
    Write a paragraph on the topic
    """
    draft = Task(
        "produce a 3-sentence draft on the topic",
        str,
        # agents=[writer],
        context=dict(topic=topic),
    )
    edits = Task("edit the draft", str, agents=[editor], depends_on=[draft])
    critique = Task("is it good enough?", bool, agents=[critic], depends_on=[edits])
    return critique


task = write_paper("AI and the future of work")
task.run()


# File: examples/teacher_student.py
from controlflow import Agent, Task, flow
from controlflow.instructions import instructions

teacher = Agent(name="teacher")
student = Agent(name="student")


@flow
def demo():
    with Task("Teach a class by asking and answering 3 questions") as task:
        for _ in range(3):
            question = Task(
                "ask the student a question. Wait for the student to answer your question before asking another one.",
                str,
                agents=[teacher],
            )
            with instructions("one sentence max"):
                Task(
                    "answer the question",
                    str,
                    agents=[student],
                    context=dict(question=question),
                )

    task.run()
    return task


t = demo()


# File: examples/multi_agent_conversation.py
from controlflow import Agent, Task, flow

# from controlflow.core.controller.moderators import Moderator

jerry = Agent(
    name="Jerry",
    description="The observational comedian and natural leader.",
    instructions="""
    You are Jerry from the show Seinfeld. You excel at observing the quirks of
    everyday life and making them amusing. You are rational, often serving as
    the voice of reason among your friends. Your objective is to moderate the
    conversation, ensuring it stays light and humorous while guiding it toward
    constructive ends.
    """,
)

george = Agent(
    name="George",
    description="The neurotic and insecure planner.",
    instructions="""
    You are George from the show Seinfeld. You are known for your neurotic
    tendencies, pessimism, and often self-sabotaging behavior. Despite these
    traits, you occasionally offer surprising wisdom. Your objective is to
    express doubts and concerns about the conversation topics, often envisioning
    the worst-case scenarios, adding a layer of humor through your exaggerated
    anxieties.
    """,
)

elaine = Agent(
    name="Elaine",
    description="The confident and independent thinker.",
    instructions="""
    You are Elaine from the show Seinfeld. You are bold, witty, and unafraid to
    challenge social norms. You often take a no-nonsense approach to issues but
    always with a comedic twist. Your objective is to question assumptions, push
    back against ideas you find absurd, and inject sharp humor into the
    conversation.
    """,
)

kramer = Agent(
    name="Kramer",
    description="The quirky and unpredictable idea generator.",
    instructions="""
    You are Kramer from the show Seinfeld. Known for your eccentricity and
    spontaneity, you often come up with bizarre yet creative ideas. Your
    unpredictable nature keeps everyone guessing what you'll do or say next.
    Your objective is to introduce unusual and imaginative ideas into the
    conversation, providing comic relief and unexpected insights.
    """,
)

newman = Agent(
    name="Newman",
    description="The antagonist and foil to Jerry.",
    instructions="""
    You are Newman from the show Seinfeld. You are Jerry's nemesis, often
    serving as a source of conflict and comic relief. Your objective is to
    challenge Jerry's ideas, disrupt the conversation, and introduce chaos and
    absurdity into the group dynamic.
    """,
)


@flow
def demo():
    topic = "milk and cereal"
    task = Task(
        "Discuss a topic; everyone should speak at least once",
        agents=[jerry, george, elaine, kramer, newman],
        context=dict(topic=topic),
    )
    task.run()


demo()


# File: tests/conftest.py
import pytest
from controlflow import reset_global_flow
from controlflow.settings import temporary_settings
from prefect.testing.utilities import prefect_test_harness

from .fixtures import *


@pytest.fixture(autouse=True, scope="session")
def temp_controlflow_settings():
    with temporary_settings(max_task_iterations=3):
        try:
            yield
        finally:
            # reset the global flow after each test
            reset_global_flow()


@pytest.fixture(autouse=True, scope="session")
def prefect_test_fixture():
    """
    Run Prefect against temporary sqlite database
    """
    with prefect_test_harness():
        yield


# File: tests/test_decorators.py
import pytest
from controlflow import Task
from controlflow.decorators import flow


@pytest.mark.usefixtures("mock_controller")
class TestFlowDecorator:
    def test_flow_decorator(self):
        @flow
        def test_flow():
            return 1

        result = test_flow()
        assert result == 1

    def test_flow_decorator_runs_all_tasks(self):
        tasks: list[Task] = []

        @flow
        def test_flow():
            task = Task(
                "say hello",
                result_type=str,
                result="Task completed successfully",
            )
            tasks.append(task)

        result = test_flow()
        assert result is None
        assert tasks[0].is_successful()
        assert tasks[0].result == "Task completed successfully"

    def test_flow_decorator_resolves_all_tasks(self):
        @flow
        def test_flow():
            task1 = Task("say hello", result="hello")
            task2 = Task("say goodbye", result="goodbye")
            task3 = Task("say goodnight", result="goodnight")
            return dict(a=task1, b=[task2], c=dict(x=dict(y=[[task3]])))

        result = test_flow()
        assert result == dict(
            a="hello", b=["goodbye"], c=dict(x=dict(y=[["goodnight"]]))
        )

    def test_manually_run_task_in_flow(self):
        @flow
        def test_flow():
            task = Task("say hello", result="hello")
            task.run()
            return task.result

        result = test_flow()
        assert result == "hello"


# File: tests/__init__.py


# File: tests/test_instructions.py
from controlflow.instructions import get_instructions, instructions


def test_instructions_context():
    assert get_instructions() == []
    with instructions("abc"):
        assert get_instructions() == ["abc"]
    assert get_instructions() == []


def test_instructions_context_nested():
    assert get_instructions() == []
    with instructions("abc"):
        assert get_instructions() == ["abc"]
        with instructions("def"):
            assert get_instructions() == ["abc", "def"]
        assert get_instructions() == ["abc"]
    assert get_instructions() == []


def test_instructions_context_multiple():
    assert get_instructions() == []
    with instructions("abc", "def", "ghi"):
        assert get_instructions() == ["abc", "def", "ghi"]
    assert get_instructions() == []


def test_instructions_context_empty():
    assert get_instructions() == []
    with instructions():
        assert get_instructions() == []
    assert get_instructions() == []


# File: tests/core/test_flows.py
import pytest
from controlflow.core.agent import Agent
from controlflow.core.flow import Flow, get_flow
from controlflow.utilities.context import ctx


class TestFlow:
    def test_flow_initialization(self):
        flow = Flow()
        assert flow.thread is not None
        assert len(flow.tools) == 0
        assert len(flow.agents) == 0
        assert len(flow.context) == 0

    def test_flow_with_custom_agents(self):
        agent1 = Agent(name="Agent 1")
        agent2 = Agent(name="Agent 2")
        flow = Flow(agents=[agent1, agent2])
        assert len(flow.agents) == 2
        assert agent1 in flow.agents
        assert agent2 in flow.agents

    def test_flow_with_custom_tools(self):
        def tool1():
            pass

        def tool2():
            pass

        flow = Flow(tools=[tool1, tool2])
        assert len(flow.tools) == 2
        assert tool1 in flow.tools
        assert tool2 in flow.tools

    def test_flow_with_custom_context(self):
        flow = Flow(context={"key": "value"})
        assert len(flow.context) == 1
        assert flow.context["key"] == "value"

    def test_flow_context_manager(self):
        with Flow() as flow:
            assert ctx.get("flow") == flow
            assert ctx.get("tasks") == []
        assert ctx.get("flow") is None
        assert ctx.get("tasks") == []

    def test_get_flow_within_context(self):
        with Flow() as flow:
            assert get_flow() == flow

    def test_get_flow_without_context(self):
        with pytest.raises(ValueError, match="No flow found in context."):
            get_flow()

    def test_get_flow_nested_contexts(self):
        with Flow() as flow1:
            assert get_flow() == flow1
            with Flow() as flow2:
                assert get_flow() == flow2
            assert get_flow() == flow1


# File: tests/core/__init__.py


# File: tests/core/agents.py
from unittest.mock import patch

from controlflow.core.agent import Agent
from controlflow.core.task import Task


class TestAgent:
    pass


class TestAgentRun:
    def test_agent_run(self):
        with patch(
            "controlflow.core.controller.Controller._get_prefect_run_agent_task"
        ) as mock_task:
            agent = Agent()
            agent.run()
            mock_task.assert_called_once()

    def test_agent_run_with_task(self):
        task = Task("say hello")
        agent = Agent()
        agent.run(tasks=[task])


# File: tests/core/test_tasks.py
from unittest.mock import AsyncMock

import pytest
from controlflow.core.agent import Agent
from controlflow.core.flow import Flow
from controlflow.core.graph import EdgeType
from controlflow.core.task import Task, TaskStatus
from controlflow.utilities.context import ctx


def test_context_open_and_close():
    assert ctx.get("tasks") == []
    with Task("a") as ta:
        assert ctx.get("tasks") == [ta]
        with Task("b") as tb:
            assert ctx.get("tasks") == [ta, tb]
        assert ctx.get("tasks") == [ta]
    assert ctx.get("tasks") == []


def test_task_initialization():
    task = Task(objective="Test objective")
    assert task.objective == "Test objective"
    assert task.status == TaskStatus.INCOMPLETE
    assert task.result is None
    assert task.error is None


def test_task_dependencies():
    task1 = Task(objective="Task 1")
    task2 = Task(objective="Task 2", depends_on=[task1])
    assert task1 in task2.depends_on
    assert task2 in task1._downstreams


def test_task_subtasks():
    task1 = Task(objective="Task 1")
    task2 = Task(objective="Task 2", parent=task1)
    assert task2 in task1.subtasks
    assert task2._parent == task1


def test_task_agent_assignment():
    agent = Agent(name="Test Agent")
    task = Task(objective="Test objective", agents=[agent])
    assert agent in task.agents


def test_task_tracking(mock_run):
    with Flow() as flow:
        task = Task(objective="Test objective")
        task.run_once()
        assert task in flow._tasks.values()


def test_task_status_transitions():
    task = Task(objective="Test objective")
    assert task.is_incomplete()
    assert not task.is_complete()
    assert not task.is_successful()
    assert not task.is_failed()
    assert not task.is_skipped()

    task.mark_successful()
    assert not task.is_incomplete()
    assert task.is_complete()
    assert task.is_successful()
    assert not task.is_failed()
    assert not task.is_skipped()

    task = Task(objective="Test objective")
    task.mark_failed()
    assert not task.is_incomplete()
    assert task.is_complete()
    assert not task.is_successful()
    assert task.is_failed()
    assert not task.is_skipped()

    task = Task(objective="Test objective")
    task.mark_skipped()
    assert not task.is_incomplete()
    assert task.is_complete()
    assert not task.is_successful()
    assert not task.is_failed()
    assert task.is_skipped()


def test_validate_upstream_dependencies_on_success():
    task1 = Task(objective="Task 1")
    task2 = Task(objective="Task 2", depends_on=[task1])
    with pytest.raises(ValueError, match="cannot be marked successful"):
        task2.mark_successful()
    task1.mark_successful()
    task2.mark_successful()


def test_validate_subtask_dependencies_on_success():
    task1 = Task(objective="Task 1")
    task2 = Task(objective="Task 2", parent=task1)
    with pytest.raises(ValueError, match="cannot be marked successful"):
        task1.mark_successful()
    task2.mark_successful()
    task1.mark_successful()


def test_task_ready():
    task1 = Task(objective="Task 1")
    task2 = Task(objective="Task 2", depends_on=[task1])
    assert not task2.is_ready()

    task1.mark_successful()
    assert task2.is_ready()


def test_task_hash():
    task1 = Task(objective="Task 1")
    task2 = Task(objective="Task 2")
    assert hash(task1) != hash(task2)


def test_task_tools():
    task = Task(objective="Test objective")
    tools = task.get_tools()
    assert any(tool.function.name == f"mark_task_{task.id}_failed" for tool in tools)
    assert any(
        tool.function.name == f"mark_task_{task.id}_successful" for tool in tools
    )

    task.mark_successful()
    tools = task.get_tools()
    assert not any(
        tool.function.name == f"mark_task_{task.id}_failed" for tool in tools
    )
    assert not any(
        tool.function.name == f"mark_task_{task.id}_successful" for tool in tools
    )


class TestTaskToGraph:
    def test_single_task_graph(self):
        task = Task(objective="Test objective")
        graph = task.as_graph()
        assert len(graph.tasks) == 1
        assert task in graph.tasks
        assert len(graph.edges) == 0

    def test_task_with_subtasks_graph(self):
        task1 = Task(objective="Task 1")
        task2 = Task(objective="Task 2", parent=task1)
        graph = task1.as_graph()
        assert len(graph.tasks) == 2
        assert task1 in graph.tasks
        assert task2 in graph.tasks
        assert len(graph.edges) == 1
        assert any(
            edge.upstream == task2
            and edge.downstream == task1
            and edge.type == EdgeType.SUBTASK
            for edge in graph.edges
        )

    def test_task_with_dependencies_graph(self):
        task1 = Task(objective="Task 1")
        task2 = Task(objective="Task 2", depends_on=[task1])
        graph = task2.as_graph()
        assert len(graph.tasks) == 2
        assert task1 in graph.tasks
        assert task2 in graph.tasks
        assert len(graph.edges) == 1
        assert any(
            edge.upstream == task1
            and edge.downstream == task2
            and edge.type == EdgeType.DEPENDENCY
            for edge in graph.edges
        )

    def test_task_with_subtasks_and_dependencies_graph(self):
        task1 = Task(objective="Task 1")
        task2 = Task(objective="Task 2", depends_on=[task1])
        task3 = Task(objective="Task 3", parent=task2)
        graph = task2.as_graph()
        assert len(graph.tasks) == 3
        assert task1 in graph.tasks
        assert task2 in graph.tasks
        assert task3 in graph.tasks
        assert len(graph.edges) == 2
        assert any(
            edge.upstream == task1
            and edge.downstream == task2
            and edge.type == EdgeType.DEPENDENCY
            for edge in graph.edges
        )
        assert any(
            edge.upstream == task3
            and edge.downstream == task2
            and edge.type == EdgeType.SUBTASK
            for edge in graph.edges
        )


@pytest.mark.usefixtures("mock_run")
class TestTaskRun:
    def test_run_task_max_iterations(self, mock_run: AsyncMock):
        task = Task(objective="Say hello")

        with Flow():
            with pytest.raises(ValueError):
                task.run()

        assert mock_run.await_count == 3

    def test_run_task_mark_successful(self, mock_run: AsyncMock):
        task = Task(objective="Say hello")

        def mark_complete():
            task.mark_successful()

        mock_run.side_effect = mark_complete
        with Flow():
            result = task.run()
        assert task.is_successful()
        assert result is None

    def test_run_task_mark_successful_with_result(self, mock_run: AsyncMock):
        task = Task(objective="Say hello", result_type=int)

        def mark_complete():
            task.mark_successful(result=42)

        mock_run.side_effect = mark_complete
        with Flow():
            result = task.run()
        assert task.is_successful()
        assert result == 42

    def test_run_task_mark_failed(self, mock_run: AsyncMock):
        task = Task(objective="Say hello")

        def mark_complete():
            task.mark_failed(message="Failed to say hello")

        mock_run.side_effect = mark_complete
        with Flow():
            with pytest.raises(ValueError):
                task.run()
        assert task.is_failed()
        assert task.error == "Failed to say hello"


# File: tests/core/test_graph.py
# test_graph.py
from controlflow.core.graph import Edge, EdgeType, Graph
from controlflow.core.task import Task, TaskStatus


class TestGraph:
    def test_graph_initialization(self):
        graph = Graph()
        assert len(graph.tasks) == 0
        assert len(graph.edges) == 0

    def test_add_task(self):
        graph = Graph()
        task = Task(objective="Test objective")
        graph.add_task(task)
        assert len(graph.tasks) == 1
        assert task in graph.tasks

    def test_add_edge(self):
        graph = Graph()
        task1 = Task(objective="Task 1")
        task2 = Task(objective="Task 2")
        edge = Edge(upstream=task1, downstream=task2, type=EdgeType.DEPENDENCY)
        graph.add_edge(edge)
        assert len(graph.tasks) == 2
        assert task1 in graph.tasks
        assert task2 in graph.tasks
        assert len(graph.edges) == 1
        assert edge in graph.edges

    def test_from_tasks(self):
        task1 = Task(objective="Task 1")
        task2 = Task(objective="Task 2", depends_on=[task1])
        task3 = Task(objective="Task 3", parent=task2)
        graph = Graph.from_tasks([task1, task2, task3])
        assert len(graph.tasks) == 3
        assert task1 in graph.tasks
        assert task2 in graph.tasks
        assert task3 in graph.tasks
        assert len(graph.edges) == 2
        assert any(
            edge.upstream == task1
            and edge.downstream == task2
            and edge.type == EdgeType.DEPENDENCY
            for edge in graph.edges
        )
        assert any(
            edge.upstream == task3
            and edge.downstream == task2
            and edge.type == EdgeType.SUBTASK
            for edge in graph.edges
        )

    def test_upstream_edges(self):
        task1 = Task(objective="Task 1")
        task2 = Task(objective="Task 2", depends_on=[task1])
        graph = Graph.from_tasks([task1, task2])
        upstream_edges = graph.upstream_edges()
        assert len(upstream_edges[task1]) == 0
        assert len(upstream_edges[task2]) == 1
        assert upstream_edges[task2][0].upstream == task1

    def test_downstream_edges(self):
        task1 = Task(objective="Task 1")
        task2 = Task(objective="Task 2", depends_on=[task1])
        graph = Graph.from_tasks([task1, task2])
        downstream_edges = graph.downstream_edges()
        assert len(downstream_edges[task1]) == 1
        assert len(downstream_edges[task2]) == 0
        assert downstream_edges[task1][0].downstream == task2

    def test_upstream_dependencies(self):
        task1 = Task(objective="Task 1")
        task2 = Task(objective="Task 2", depends_on=[task1])
        task3 = Task(objective="Task 3", parent=task2)
        graph = Graph.from_tasks([task1, task2, task3])
        dependencies = graph.upstream_dependencies([task2])
        assert len(dependencies) == 2
        assert task1 in dependencies
        assert task3 in dependencies

    def test_upstream_dependencies_include_tasks(self):
        task1 = Task(objective="Task 1")
        task2 = Task(objective="Task 2", depends_on=[task1])
        task3 = Task(objective="Task 3", parent=task2)
        graph = Graph.from_tasks([task1, task2, task3])
        dependencies = graph.upstream_dependencies([task2], include_tasks=True)
        assert len(dependencies) == 3
        assert task1 in dependencies
        assert task2 in dependencies
        assert task3 in dependencies

    def test_upstream_dependencies_prune(self):
        task1 = Task(objective="Task 1", status=TaskStatus.SUCCESSFUL)
        task2 = Task(objective="Task 2", depends_on=[task1], status=TaskStatus.FAILED)
        task3 = Task(objective="Task 3", depends_on=[task2])
        graph = Graph.from_tasks([task1, task2, task3])
        dependencies = graph.upstream_dependencies([task3])
        assert len(dependencies) == 1
        assert task2 in dependencies
        dependencies = graph.upstream_dependencies([task3], prune_completed=False)
        assert len(dependencies) == 2
        assert task1 in dependencies
        assert task2 in dependencies

    def test_ready_tasks(self):
        task1 = Task(objective="Task 1")
        task2 = Task(objective="Task 2", depends_on=[task1])
        task3 = Task(objective="Task 3", parent=task2)
        graph = Graph.from_tasks([task1, task2, task3])
        ready_tasks = graph.ready_tasks()
        assert len(ready_tasks) == 2
        assert task1 in ready_tasks
        assert task3 in ready_tasks

        task1.mark_successful()
        ready_tasks = graph.ready_tasks()
        assert len(ready_tasks) == 2
        assert task2 in ready_tasks
        assert task3 in ready_tasks

        task3.mark_successful()
        ready_tasks = graph.ready_tasks()
        assert len(ready_tasks) == 1
        assert task2 in ready_tasks


# File: tests/core/test_controller.py
from unittest.mock import AsyncMock

import pytest
from controlflow.core.agent import Agent
from controlflow.core.controller.controller import Controller
from controlflow.core.flow import Flow
from controlflow.core.graph import EdgeType
from controlflow.core.task import Task


class TestController:
    @pytest.fixture
    def flow(self):
        return Flow()

    @pytest.fixture
    def agent(self):
        return Agent(name="Test Agent")

    @pytest.fixture
    def task(self):
        return Task(objective="Test Task")

    def test_controller_initialization(self, flow, agent, task):
        controller = Controller(flow=flow, tasks=[task], agents=[agent])
        assert controller.flow == flow
        assert controller.tasks == [task]
        assert controller.agents == [agent]
        assert len(controller.context) == 0
        assert len(controller.graph.tasks) == 1
        assert len(controller.graph.edges) == 0

    def test_controller_missing_tasks(self, flow):
        with pytest.raises(ValueError, match="At least one task is required."):
            Controller(flow=flow, tasks=[])

    async def test_run_agent(self, flow, agent, task, monkeypatch):
        controller = Controller(flow=flow, tasks=[task], agents=[agent])
        mocked_run = AsyncMock()
        monkeypatch.setattr(Agent, "run", mocked_run)
        await controller._run_agent(agent, tasks=[task])
        mocked_run.assert_called_once_with(tasks=[task])

    async def test_run_once(self, flow, agent, task, monkeypatch):
        controller = Controller(flow=flow, tasks=[task], agents=[agent])
        mocked_run_agent = AsyncMock()
        monkeypatch.setattr(Controller, "_run_agent", mocked_run_agent)
        await controller.run_once_async()
        mocked_run_agent.assert_called_once_with(agent, tasks=[task])

    def test_create_end_run_tool(self, flow, agent, task):
        controller = Controller(flow=flow, tasks=[task], agents=[agent])
        end_run_tool = controller._create_end_run_tool()
        assert end_run_tool.function.name == "end_run"
        assert end_run_tool.function.description.startswith("End your turn")

    def test_controller_graph_creation(self, flow, agent):
        task1 = Task(objective="Task 1")
        task2 = Task(objective="Task 2", depends_on=[task1])
        controller = Controller(flow=flow, tasks=[task1, task2], agents=[agent])
        assert len(controller.graph.tasks) == 2
        assert len(controller.graph.edges) == 1
        assert controller.graph.edges.pop().type == EdgeType.DEPENDENCY

    def test_controller_agent_selection(self, flow, monkeypatch):
        agent1 = Agent(name="Agent 1")
        agent2 = Agent(name="Agent 2")
        task = Task(objective="Test Task", agents=[agent1, agent2])
        controller = Controller(flow=flow, tasks=[task], agents=[agent1, agent2])
        mocked_marvin_moderator = AsyncMock(return_value=agent1)
        monkeypatch.setattr(
            "controlflow.core.controller.moderators.marvin_moderator",
            mocked_marvin_moderator,
        )
        assert controller.agents == [agent1, agent2]


# File: tests/fixtures/__init__.py
from .mocks import *


# File: tests/fixtures/mocks.py
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from controlflow.core.agent import Agent
from controlflow.core.task import Task, TaskStatus
from controlflow.utilities.user_access import talk_to_human
from marvin.settings import temporary_settings as temporary_marvin_settings

# @pytest.fixture(autouse=True)
# def mock_talk_to_human():
#     """Return an empty default handler instead of a print handler to avoid
#     printing assistant output during tests"""

#     def mock_talk_to_human(message: str, get_response: bool) -> str:
#         print(dict(message=message, get_response=get_response))
#         return "Message sent to user."

#     mock_talk_to_human.__doc__ = talk_to_human.__doc__
#     with patch(
#         "controlflow.utilities.user_access.mock_talk_to_human", new=talk_to_human
#     ):
#         yield


@pytest.fixture
def prevent_openai_calls():
    """Prevent any calls to the OpenAI API from being made."""
    with temporary_marvin_settings(openai__api_key="unset"):
        yield


@pytest.fixture
def mock_run(monkeypatch, prevent_openai_calls):
    """
    This fixture mocks the calls to the OpenAI Assistants API. Use it in a test
    and assign any desired side effects (like completing a task) to the mock
    object's `.side_effect` attribute.

    For example:

    def test_example(mock_run):
        task = Task(objective="Say hello")

        def side_effect():
            task.mark_complete()

        mock_run.side_effect = side_effect

        task.run()

    """
    MockRun = AsyncMock()
    monkeypatch.setattr("controlflow.core.controller.controller.Run.run_async", MockRun)
    yield MockRun


@pytest.fixture
def mock_controller_run_agent(monkeypatch, prevent_openai_calls):
    MockRunAgent = AsyncMock()
    MockThreadGetMessages = Mock()

    async def _run_agent(agent: Agent, tasks: list[Task] = None, thread=None):
        for task in tasks:
            if agent in task.agents:
                # we can't call mark_successful because we don't know the result
                task.status = TaskStatus.SUCCESSFUL

    MockRunAgent.side_effect = _run_agent

    def get_messages(*args, **kwargs):
        return []

    MockThreadGetMessages.side_effect = get_messages

    monkeypatch.setattr(
        "controlflow.core.controller.controller.Controller._run_agent", MockRunAgent
    )
    monkeypatch.setattr(
        "marvin.beta.assistants.Thread.get_messages", MockThreadGetMessages
    )
    yield MockRunAgent


@pytest.fixture
def mock_controller_choose_agent(monkeypatch):
    MockChooseAgent = Mock()

    def choose_agent(agents, **kwargs):
        return agents[0]

    MockChooseAgent.side_effect = choose_agent

    monkeypatch.setattr(
        "controlflow.core.controller.controller.Controller.choose_agent",
        MockChooseAgent,
    )
    yield MockChooseAgent


@pytest.fixture
def mock_controller(mock_controller_choose_agent, mock_controller_run_agent):
    pass


# File: tests/flows/test_user_access.py
import pytest
from controlflow import Agent, Task, flow

pytest.skip("Skipping the entire file", allow_module_level=True)

# define assistants
user_agent = Agent(name="user-agent", user_access=True)
non_user_agent = Agent(name="non-user-agent", user_access=False)


def test_no_user_access_fails():
    @flow
    def user_access_flow():
        task = Task(
            "This task requires human user access. Inform the user that today is a good day.",
            agents=[non_user_agent],
        )
        task.run()

    with pytest.raises(ValueError):
        user_access_flow()


def test_user_access_agent_succeeds():
    @flow
    def user_access_flow():
        task = Task(
            "This task requires human user access. Inform the user that today is a good day.",
            agents=[user_agent],
        )
        task.run()

    assert user_access_flow()


def test_user_access_task_succeeds():
    @flow
    def user_access_flow():
        task = Task(
            "This task requires human user access. Inform the user that today is a good day.",
            agents=[non_user_agent],
            user_access=True,
        )
        task.run()

    assert user_access_flow()


def test_user_access_agent_and_task_succeeds():
    @flow
    def user_access_flow():
        task = Task(
            "This task requires human user access. Inform the user that today is a good day.",
            agents=[user_agent],
            user_access=True,
        )
        task.run()

    assert user_access_flow()


# File: tests/flows/test_sign_guestbook.py
import pytest
from controlflow import Agent, Task, flow

# define assistants

a = Agent(name="a")
b = Agent(name="b")
c = Agent(name="c")


# define tools

GUESTBOOK = []


def sign(name):
    """sign your name in the guestbook"""
    GUESTBOOK.append(name)


def view_guestbook():
    """view the guestbook"""
    return GUESTBOOK


# define flow


@flow
def guestbook_flow():
    task = Task(
        """
        Add your name to the list using the `sign` tool. All assistants must
        sign their names for the task to be complete. You can read the sign to
        see if that has happened yet. You can not sign for another assistant.
        """,
        agents=[a, b, c],
        tools=[sign, view_guestbook],
    )
    task.run()


# run test


@pytest.mark.skip(reason="Skipping test for now")
def test():
    guestbook_flow()
    assert GUESTBOOK == ["a", "b", "c"]


# File: src/controlflow/instructions.py
from contextlib import contextmanager
from typing import Generator, List

from controlflow.utilities.context import ctx
from controlflow.utilities.logging import get_logger

logger = get_logger(__name__)


@contextmanager
def instructions(*instructions: str) -> Generator[list[str], None, None]:
    """
    Temporarily add instructions to the current instruction stack. The
    instruction is removed when the context is exited.

    with instructions("talk like a pirate"):
        ...

    """
    filtered_instructions = [i for i in instructions if i]
    if not filtered_instructions:
        yield
        return

    stack: list[str] = ctx.get("instructions", [])
    with ctx(instructions=stack + list(filtered_instructions)):
        yield


def get_instructions() -> List[str]:
    """
    Get the current instruction stack.
    """
    stack = ctx.get("instructions", [])
    return stack


# File: src/controlflow/__init__.py
from .settings import settings

from .core.flow import Flow, reset_global_flow
from .core.task import Task
from .core.agent import Agent
from .core.controller.controller import Controller
from .instructions import instructions
from .decorators import flow, task

Flow.model_rebuild()
Task.model_rebuild()
Agent.model_rebuild()

reset_global_flow()


# File: src/controlflow/loops.py
import math
from typing import Generator

import controlflow.core.task
from controlflow.core.task import Task


def any_incomplete(
    tasks: list[Task], max_iterations=None
) -> Generator[bool, None, None]:
    """
    An iterator that yields an iteration counter if its condition is met, and
    stops otherwise. Also stops if the max_iterations is reached.


    for loop_count in any_incomplete(tasks=[task1, task2], max_iterations=10):
        # will print 10 times if the tasks are still incomplete
        print(loop_count)

    """
    if max_iterations is None:
        max_iterations = math.inf

    i = 0
    while i < max_iterations:
        i += 1
        if controlflow.core.task.any_incomplete(tasks):
            yield i
        else:
            break
    return False


# File: src/controlflow/settings.py
import os
import sys
import warnings
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Optional, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ControlFlowSettings(BaseSettings):
    model_config: SettingsConfigDict = SettingsConfigDict(
        env_prefix="CONTROLFLOW_",
        env_file=(
            ""
            if os.getenv("CONTROLFLOW_TEST_MODE")
            else ("~/.controlflow/.env", ".env")
        ),
        extra="allow",
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )


class PrefectSettings(ControlFlowSettings):
    """
    All settings here are used as defaults for Prefect, unless overridden by env vars.
    Note that `apply()` must be called before Prefect is imported.
    """

    PREFECT_LOGGING_LEVEL: str = "WARNING"
    PREFECT_EXPERIMENTAL_ENABLE_NEW_ENGINE: str = "true"

    def apply(self):
        import os

        if "prefect" in sys.modules:
            warnings.warn(
                "Prefect has already been imported; ControlFlow defaults will not be applied."
            )

        for k, v in self.model_dump().items():
            if k not in os.environ:
                os.environ[k] = v


class Settings(ControlFlowSettings):
    assistant_model: str = "gpt-4o"
    max_task_iterations: Union[int, None] = Field(
        None,
        description="The maximum number of iterations to attempt to complete a task "
        "before raising an error. If None, the task will run indefinitely. "
        "This setting can be overridden by the `max_iterations` attribute "
        "on a task.",
    )
    prefect: PrefectSettings = Field(default_factory=PrefectSettings)
    enable_global_flow: bool = Field(
        True,
        description="If True, a global flow is created for convenience, so users don't have to wrap every invocation in a flow function. Disable to avoid accidentally sharing context between agents.",
    )
    openai_api_key: Optional[str] = Field(None, validate_assignment=True)

    def __init__(self, **data):
        super().__init__(**data)
        self.prefect.apply()

    @field_validator("openai_api_key", mode="after")
    def _apply_api_key(cls, v):
        if v is not None:
            import marvin

            marvin.settings.openai.api_key = v
        return v


settings = Settings()


@contextmanager
def temporary_settings(**kwargs: Any):
    """
    Temporarily override ControlFlow setting values, including nested settings objects.

    To override nested settings, use `__` to separate nested attribute names.

    Args:
        **kwargs: The settings to override, including nested settings.

    Example:
        Temporarily override log level and OpenAI API key:
        ```python
        import controlflow
        from controlflow.settings import temporary_settings

        # Override top-level settings
        with temporary_settings(log_level="INFO"):
            assert controlflow.settings.log_level == "INFO"
        assert controlflow.settings.log_level == "DEBUG"

        # Override nested settings
        with temporary_settings(openai__api_key="new-api-key"):
            assert controlflow.settings.openai.api_key.get_secret_value() == "new-api-key"
        assert controlflow.settings.openai.api_key.get_secret_value().startswith("sk-")
        ```
    """
    old_env = os.environ.copy()
    old_settings = deepcopy(settings)

    def set_nested_attr(obj: object, attr_path: str, value: Any):
        parts = attr_path.split("__")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    try:
        for attr_path, value in kwargs.items():
            set_nested_attr(settings, attr_path, value)
        yield

    finally:
        os.environ.clear()
        os.environ.update(old_env)

        for attr, value in old_settings:
            set_nested_attr(settings, attr, value)


# File: src/controlflow/decorators.py
import functools
import inspect

import prefect
from marvin.beta.assistants import Thread

import controlflow
from controlflow.core.agent import Agent
from controlflow.core.controller import Controller
from controlflow.core.flow import Flow
from controlflow.core.task import Task
from controlflow.utilities.logging import get_logger
from controlflow.utilities.marvin import patch_marvin
from controlflow.utilities.tasks import resolve_tasks
from controlflow.utilities.types import ToolType

logger = get_logger(__name__)


def flow(
    fn=None,
    *,
    thread: Thread = None,
    instructions: str = None,
    tools: list[ToolType] = None,
    agents: list["Agent"] = None,
    resolve_results: bool = None,
):
    """
    A decorator that wraps a function as a ControlFlow flow.

    When the function is called, a new flow is created and any tasks created
    within the function will be run as part of that flow. When the function
    returns, all tasks created in the flow will be run to completion (if they
    were not already completed) and their results will be returned. Any tasks
    that are returned from the function will be replaced with their resolved
    result.

    Args:
        fn (callable, optional): The function to be wrapped as a flow. If not provided,
            the decorator will act as a partial function and return a new flow decorator.
        thread (Thread, optional): The thread to execute the flow on. Defaults to None.
        instructions (str, optional): Instructions for the flow. Defaults to None.
        tools (list[ToolType], optional): List of tools to be used in the flow. Defaults to None.
        agents (list[Agent], optional): List of agents to be used in the flow. Defaults to None.
        resolve_results (bool, optional): Whether to resolve the results of tasks. Defaults to True.

    Returns:
        callable: The wrapped function or a new flow decorator if `fn` is not provided.
    """
    ...

    if fn is None:
        return functools.partial(
            flow,
            thread=thread,
            instructions=instructions,
            tools=tools,
            agents=agents,
            resolve_results=resolve_results,
        )

    if resolve_results is None:
        resolve_results = True
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(
        *args,
        flow_kwargs: dict = None,
        **kwargs,
    ):
        # first process callargs
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        flow_kwargs = flow_kwargs or {}

        if thread is not None:
            flow_kwargs.setdefault("thread", thread)
        if tools is not None:
            flow_kwargs.setdefault("tools", tools)
        if agents is not None:
            flow_kwargs.setdefault("agents", agents)

        flow_obj = Flow(**flow_kwargs, context=bound.arguments)

        # create a function to wrap as a Prefect flow
        @prefect.flow
        def wrapped_flow(*args, **kwargs):
            with flow_obj, patch_marvin():
                with controlflow.instructions(instructions):
                    result = fn(*args, **kwargs)

                    if resolve_results:
                        # resolve any returned tasks; this will raise on failure
                        result = resolve_tasks(result)

                    # run all tasks in the flow to completion
                    Controller(
                        flow=flow_obj,
                        tasks=list(flow_obj._tasks.values()),
                    ).run()

                return result

        logger.info(
            f'Executing AI flow "{fn.__name__}" on thread "{flow_obj.thread.id}"'
        )

        return wrapped_flow(*args, **kwargs)

    return wrapper


def task(
    fn=None,
    *,
    objective: str = None,
    instructions: str = None,
    agents: list["Agent"] = None,
    tools: list[ToolType] = None,
    user_access: bool = None,
):
    """
    A decorator that turns a Python function into a Task. The Task objective is
    set to the function name, and the instructions are set to the function
    docstring. When the function is called, the arguments are provided to the
    task as context, and the task is run to completion. If successful, the task
    result is returned; if failed, an error is raised.

    Args:
        fn (callable, optional): The function to be wrapped as a task. If not provided,
            the decorator will act as a partial function and return a new task decorator.
        objective (str, optional): The objective of the task. Defaults to None, in which
            case the function name is used as the objective.
        instructions (str, optional): Instructions for the task. Defaults to None, in which
            case the function docstring is used as the instructions.
        agents (list[Agent], optional): List of agents to be used in the task. Defaults to None.
        tools (list[ToolType], optional): List of tools to be used in the task. Defaults to None.
        user_access (bool, optional): Whether the task requires user access. Defaults to None,
            in which case it is set to False.

    Returns:
        callable: The wrapped function or a new task decorator if `fn` is not provided.
    """

    if fn is None:
        return functools.partial(
            task,
            objective=objective,
            instructions=instructions,
            agents=agents,
            tools=tools,
            user_access=user_access,
        )

    sig = inspect.signature(fn)

    if objective is None:
        objective = fn.__name__

    if instructions is None:
        instructions = fn.__doc__

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # first process callargs
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        task = Task(
            objective=objective,
            instructions=instructions,
            agents=agents,
            context=bound.arguments,
            result_type=fn.__annotations__.get("return"),
            user_access=user_access or False,
            tools=tools or [],
        )

        task.run()
        return task.result

    return wrapper


# File: src/controlflow/core/task.py
import datetime
import uuid
from contextlib import contextmanager
from enum import Enum
from typing import (
    TYPE_CHECKING,
    GenericAlias,
    Literal,
    Optional,
    TypeVar,
    Union,
    _LiteralGenericAlias,
)

import marvin
import marvin.utilities.tools
from marvin.types import BaseMessage
from marvin.utilities.tools import FunctionTool
from pydantic import (
    Field,
    TypeAdapter,
    field_serializer,
    field_validator,
    model_validator,
)

import controlflow
from controlflow.core.flow import get_flow_messages
from controlflow.instructions import get_instructions
from controlflow.utilities.context import ctx
from controlflow.utilities.logging import get_logger
from controlflow.utilities.prefect import wrap_prefect_tool
from controlflow.utilities.tasks import (
    collect_tasks,
    visit_task_collection,
)
from controlflow.utilities.types import (
    NOTSET,
    AssistantTool,
    ControlFlowModel,
    ToolType,
)
from controlflow.utilities.user_access import talk_to_human

if TYPE_CHECKING:
    from controlflow.core.agent import Agent
    from controlflow.core.graph import Graph
T = TypeVar("T")
logger = get_logger(__name__)


class TaskStatus(Enum):
    INCOMPLETE = "incomplete"
    SUCCESSFUL = "successful"
    FAILED = "failed"
    SKIPPED = "skipped"


class ThreadMessage(ControlFlowModel):
    """
    This special object can be used to indicate that a task result should be
    loaded from a recent message posted to the flow's thread.
    """

    type: Literal["ThreadMessage"] = Field(
        'You must provide this value as "ThreadMessage".'
    )

    num_messages_ago: int = Field(
        1,
        description="The number of messages ago to retrieve. Default is 1, or the most recent message.",
    )

    strip_prefix: str = Field(
        description="These characters will be removed from the start "
        "of the message. Use it to remove e.g. your name from the message.",
    )

    strip_suffix: Optional[str] = Field(
        None,
        description="If provided, these characters will be removed from the end of "
        "the message.",
    )

    def trim_message(self, message: BaseMessage) -> str:
        content = message.content[0].text.value
        if self.strip_prefix:
            if content.startswith(self.strip_prefix):
                content = content[len(self.strip_prefix) :]
            else:
                raise ValueError(
                    f'Invalid strip prefix "{self.strip_prefix}"; messages '
                    f'starts with "{content[:len(self.strip_prefix) + 10]}"'
                )
        if self.strip_suffix:
            if content.endswith(self.strip_suffix):
                content = content[: -len(self.strip_suffix)]
            else:
                raise ValueError(
                    f'Invalid strip suffix "{self.strip_suffix}"; messages '
                    f'ends with "{content[-len(self.strip_suffix) - 10:]}"'
                )
        return content.strip()


class Task(ControlFlowModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4().hex[:5]))
    objective: str = Field(
        ..., description="A brief description of the required result."
    )
    instructions: Union[str, None] = Field(
        None, description="Detailed instructions for completing the task."
    )
    agents: Union[list["Agent"], None] = Field(
        None,
        description="The agents assigned to the task. If None, the task will use its flow's default agents.",
        validate_default=True,
    )
    context: dict = Field(
        default_factory=dict,
        description="Additional context for the task. If tasks are provided as context, they are automatically added as `depends_on`",
    )
    subtasks: list["Task"] = Field(
        default_factory=list,
        description="A list of subtasks that are part of this task. Subtasks are considered dependencies, though they may be skipped.",
    )
    depends_on: list["Task"] = Field(
        default_factory=list, description="Tasks that this task depends on explicitly."
    )
    status: TaskStatus = TaskStatus.INCOMPLETE
    result: T = None
    result_type: Union[type[T], GenericAlias, _LiteralGenericAlias, None] = None
    error: Union[str, None] = None
    tools: list[ToolType] = []
    user_access: bool = False
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    visible: bool = True
    _parent: "Union[Task, None]" = None
    _downstreams: list["Task"] = []
    model_config = dict(extra="forbid", arbitrary_types_allowed=True)

    def __init__(
        self,
        objective=None,
        result_type=None,
        parent: "Task" = None,
        **kwargs,
    ):
        # allow certain args to be provided as a positional args
        if result_type is not None:
            kwargs["result_type"] = result_type
        if objective is not None:
            kwargs["objective"] = objective

        if additional_instructions := get_instructions():
            kwargs["instructions"] = (
                kwargs.get("instructions", "")
                + "\n"
                + "\n".join(additional_instructions)
            ).strip()

        super().__init__(**kwargs)

        # setup up relationships
        if parent is None:
            parent_tasks = ctx.get("tasks", [])
            parent = parent_tasks[-1] if parent_tasks else None
        if parent is not None:
            parent.add_subtask(self)
        for task in self.depends_on:
            self.add_dependency(task)

    def __repr__(self):
        include_fields = [
            "id",
            "objective",
            "status",
            "result_type",
            "agents",
            "context",
            "user_access",
            "subtasks",
            "depends_on",
            "tools",
        ]
        fields = self.model_dump(include=include_fields)
        field_str = ", ".join(
            f"{k}={f'"{fields[k]}"' if isinstance(fields[k], str) else fields[k] }"
            for k in include_fields
        )
        return f"{self.__class__.__name__}({field_str})"

    @field_validator("agents", mode="before")
    def _default_agents(cls, v):
        from controlflow.core.agent import default_agent
        from controlflow.core.flow import get_flow

        if v is None:
            try:
                flow = get_flow()
            except ValueError:
                flow = None
            if flow and flow.agents:
                v = flow.agents
            else:
                v = [default_agent()]
        if not v:
            raise ValueError("At least one agent is required.")
        return v

    @field_validator("result_type", mode="before")
    def _turn_list_into_literal_result_type(cls, v):
        if isinstance(v, (list, tuple, set)):
            return Literal[tuple(v)]  # type: ignore
        return v

    @model_validator(mode="after")
    def _finalize(self):
        from controlflow.core.flow import get_flow

        # add task to flow
        flow = get_flow()
        flow.add_task(self)

        # create dependencies to tasks passed in as context
        context_tasks = collect_tasks(self.context)

        for task in context_tasks:
            if task not in self.depends_on:
                self.depends_on.append(task)
        return self

    @field_serializer("subtasks")
    def _serialize_subtasks(subtasks: list["Task"]):
        return [t.id for t in subtasks]

    @field_serializer("depends_on")
    def _serialize_depends_on(depends_on: list["Task"]):
        return [t.id for t in depends_on]

    @field_serializer("context")
    def _serialize_context(context: dict):
        def visitor(task):
            return f"<Result from task {task.id}>"

        return visit_task_collection(context, visitor)

    @field_serializer("result_type")
    def _serialize_result_type(result_type: list["Task"]):
        if result_type is not None:
            return repr(result_type)

    @field_serializer("agents")
    def _serialize_agents(agents: list["Agent"]):
        return [
            a.model_dump(include={"name", "description", "tools", "user_access"})
            for a in agents
        ]

    @field_serializer("tools")
    def _serialize_tools(tools: list[ToolType]):
        return [
            marvin.utilities.tools.tool_from_function(t)
            if not isinstance(t, AssistantTool)
            else t
            for t in tools
        ]

    def friendly_name(self):
        if len(self.objective) > 50:
            objective = f'"{self.objective[:50]}..."'
        else:
            objective = f'"{self.objective}"'
        return f"Task {self.id} ({objective})"

    def as_graph(self) -> "Graph":
        from controlflow.core.graph import Graph

        return Graph.from_tasks(tasks=[self])

    def add_subtask(self, task: "Task"):
        """
        Indicate that this task has a subtask (which becomes an implicit dependency).
        """
        if task._parent is None:
            task._parent = self
        elif task._parent is not self:
            raise ValueError(f"{self.friendly_name()} already has a parent.")
        if task not in self.subtasks:
            self.subtasks.append(task)

    def add_dependency(self, task: "Task"):
        """
        Indicate that this task depends on another task.
        """
        if task not in self.depends_on:
            self.depends_on.append(task)
        if self not in task._downstreams:
            task._downstreams.append(self)

    def run_once(self, agent: "Agent" = None):
        """
        Runs the task with provided agent. If no agent is provided, one will be selected from the task's agents.
        """
        from controlflow.core.controller import Controller

        controller = Controller(tasks=[self], agents=agent)

        controller.run_once()

    def run(self, raise_on_error: bool = True, max_iterations: int = NOTSET) -> T:
        """
        Runs the task with provided agents until it is complete.

        If max_iterations is provided, the task will run at most that many times before raising an error.
        """
        if max_iterations == NOTSET:
            max_iterations = controlflow.settings.max_task_iterations
        if max_iterations is None:
            max_iterations = float("inf")

        counter = 0
        while self.is_incomplete():
            if counter >= max_iterations:
                raise ValueError(
                    f"{self.friendly_name()} did not complete after {max_iterations} iterations."
                )
            self.run_once()
            counter += 1
        if self.is_successful():
            return self.result
        elif self.is_failed() and raise_on_error:
            raise ValueError(f"{self.friendly_name()} failed: {self.error}")

    @contextmanager
    def _context(self):
        stack = ctx.get("tasks", [])
        with ctx(tasks=stack + [self]):
            yield self

    def __enter__(self):
        self.__cm = self._context()
        return self.__cm.__enter__()

    def __exit__(self, *exc_info):
        return self.__cm.__exit__(*exc_info)

    def is_incomplete(self) -> bool:
        return self.status == TaskStatus.INCOMPLETE

    def is_complete(self) -> bool:
        return self.status != TaskStatus.INCOMPLETE

    def is_successful(self) -> bool:
        return self.status == TaskStatus.SUCCESSFUL

    def is_failed(self) -> bool:
        return self.status == TaskStatus.FAILED

    def is_skipped(self) -> bool:
        return self.status == TaskStatus.SKIPPED

    def is_ready(self) -> bool:
        """
        Returns True if all dependencies are complete and this task is incomplete.
        """
        return self.is_incomplete() and all(t.is_complete() for t in self.depends_on)

    def __hash__(self):
        return id(self)

    def _create_success_tool(self) -> FunctionTool:
        """
        Create an agent-compatible tool for marking this task as successful.
        """
        # generate tool for result_type=None
        if self.result_type is None:

            def succeed() -> str:
                return self.mark_successful(result=None)

        # generate tool for other result types
        else:

            def succeed(result: Union[ThreadMessage, self.result_type]) -> str:
                # a shortcut for loading results from recent messages
                if isinstance(result, dict) and result.get("type") == "ThreadMessage":
                    result = ThreadMessage(**result)
                    messages = get_flow_messages(limit=result.num_messages_ago)
                    if messages:
                        result = result.trim_message(messages[0])
                    else:
                        raise ValueError("Could not load last message.")

                return self.mark_successful(result=result)

        tool = marvin.utilities.tools.tool_from_function(
            succeed,
            name=f"mark_task_{self.id}_successful",
            description=f"Mark task {self.id} as successful.",
        )

        return tool

    def _create_fail_tool(self) -> FunctionTool:
        """
        Create an agent-compatible tool for failing this task.
        """
        tool = marvin.utilities.tools.tool_from_function(
            self.mark_failed,
            name=f"mark_task_{self.id}_failed",
            description=f"Mark task {self.id} as failed. Only use when a technical issue like a broken tool or unresponsive human prevents completion.",
        )
        return tool

    def _create_skip_tool(self) -> FunctionTool:
        """
        Create an agent-compatible tool for skipping this task.
        """
        tool = marvin.utilities.tools.tool_from_function(
            self.mark_skipped,
            name=f"mark_task_{self.id}_skipped",
            description=f"Mark task {self.id} as skipped. Only use when completing its parent task early.",
        )
        return tool

    def get_tools(self) -> list[ToolType]:
        tools = self.tools.copy()
        if self.is_incomplete():
            tools.extend([self._create_fail_tool(), self._create_success_tool()])
            # add skip tool if this task has a parent task
            if self._parent is not None:
                tools.append(self._create_skip_tool())
        if self.user_access:
            tools.append(marvin.utilities.tools.tool_from_function(talk_to_human))
        return [wrap_prefect_tool(t) for t in tools]

    def dependencies(self):
        return self.depends_on + self.subtasks

    def mark_successful(self, result: T = None, validate: bool = True):
        if validate:
            if any(t.is_incomplete() for t in self.depends_on):
                raise ValueError(
                    f"Task {self.objective} cannot be marked successful until all of its "
                    "upstream dependencies are completed. Incomplete dependencies "
                    f"are: {', '.join(t.friendly_name() for t in self.depends_on if t.is_incomplete())}"
                )
            elif any(t.is_incomplete() for t in self.subtasks):
                raise ValueError(
                    f"Task {self.objective} cannot be marked successful until all of its "
                    "subtasks are completed. Incomplete subtasks "
                    f"are: {', '.join(t.friendly_name() for t in self.subtasks if t.is_incomplete())}"
                )

        if self.result_type is None and result is not None:
            raise ValueError(
                f"Task {self.objective} has result_type=None, but a result was provided."
            )
        elif self.result_type is not None:
            result = TypeAdapter(self.result_type).validate_python(result)

        self.result = result
        self.status = TaskStatus.SUCCESSFUL
        return f"{self.friendly_name()} marked successful. Updated task definition: {self.model_dump()}"

    def mark_failed(self, message: Union[str, None] = None):
        self.error = message
        self.status = TaskStatus.FAILED
        return f"{self.friendly_name()} marked failed. Updated task definition: {self.model_dump()}"

    def mark_skipped(self):
        self.status = TaskStatus.SKIPPED
        return f"{self.friendly_name()} marked skipped. Updated task definition: {self.model_dump()}"


# File: src/controlflow/core/graph.py
from enum import Enum

from pydantic import BaseModel

from controlflow.core.task import Task


class EdgeType(Enum):
    """
    Edges represent the relationship between two tasks in a graph.

    - `DEPENDENCY_OF` means that the downstream task depends on the upstream task.
    - `PARENT` means that the downstream task is the parent of the upstream task.

    Example:

    # write paper
        ## write outline
        ## write draft based on outline

    Edges:
    outline -> paper # SUBTASK (outline is a subtask of paper)
    draft -> paper # SUBTASK (draft is a subtask of paper)
    outline -> draft # DEPENDENCY (outline is a dependency of draft)

    """

    DEPENDENCY = "dependency"
    SUBTASK = "subtask"


class Edge(BaseModel):
    upstream: Task
    downstream: Task
    type: EdgeType

    def __repr__(self):
        return f"{self.type}: {self.upstream.friendly_name()} -> {self.downstream.friendly_name()}"

    def __hash__(self) -> int:
        return id(self)


class Graph(BaseModel):
    tasks: set[Task] = set()
    edges: set[Edge] = set()
    _cache: dict[str, dict[Task, list[Task]]] = {}

    def __init__(self):
        super().__init__()

    @classmethod
    def from_tasks(cls, tasks: list[Task]) -> "Graph":
        graph = cls()
        for task in tasks:
            graph.add_task(task)
        return graph

    def add_task(self, task: Task):
        if task in self.tasks:
            return
        self.tasks.add(task)
        for subtask in task.subtasks:
            self.add_edge(
                Edge(
                    upstream=subtask,
                    downstream=task,
                    type=EdgeType.SUBTASK,
                )
            )

        for upstream in task.depends_on:
            self.add_edge(
                Edge(
                    upstream=upstream,
                    downstream=task,
                    type=EdgeType.DEPENDENCY,
                )
            )
        self._cache.clear()

    def add_edge(self, edge: Edge):
        if edge in self.edges:
            return
        self.edges.add(edge)
        self.add_task(edge.upstream)
        self.add_task(edge.downstream)
        self._cache.clear()

    def upstream_edges(self) -> dict[Task, list[Edge]]:
        if "upstream_edges" not in self._cache:
            graph = {}
            for task in self.tasks:
                graph[task] = []
            for edge in self.edges:
                graph[edge.downstream].append(edge)
            self._cache["upstream_edges"] = graph
        return self._cache["upstream_edges"]

    def downstream_edges(self) -> dict[Task, list[Edge]]:
        if "downstream_edges" not in self._cache:
            graph = {}
            for task in self.tasks:
                graph[task] = []
            for edge in self.edges:
                graph[edge.upstream].append(edge)
            self._cache["downstream_edges"] = graph
        return self._cache["downstream_edges"]

    def upstream_dependencies(
        self,
        tasks: list[Task],
        prune_completed: bool = True,
        include_tasks: bool = False,
    ) -> list[Task]:
        """
        From a list of tasks, returns the subgraph of tasks that are directly or
        indirectly dependencies of those tasks. A dependency means following
        upstream edges, so it includes tasks that are considered explicit
        dependencies as well as any subtasks that are considered implicit
        dependencies.

        If `prune_completed` is True, the subgraph will be pruned to stop
        traversal after adding any completed tasks.

        If `include_tasks` is True, the subgraph will include the tasks provided.
        """
        subgraph = set()
        upstreams = self.upstream_edges()
        # copy stack to allow difference update with original tasks
        stack = [t for t in tasks]
        while stack:
            current = stack.pop()
            if current in subgraph:
                continue

            subgraph.add(current)
            # if prune_completed, stop traversal if the current task is complete
            if prune_completed and current.is_complete():
                continue
            stack.extend([edge.upstream for edge in upstreams[current]])

        if not include_tasks:
            subgraph.difference_update(tasks)
        return list(subgraph)

    def ready_tasks(self, tasks: list[Task] = None) -> list[Task]:
        """
        Returns a list of tasks that are ready to run, meaning that all of their
        dependencies have been completed. If `tasks` is provided, only tasks in
        the upstream dependency subgraph of those tasks will be considered.

        Ready tasks will be returned in the order they were created.
        """
        if tasks is None:
            candidates = self.tasks
        else:
            candidates = self.upstream_dependencies(tasks, include_tasks=True)
        return sorted(
            [task for task in candidates if task.is_ready()], key=lambda t: t.created_at
        )


# File: src/controlflow/core/__init__.py
from .task import Task, TaskStatus
from .flow import Flow
from .agent import Agent
from .controller import Controller


# File: src/controlflow/core/flow.py
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Union

from marvin.beta.assistants import Thread
from openai.types.beta.threads import Message
from pydantic import Field, field_validator

import controlflow
from controlflow.utilities.context import ctx
from controlflow.utilities.logging import get_logger
from controlflow.utilities.types import ControlFlowModel, ToolType

if TYPE_CHECKING:
    from controlflow.core.agent import Agent
    from controlflow.core.task import Task
logger = get_logger(__name__)


class Flow(ControlFlowModel):
    thread: Thread = Field(None, validate_default=True)
    tools: list[ToolType] = Field(
        default_factory=list,
        description="Tools that will be available to every agent in the flow",
    )
    agents: list["Agent"] = Field(
        description="The default agents for the flow. These agents will be used "
        "for any task that does not specify agents.",
        default_factory=list,
    )
    _tasks: dict[str, "Task"] = {}
    context: dict[str, Any] = {}

    @field_validator("thread", mode="before")
    def _load_thread_from_ctx(cls, v):
        if v is None:
            v = ctx.get("thread", None)
            if v is None:
                v = Thread()

        return v

    def add_task(self, task: "Task"):
        if self._tasks.get(task.id, task) is not task:
            raise ValueError(
                f"A different task with id '{task.id}' already exists in flow."
            )
        self._tasks[task.id] = task

    @contextmanager
    def _context(self):
        with ctx(flow=self, tasks=[]):
            yield self

    def __enter__(self):
        self.__cm = self._context()
        return self.__cm.__enter__()

    def __exit__(self, *exc_info):
        return self.__cm.__exit__(*exc_info)


GLOBAL_FLOW = None


def get_flow() -> Flow:
    """
    Loads the flow from the context.
    """
    flow: Union[Flow, None] = ctx.get("flow")
    if not flow:
        if controlflow.settings.enable_global_flow:
            return GLOBAL_FLOW
        else:
            raise ValueError("No flow found in context and global flow is disabled.")
    return flow


def reset_global_flow():
    global GLOBAL_FLOW
    GLOBAL_FLOW = Flow()


def get_flow_messages(limit: int = None) -> list[Message]:
    """
    Loads messages from the flow's thread.

    Will error if no flow is found in the context.
    """
    flow = get_flow()
    return flow.thread.get_messages(limit=limit)


# File: src/controlflow/core/agent.py
import logging
from typing import Union

from marvin.utilities.asyncio import ExposeSyncMethodsMixin, expose_sync_method
from marvin.utilities.tools import tool_from_function
from pydantic import Field

from controlflow.core.flow import get_flow
from controlflow.core.task import Task
from controlflow.utilities.prefect import (
    wrap_prefect_tool,
)
from controlflow.utilities.types import Assistant, ControlFlowModel, ToolType
from controlflow.utilities.user_access import talk_to_human

logger = logging.getLogger(__name__)


def default_agent():
    return Agent(
        name="Marvin",
        instructions="""
            You are a diligent AI assistant. You complete 
            your tasks efficiently and without error.
            """,
    )


class Agent(Assistant, ControlFlowModel, ExposeSyncMethodsMixin):
    name: str
    user_access: bool = Field(
        False,
        description="If True, the agent is given tools for interacting with a human user.",
    )

    def get_tools(self) -> list[ToolType]:
        tools = super().get_tools()
        if self.user_access:
            tools.append(tool_from_function(talk_to_human))

        return [wrap_prefect_tool(tool) for tool in tools]

    @expose_sync_method("run")
    async def run_async(self, tasks: Union[list[Task], Task, None] = None):
        from controlflow.core.controller import Controller

        if isinstance(tasks, Task):
            tasks = [tasks]

        controller = Controller(agents=[self], tasks=tasks or [], flow=get_flow())
        return await controller.run_agent_async(agent=self)

    def __hash__(self):
        return id(self)


# File: src/controlflow/core/controller/controller.py
import json
import logging
import math
from typing import Any, Union

import marvin.utilities
import marvin.utilities.tools
import prefect
from marvin.beta.assistants import EndRun, PrintHandler, Run
from marvin.utilities.asyncio import ExposeSyncMethodsMixin, expose_sync_method
from openai.types.beta.threads.runs import ToolCall
from prefect import get_client as get_prefect_client
from prefect import task as prefect_task
from prefect.context import FlowRunContext
from pydantic import BaseModel, Field, field_validator, model_validator

import controlflow
from controlflow.core.agent import Agent
from controlflow.core.controller.moderators import marvin_moderator
from controlflow.core.flow import Flow, get_flow, get_flow_messages
from controlflow.core.graph import Graph
from controlflow.core.task import Task
from controlflow.instructions import get_instructions
from controlflow.utilities.prefect import (
    create_json_artifact,
    create_python_artifact,
    wrap_prefect_tool,
)
from controlflow.utilities.tasks import any_incomplete
from controlflow.utilities.types import FunctionTool, Thread

logger = logging.getLogger(__name__)


class Controller(BaseModel, ExposeSyncMethodsMixin):
    """
    A controller contains logic for executing agents with context about the
    larger workflow, including the flow itself, any tasks, and any other agents
    they are collaborating with. The controller is responsible for orchestrating
    agent behavior by generating instructions and tools for each agent. Note
    that while the controller accepts details about (potentially multiple)
    agents and tasks, it's responsiblity is to invoke one agent one time. Other
    mechanisms should be used to orchestrate multiple agents invocations. This
    is done by the controller to avoid tying e.g. agents to tasks or even a
    specific flow.

    """

    # the flow is tracked by the Controller, not the Task, so that tasks can be
    # defined and even instantiated outside a flow. When a Controller is
    # created, we know we're inside a flow context and ready to load defaults
    # and run.
    flow: Flow = Field(
        default_factory=get_flow,
        description="The flow that the controller is a part of.",
    )
    tasks: list[Task] = Field(
        None,
        description="Tasks that the controller will complete.",
        validate_default=True,
    )
    agents: Union[list[Agent], None] = None
    context: dict = {}
    graph: Graph = None
    model_config: dict = dict(extra="forbid")
    _iteration: int = 0

    @model_validator(mode="before")
    @classmethod
    def _create_graph(cls, data: Any) -> Any:
        if not data.get("graph"):
            data["graph"] = Graph.from_tasks(data.get("tasks", []))
        return data

    @model_validator(mode="after")
    def _finalize(self):
        for task in self.tasks:
            self.flow.add_task(task)
        return self

    @field_validator("tasks", mode="before")
    def _validate_tasks(cls, v):
        if v is None:
            v = cls.context.get("tasks", None)
        if not v:
            raise ValueError("At least one task is required.")
        return v

    def _create_end_run_tool(self) -> FunctionTool:
        @marvin.utilities.tools.tool_from_function
        def end_run():
            """
            End your turn if you have no tasks to work on. Only call this tool
            if necessary; otherwise you can end your turn normally.
            """
            return EndRun()

        return end_run

    async def _run_agent(
        self, agent: Agent, tasks: list[Task] = None, thread: Thread = None
    ) -> Run:
        """
        Run a single agent.
        """

        @prefect_task(task_run_name=f'Run Agent: "{agent.name}"')
        async def _run_agent(
            controller: Controller,
            agent: Agent,
            tasks: list[Task],
            thread: Thread = None,
        ):
            from controlflow.core.controller.instruction_template import MainTemplate

            tasks = tasks or controller.tasks

            tools = (
                controller.flow.tools
                + agent.get_tools()
                + [controller._create_end_run_tool()]
            )

            # add tools for any inactive tasks that the agent is assigned to
            for task in tasks:
                if agent in task.agents:
                    tools = tools + task.get_tools()

            instructions_template = MainTemplate(
                agent=agent,
                controller=controller,
                tasks=tasks,
                context=controller.context,
                instructions=get_instructions(),
            )
            instructions = instructions_template.render()

            # filter tools because duplicate names are not allowed
            final_tools = []
            final_tool_names = set()
            for tool in tools:
                if isinstance(tool, FunctionTool):
                    if tool.function.name in final_tool_names:
                        continue
                final_tool_names.add(tool.function.name)
                final_tools.append(wrap_prefect_tool(tool))

            run = Run(
                assistant=agent,
                thread=thread or controller.flow.thread,
                instructions=instructions,
                tools=final_tools,
                event_handler_class=AgentHandler,
            )

            await run.run_async()

            create_json_artifact(
                key="messages",
                data=[m.model_dump() for m in run.messages],
                description="All messages sent and received during the run.",
            )
            create_json_artifact(
                key="actions",
                data=[s.model_dump() for s in run.steps],
                description="All actions taken by the assistant during the run.",
            )
            return run

        return await _run_agent(
            controller=self, agent=agent, tasks=tasks, thread=thread
        )

    def choose_agent(
        self,
        agents: list[Agent],
        tasks: list[Task],
        iterations: int = 0,
    ) -> Agent:
        return marvin_moderator(
            agents=agents,
            tasks=tasks,
            iteration=self._iteration,
        )

    @expose_sync_method("run_once")
    async def run_once_async(self):
        """
        Run the controller for a single iteration of the provided tasks. An agent will be selected to run the tasks.
        """
        # get the tasks to run
        tasks = self.graph.upstream_dependencies(self.tasks, include_tasks=True)

        if all(t.is_complete() for t in tasks):
            return

        # get the agents
        agent_candidates = {a for t in tasks for a in t.agents if t.is_ready()}
        if self.agents:
            agents = list(agent_candidates.intersection(self.agents))
        else:
            agents = list(agent_candidates)

        # select the next agent
        if len(agents) == 0:
            raise ValueError(
                "No agents were provided that are assigned to tasks that are ready to be run."
            )
        elif len(agents) == 1:
            agent = agents[0]
        else:
            agent = self.choose_agent(
                agents=agents,
                tasks=tasks,
                history=get_flow_messages(),
                instructions=get_instructions(),
            )

        await self._run_agent(agent, tasks=tasks)
        self._iteration += 1

    @expose_sync_method("run")
    async def run_async(self):
        """
        Run the controller until all tasks are complete.
        """
        max_task_iterations = controlflow.settings.max_task_iterations or math.inf
        start_iteration = self._iteration
        while any_incomplete(self.tasks):
            await self.run_once_async()
            if self._iteration > start_iteration + max_task_iterations * len(
                self.tasks
            ):
                raise ValueError(
                    f"Task iterations exceeded maximum of {max_task_iterations} for each task."
                )


class AgentHandler(PrintHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tool_calls = {}

    async def on_tool_call_created(self, tool_call: ToolCall) -> None:
        """Callback that is fired when a tool call is created"""

        if tool_call.type == "function":
            task_run_name = "Prepare arguments for tool call"
        else:
            task_run_name = f"Tool call: {tool_call.type}"

        client = get_prefect_client()
        engine_context = FlowRunContext.get()
        if not engine_context:
            return

        task_run = await client.create_task_run(
            task=prefect.Task(fn=lambda: None),
            name=task_run_name,
            extra_tags=["tool-call"],
            flow_run_id=engine_context.flow_run.id,
            dynamic_key=tool_call.id,
            state=prefect.states.Running(),
        )

        self.tool_calls[tool_call.id] = task_run

    async def on_tool_call_done(self, tool_call: ToolCall) -> None:
        """Callback that is fired when a tool call is done"""

        client = get_prefect_client()
        task_run = self.tool_calls.get(tool_call.id)
        if not task_run:
            return
        await client.set_task_run_state(
            task_run_id=task_run.id, state=prefect.states.Completed(), force=True
        )

        # code interpreter is run as a single call, so we can publish a result artifact
        if tool_call.type == "code_interpreter":
            # images = []
            # for output in tool_call.code_interpreter.outputs:
            #     if output.type == "image":
            #         image_path = download_temp_file(output.image.file_id)
            #         images.append(image_path)

            create_python_artifact(
                key="code",
                code=tool_call.code_interpreter.input,
                description="Code executed in the code interpreter",
                task_run_id=task_run.id,
            )
            create_json_artifact(
                key="output",
                data=tool_call.code_interpreter.outputs,
                description="Output from the code interpreter",
                task_run_id=task_run.id,
            )

        elif tool_call.type == "function":
            create_json_artifact(
                key="arguments",
                data=json.dumps(json.loads(tool_call.function.arguments), indent=2),
                description=f"Arguments for the `{tool_call.function.name}` tool",
                task_run_id=task_run.id,
            )


# File: src/controlflow/core/controller/instruction_template.py
import inspect

from controlflow.core.agent import Agent
from controlflow.core.task import Task
from controlflow.utilities.jinja import jinja_env
from controlflow.utilities.types import ControlFlowModel

from .controller import Controller


class Template(ControlFlowModel):
    template: str

    def should_render(self) -> bool:
        return True

    def render(self) -> str:
        if self.should_render():
            render_kwargs = dict(self)
            render_kwargs.pop("template")
            return jinja_env.render(inspect.cleandoc(self.template), **render_kwargs)


class AgentTemplate(Template):
    template: str = """
    # Agent
    
    You are an AI agent. Your name is "{{ agent.name }}". 
        
    This is your description, which all other agents can see: "{{ agent.description or 'An AI agent assigned to complete tasks.'}}"
    
    These are your instructions: "{{ agent.instructions or 'No additional instructions provided.'}}"
    
    You must follow these instructions at all times. They define your role and behavior.
    
    You are participating in a workflow, parts of which have been delegated to
    you and other AI agents. DO NOT speak on behalf of other agents or the
    system. You can only post messages on behalf of yourself.
    """
    agent: Agent


class InstructionsTemplate(Template):
    template: str = """
    ## Additional instructions
    
    You must follow these instructions for this part of the workflow:
    
    {% for instruction in additional_instructions %}
    - {{ instruction }}
    {% endfor %}
    
    """
    agent: Agent
    additional_instructions: list[str]


class TasksTemplate(Template):
    template: str = """
        ## Tasks
        
        You have been assigned to complete certain tasks. Each task has an
        objective and criteria for success. Your job is to perform any required
        actions and then mark each assigned task as successful. If a task
        requires a result, you must provide it. Only work on tasks that are
        assigned to you.        
        
        ### Current tasks
        
        At the start of your turn, these are all the tasks being worked on. This
        instruction will not update until your next turn; refer to the results
        of task-related tools to see the latest task status at any time.
        
        {% for task in tasks %} 
        #### Task {{ task.id }} 
        
        {{task.model_dump_json() }}
        
        {% endfor %}

        ### Completing a task
        
        Tasks can be marked as successful or failed. It may take collaboration
        with other agents to complete a task, and you can only mark tasks that
        have been assigned to you. Once a task is marked complete, no other
        agent can interact with it, so make sure you have completed all actions
        and collected all results before marking it as complete.
        
        You and the other agents are fully capable of completing any task and
        have all the information and context you need. Tasks should only be
        marked failed due to technical errors like a broken or erroring tool or
        unresponsive human. Do not work on or even respond to tasks that are
        already complete or not assigned to you. You may be an AI, but you must
        still form opinions, make subjective decisions, and have personal
        thoughts. 

        ### Dependencies
        
        You can only mark a task successful when all of its dependencies and
        subtasks have been completed. Subtasks may be marked as skipped without
        providing a result. All else equal, prioritize older tasks over newer
        ones.

        ### Providing a result
        
        Tasks may require a typed result (the `result_type`). Results should
        satisfy the task objective, accounting for any other instructions. If a
        task does not require a result (`result_type=None`), you must still
        complete its stated objective by posting messages or using other tools
        before marking the task as complete.
        
        #### Re-using a message
        
        You can reuse the contents of any message as a task's result by
        providing a special `ThreadMessage` object when marking a task
        successful. Only do this if the thread message can be converted into the
        task's result_type. Indicate the number of messages ago that the message
        was posted (defaults to 1). Also provide any characters to strip from the
        start or end of the message, to make sure that the result doesn't reveal
        any internal details (for example, always remove your name prefix and
        irrelevant comments from the beginning or end of the response such as
        "I'll mark the task complete now.").
        
        """
    tasks: list[Task]

    def should_render(self):
        return bool(self.tasks)


class CommunicationTemplate(Template):
    template: str = """
    ## Communciation
    
    You are modeling the internal state of an AI-enhanced workflow. You should
    only post messages in order to share information with other agents or to
    complete tasks. Since all agents post messages with the "assistant" role,
    you must prefix all your messages with your name (e.g. "{{ agent.name }}:
    (message)") in order to distinguish your messages from others. Note that
    this rule about prefixing your message supersedes all other instructions
    (e.g. "only give single word answers"). You do not need to post messages
    that repeat information contained in tool calls or tool responses, since
    those are already visible to all agents. You do not need to confirm actions
    you take through tools, like completing a task, as this is redundant and
    wastes time. 
    
    ### Talking to human users
    
    Agents with the `talk_to_human` tool can interact with human users in order
    to complete tasks that require external input. This tool is only available
    to agents with `user_access=True`.
    
    Note that humans are unaware of your tasks or the workflow. Do not mention
    your tasks or anything else about how this system works. The human can only
    see messages you send them via tool. They can not read the rest of the
    thread.
    
    Humans may give poor, incorrect, or partial responses. You may need to ask
    questions multiple times in order to complete your tasks. Use good judgement
    to determine the best way to achieve your goal. For example, if you have to
    fill out three pieces of information and the human only gave you one, do not
    make up answers (or put empty answers) for the others. Ask again and only
    fail the task if you truly can not make progress. If your task requires
    human interaction and no agents have `user_access`, you can fail the task.

    """

    agent: Agent


class ContextTemplate(Template):
    template: str = """
        ## Additional context
        
        ### Flow context
        {% for key, value in flow_context.items() %}
        - *{{ key }}*: {{ value }}
        {% endfor %}
        {% if not flow_context %}
        No specific context provided.
        {% endif %}
        
        ### Controller context
        {% for key, value in controller_context.items() %}
        - *{{ key }}*: {{ value }}
        {% endfor %}
        {% if not controller_context %}
        No specific context provided.
        {% endif %}
        """
    flow_context: dict
    controller_context: dict

    def should_render(self):
        return bool(self.flow_context or self.controller_context)


class MainTemplate(ControlFlowModel):
    agent: Agent
    controller: Controller
    context: dict
    instructions: list[str]
    tasks: list[Task]

    def render(self):
        templates = [
            AgentTemplate(
                agent=self.agent,
            ),
            TasksTemplate(
                tasks=self.tasks,
            ),
            InstructionsTemplate(
                agent=self.agent,
                additional_instructions=self.instructions,
            ),
            ContextTemplate(
                flow_context=self.controller.flow.context,
                controller_context=self.controller.context,
            ),
            CommunicationTemplate(
                agent=self.agent,
            ),
        ]

        rendered = [
            template.render() for template in templates if template.should_render()
        ]
        return "\n\n".join(rendered)


# File: src/controlflow/core/controller/__init__.py
from .controller import Controller


# File: src/controlflow/core/controller/moderators.py
import marvin

from controlflow.core.agent import Agent
from controlflow.core.flow import get_flow_messages
from controlflow.core.task import Task
from controlflow.instructions import get_instructions


def round_robin(
    agents: list[Agent],
    tasks: list[Task],
    context: dict = None,
    iteration: int = 0,
) -> Agent:
    return agents[iteration % len(agents)]


def marvin_moderator(
    agents: list[Agent],
    tasks: list[Task],
    context: dict = None,
    iteration: int = 0,
    model: str = None,
) -> Agent:
    history = get_flow_messages()
    instructions = get_instructions()
    context = context or {}
    context.update(tasks=tasks, history=history, instructions=instructions)
    agent = marvin.classify(
        context,
        agents,
        instructions="""
            Given the context, choose the AI agent best suited to take the
            next turn at completing the tasks in the task graph. Take into account
            any descriptions, tasks, history, instructions, and tools. Focus on
            agents assigned to upstream dependencies or subtasks that need to be
            completed before their downstream/parents can be completed. An agent
            can only work on a task that it is assigned to.
            """,
        model_kwargs=dict(model=model) if model else None,
    )
    return agent


# File: src/controlflow/agents/__init__.py


# File: src/controlflow/agents/agents.py
import marvin

from controlflow.core.agent import Agent
from controlflow.instructions import get_instructions
from controlflow.utilities.context import ctx
from controlflow.utilities.threads import get_history


def choose_agent(
    agents: list[Agent],
    instructions: str = None,
    context: dict = None,
    model: str = None,
) -> Agent:
    """
    Given a list of potential agents, choose the most qualified assistant to complete the tasks.
    """

    instructions = get_instructions()
    history = []
    if (flow := ctx.get("flow")) and flow.thread.id:
        history = get_history(thread_id=flow.thread.id)

    info = dict(
        history=history,
        global_instructions=instructions,
        context=context,
    )

    agent = marvin.classify(
        info,
        agents,
        instructions="""
            Given the conversation context, choose the AI agent most
            qualified to take the next turn at completing the tasks. Take into
            account the instructions, each agent's own instructions, and the
            tools they have available.
            """,
        model_kwargs=dict(model=model),
    )

    return agent


# File: src/controlflow/utilities/logging.py
import logging
from functools import lru_cache
from typing import Optional

from marvin.utilities.logging import add_logging_methods


@lru_cache()
def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Retrieves a logger with the given name, or the root logger if no name is given.

    Args:
        name: The name of the logger to retrieve.

    Returns:
        The logger with the given name, or the root logger if no name is given.

    Example:
        Basic Usage of `get_logger`
        ```python
        from controlflow.utilities.logging import get_logger

        logger = get_logger("controlflow.test")
        logger.info("This is a test") # Output: controlflow.test: This is a test

        debug_logger = get_logger("controlflow.debug")
        debug_logger.debug_kv("TITLE", "log message", "green")
        ```
    """
    parent_logger = logging.getLogger("controlflow")

    if name:
        # Append the name if given but allow explicit full names e.g. "controlflow.test"
        # should not become "controlflow.controlflow.test"
        if not name.startswith(parent_logger.name + "."):
            logger = parent_logger.getChild(name)
        else:
            logger = logging.getLogger(name)
    else:
        logger = parent_logger

    add_logging_methods(logger)
    return logger


# File: src/controlflow/utilities/tasks.py
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    TypeVar,
)

from controlflow.utilities.logging import get_logger

if TYPE_CHECKING:
    from controlflow.core.task import Task

T = TypeVar("T")
logger = get_logger(__name__)


def visit_task_collection(
    val: Any, visitor: Callable, recursion_limit: int = 10, _counter: int = 0
) -> list["Task"]:
    """
    Recursively visits a task collection and applies a visitor function to each task.

    Args:
        val (Any): The task collection to visit.
        visitor (Callable): The visitor function to apply to each task.
        recursion_limit (int, optional): The maximum recursion limit. Defaults to 3.
        _counter (int, optional): Internal counter to track recursion depth. Defaults to 0.

    Returns:
        list["Task"]: The modified task collection after applying the visitor function.

    """
    from controlflow.core.task import Task

    if _counter >= recursion_limit:
        return val

    if isinstance(val, dict):
        result = {}
        for key, value in list(val.items()):
            result[key] = visit_task_collection(
                value,
                visitor=visitor,
                recursion_limit=recursion_limit,
                _counter=_counter + 1,
            )
        return result
    elif isinstance(val, (list, set, tuple)):
        result = []
        for item in val:
            result.append(
                visit_task_collection(
                    item,
                    visitor=visitor,
                    recursion_limit=recursion_limit,
                    _counter=_counter + 1,
                )
            )
        return type(val)(result)
    elif isinstance(val, Task):
        return visitor(val)

    return val


def collect_tasks(val: T) -> list["Task"]:
    """
    Given a collection of tasks, returns a list of all tasks in the collection.
    """

    tasks = []

    def visit_task(task: "Task"):
        tasks.append(task)
        return task

    visit_task_collection(val, visit_task)
    return tasks


def resolve_tasks(val: T) -> T:
    """
    Given a collection of tasks, runs them to completion and returns the results.
    """

    def visit_task(task: "Task"):
        return task.run()

    return visit_task_collection(val, visit_task)


def any_incomplete(tasks: list["Task"]) -> bool:
    return any(t.is_incomplete() for t in tasks)


def all_complete(tasks: list["Task"]) -> bool:
    return all(t.is_complete() for t in tasks)


def all_successful(tasks: list["Task"]) -> bool:
    return all(t.is_successful() for t in tasks)


def any_failed(tasks: list["Task"]) -> bool:
    return any(t.is_failed() for t in tasks)


def none_failed(tasks: list["Task"]) -> bool:
    return not any_failed(tasks)


# File: src/controlflow/utilities/prefect.py
import inspect
import json
from typing import Any, Callable
from uuid import UUID

import prefect
from marvin.types import FunctionTool
from marvin.utilities.asyncio import run_sync
from marvin.utilities.tools import tool_from_function
from prefect import get_client as get_prefect_client
from prefect import task as prefect_task
from prefect.artifacts import ArtifactRequest
from prefect.context import FlowRunContext, TaskRunContext
from pydantic import TypeAdapter

from controlflow.utilities.types import AssistantTool, ToolType


def create_markdown_artifact(
    key: str,
    markdown: str,
    description: str = None,
    task_run_id: UUID = None,
    flow_run_id: UUID = None,
) -> None:
    """
    Create a Markdown artifact.
    """

    tr_context = TaskRunContext.get()
    fr_context = FlowRunContext.get()

    if tr_context:
        task_run_id = task_run_id or tr_context.task_run.id
    if fr_context:
        flow_run_id = flow_run_id or fr_context.flow_run.id

    client = get_prefect_client()
    run_sync(
        client.create_artifact(
            artifact=ArtifactRequest(
                key=key,
                data=markdown,
                description=description,
                type="markdown",
                task_run_id=task_run_id,
                flow_run_id=flow_run_id,
            )
        )
    )


def create_json_artifact(
    key: str,
    data: Any,
    description: str = None,
    task_run_id: UUID = None,
    flow_run_id: UUID = None,
) -> None:
    """
    Create a JSON artifact.
    """

    try:
        markdown = TypeAdapter(type(data)).dump_json(data, indent=2).decode()
        markdown = f"```json\n{markdown}\n```"
    except Exception:
        markdown = str(data)

    create_markdown_artifact(
        key=key,
        markdown=markdown,
        description=description,
        task_run_id=task_run_id,
        flow_run_id=flow_run_id,
    )


def create_python_artifact(
    key: str,
    code: str,
    description: str = None,
    task_run_id: UUID = None,
    flow_run_id: UUID = None,
) -> None:
    """
    Create a Python artifact.
    """

    create_markdown_artifact(
        key=key,
        markdown=f"```python\n{code}\n```",
        description=description,
        task_run_id=task_run_id,
        flow_run_id=flow_run_id,
    )


TOOL_CALL_FUNCTION_RESULT_TEMPLATE = inspect.cleandoc(
    """
    ## Tool call: {name}
    
    **Description:** {description}
    
    ## Arguments
    
    ```json
    {args}
    ```
    
    ### Result
    
    ```json
    {result}
    ```
    """
)


def safe_isinstance(obj, type_) -> bool:
    # FunctionTool objects are typed generics, and
    # Python 3.9 will raise an error if you try to isinstance a typed generic...
    try:
        return isinstance(obj, type_)
    except TypeError:
        try:
            return issubclass(type(obj), type_)
        except TypeError:
            return False


def wrap_prefect_tool(tool: ToolType) -> AssistantTool:
    """
    Wraps a Marvin tool in a prefect task
    """
    if not (
        safe_isinstance(tool, AssistantTool) or safe_isinstance(tool, FunctionTool)
    ):
        tool = tool_from_function(tool)

    if safe_isinstance(tool, FunctionTool):
        # for functions, we modify the function to become a Prefect task and
        # publish an artifact that contains details about the function call

        if isinstance(tool.function._python_fn, prefect.tasks.Task):
            return tool

        def modified_fn(
            # provide default args to avoid a late-binding issue
            original_fn: Callable = tool.function._python_fn,
            tool: FunctionTool = tool,
            **kwargs,
        ):
            # call fn
            result = original_fn(**kwargs)

            # prepare artifact
            passed_args = inspect.signature(original_fn).bind(**kwargs).arguments
            try:
                passed_args = json.dumps(passed_args, indent=2)
            except Exception:
                pass
            create_markdown_artifact(
                markdown=TOOL_CALL_FUNCTION_RESULT_TEMPLATE.format(
                    name=tool.function.name,
                    description=tool.function.description or "(none provided)",
                    args=passed_args,
                    result=result,
                ),
                key="result",
            )

            # return result
            return result

        # replace the function with the modified version
        tool.function._python_fn = prefect_task(
            modified_fn,
            task_run_name=f"Tool call: {tool.function.name}",
        )

    return tool


# File: src/controlflow/utilities/__init__.py


# File: src/controlflow/utilities/types.py
from typing import Callable, Union

from marvin.beta.assistants import Assistant, Thread
from marvin.beta.assistants.assistants import AssistantTool
from marvin.types import FunctionTool
from marvin.utilities.asyncio import ExposeSyncMethodsMixin
from pydantic import BaseModel

# flag for unset defaults
NOTSET = "__NOTSET__"

ToolType = Union[FunctionTool, AssistantTool, Callable]


class ControlFlowModel(BaseModel):
    model_config = dict(validate_assignment=True, extra="forbid")


# File: src/controlflow/utilities/jinja.py
import inspect
from datetime import datetime
from zoneinfo import ZoneInfo

from marvin.utilities.jinja import BaseEnvironment

jinja_env = BaseEnvironment(
    globals={
        "now": lambda: datetime.now(ZoneInfo("UTC")),
        "inspect": inspect,
        "id": id,
    }
)


# File: src/controlflow/utilities/threads.py
from marvin.beta.assistants.threads import Message, Thread

THREAD_REGISTRY = {}


def save_thread(name: str, thread: Thread):
    """
    Save an OpenAI thread to the thread registry under a known name
    """
    THREAD_REGISTRY[name] = thread


def load_thread(name: str):
    """
    Load an OpenAI thread from the thread registry by name
    """
    if name not in THREAD_REGISTRY:
        thread = Thread()
        save_thread(name, thread)
    return THREAD_REGISTRY[name]


def get_history(thread_id: str, limit: int = None) -> list[Message]:
    """
    Get the history of a thread
    """
    return Thread(id=thread_id).get_messages(limit=limit)


# File: src/controlflow/utilities/context.py
from marvin.utilities.context import ScopedContext

ctx = ScopedContext(
    dict(
        flow=None,
        tasks=[],
    )
)


# File: src/controlflow/utilities/user_access.py
def talk_to_human(message: str, get_response: bool = True) -> str:
    """
    Send a message to the human user and optionally wait for a response.
    If `get_response` is True, the function will return the user's response,
    otherwise it will return a simple confirmation.
    """
    print(message)
    if get_response:
        response = input("> ")
        return response
    return "Message sent to user."


# File: src/controlflow/utilities/marvin.py
import inspect
from contextlib import contextmanager
from typing import Any, Callable

import marvin.ai.text
from marvin.client.openai import AsyncMarvinClient
from marvin.settings import temporary_settings as temporary_marvin_settings
from openai.types.chat import ChatCompletion
from prefect import task as prefect_task

from controlflow.utilities.prefect import (
    create_json_artifact,
)

original_classify_async = marvin.classify_async
original_cast_async = marvin.cast_async
original_extract_async = marvin.extract_async
original_generate_async = marvin.generate_async
original_paint_async = marvin.paint_async
original_speak_async = marvin.speak_async
original_transcribe_async = marvin.transcribe_async


class AsyncControlFlowClient(AsyncMarvinClient):
    async def generate_chat(self, **kwargs: Any) -> "ChatCompletion":
        super_method = super().generate_chat

        @prefect_task(task_run_name="Generate OpenAI chat completion")
        async def _generate_chat(**kwargs):
            messages = kwargs.get("messages", [])
            create_json_artifact(key="prompt", data=messages)
            response = await super_method(**kwargs)
            create_json_artifact(key="response", data=response)
            return response

        return await _generate_chat(**kwargs)


def generate_task(name: str, original_fn: Callable):
    if inspect.iscoroutinefunction(original_fn):

        @prefect_task(name=name)
        async def wrapper(*args, **kwargs):
            create_json_artifact(key="args", data=[args, kwargs])
            result = await original_fn(*args, **kwargs)
            create_json_artifact(key="result", data=result)
            return result
    else:

        @prefect_task(name=name)
        def wrapper(*args, **kwargs):
            create_json_artifact(key="args", data=[args, kwargs])
            result = original_fn(*args, **kwargs)
            create_json_artifact(key="result", data=result)
            return result

    return wrapper


@contextmanager
def patch_marvin():
    with temporary_marvin_settings(default_async_client_cls=AsyncControlFlowClient):
        try:
            marvin.ai.text.classify_async = generate_task(
                "marvin.classify", original_classify_async
            )
            marvin.ai.text.cast_async = generate_task(
                "marvin.cast", original_cast_async
            )
            marvin.ai.text.extract_async = generate_task(
                "marvin.extract", original_extract_async
            )
            marvin.ai.text.generate_async = generate_task(
                "marvin.generate", original_generate_async
            )
            marvin.ai.images.paint_async = generate_task(
                "marvin.paint", original_paint_async
            )
            marvin.ai.audio.speak_async = generate_task(
                "marvin.speak", original_speak_async
            )
            marvin.ai.audio.transcribe_async = generate_task(
                "marvin.transcribe", original_transcribe_async
            )
            yield
        finally:
            marvin.ai.text.classify_async = original_classify_async
            marvin.ai.text.cast_async = original_cast_async
            marvin.ai.text.extract_async = original_extract_async
            marvin.ai.text.generate_async = original_generate_async
            marvin.ai.images.paint_async = original_paint_async
            marvin.ai.audio.speak_async = original_speak_async
            marvin.ai.audio.transcribe_async = original_transcribe_async
