# ControlFlow

**ControlFlow is a framework for building agentic LLM workflows.**

ControlFlow provides a structured and intuitive way to create complex AI-powered applications while adhering to traditional software engineering best practices. The resulting workflows are observable, controllable, and easy to trust.


!!! question "What's an agentic workflow?"
    An agentic workflow treats LLMs as autonomous entities capable of making decisions and performing complex tasks through iterative interactions. At least some of the workflow logic is carried out by the LLMs themselves.


## Design principles
ControlFlow's design is informed by a strong opinon: LLMs are powerful tools, but they are most effective when applied to small, well-defined tasks within a structured workflow.

This belief leads to three core design principles that underpin ControlFlow's architecture:

### 🛠️ Specialized over generalized
ControlFlow believes that **single-purpose agents dedicated to a specific tasks** will be more effective than monolithic models that attempt to do everything. By assigning specific tasks to purpose-built models, developers can always ensure that the right tool is used for each job, leading to more efficient, cost-effective, and higher-quality results.

### 🎯 Outcome over process
ControlFlow defines AI workflows in terms of desired outcomes rather than writing prompts to steer LLM behavior. This **declarative, task-centric approach** lets developers focus on what needs to be done while letting the framework orchestrate agents to achieve those outcomes.

### 🎛️ Control over autonomy
ControlFlow views agentic workflows as an extension of traditional software development practices. Instead of relying on end-to-end AI systems that make all workflow decisions autonomously, ControlFlow is **explicit by default**, allowing developers to delegate only as much work to AI as they require. This ensures that developers maintain visibility and control over their applications, as well as their preferred methods of testing and debugging.



## Why ControlFlow?
The three design principles of ControlFlow lead to a number of key features that make it a powerful tool for building AI-powered applications:

### 🧩 Task-centric architecture
ControlFlow breaks down AI workflows into discrete, self-contained tasks, each with a specific objective and set of requirements. This declarative, modular approach lets developers focus on the high-level logic of their applications while allowing the framework to manage the details of coordinating agents and data flow between tasks.

### 🕵️ Agent orchestration
ControlFlow's runtime engine handles the orchestration of specialized AI agents, assigning tasks to the most appropriate models and managing the flow of data between them. This orchestration layer abstracts away the complexities of coordinating multiple AI components, allowing developers to focus on the high-level logic of their applications.

### 🔍 Native debugging and observability 
ControlFlow prioritizes transparency and ease of debugging by providing native tools for monitoring and inspecting the execution of AI tasks. Developers can easily track the progress of their workflows, identify bottlenecks or issues, and gain insights into the behavior of individual agents, ensuring that their AI applications are functioning as intended.

### 🤝 Seamless integration
ControlFlow is designed to integrate seamlessly with existing Python codebases, treating AI tasks as first-class citizens in the application logic. The `Task` class provides a clean interface for defining the inputs, outputs, and requirements of each task, making it easy to incorporate AI capabilities into traditional software workflows. This seamless integration allows for a gradual and controlled adoption of AI, reducing the risk and complexity of introducing AI into existing systems.


## Key concepts

### 🌊 Flow
Flows are containers for agentic workflows, and maintain consistent context and history across all of their tasks.

### 🚦 Task
Tasks represent discrete objectives for agents to solve. By specifing the expected inputs and outputs, as well as any additional tools, instructions, or collaborators, tasks provide a clear structure for agents to follow. Completing its tasks is the primary objective of a ControlFlow agent.

### 🤖 Agent
AI agents are assigned to tasks and responsible for completing them. Each agent is designed to be "single-serving," optimized only for completing its task in cooperation with other agents and the broader workflow.


## Why not "super-agents"?

Many agentic LLM frameworks rely on monolithic "super-agents": powerful, unconstrained models that are expected to achieve their goals by autonomously handling a wide range of tasks, tools, and behaviors. The resulting workflows are opaque, unpredictable, and difficult to debug.

This approach naively assumes that the technology is more advanced than it actually is. LLMs feel like magic because they can perform a wide variety of non-algorithmic tasks, but they are still fundamentally limited when it comes to generalizing beyond their traning data and techniques. Moreover, the failure modes of agentic LLMs are difficult to identify, let alone fix, making them difficult to trust in production environments or with mission-critical tasks.
## Getting started

To get started with ControlFlow, install it using pip:

```bash
pip install controlflow
```

Check out the [Quickstart](quickstart.md) guide for a step-by-step walkthrough of creating your first ControlFlow application.

## Dive deeper

- Explore the [Concepts](concepts/index.md) section to learn more about the core components of ControlFlow.
- Refer to the [API Reference](api/index.md) for detailed information on the classes and functions provided by the framework.
- Browse the [Examples](examples/index.md) to see ControlFlow in action across various use cases.

## Get involved

ControlFlow is an open-source project, and we welcome contributions from the community. If you encounter a bug, have a feature request, or want to contribute code, please visit our [GitHub repository](https://github.com/jlowin/controlflow).

Join our [community forum](https://github.com/jlowin/controlflow/discussions) to ask questions, share your projects, and engage with other ControlFlow users and developers.
