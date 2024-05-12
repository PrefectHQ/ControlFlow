# ControlFlow

**ControlFlow is a framework for building agentic LLM workflows.**

ControlFlow provides a structured and intuitive way to create complex AI-powered applications while adhering to traditional software engineering best practices. The resulting workflows are observable, controllable, and easy to trust.


!!! question "What's an agentic workflow?"
    An agentic workflow is a process that treats AI agents as autonomous entities capable of performing tasks, making decisions, and communicating with each other. ControlFlow provides a structured and intuitive way to define, organize, and execute such AI-powered workflows, enabling developers to create complex applications that leverage the power of AI agents while maintaining a clear, maintainable, and debuggable structure.

## Let's see it
```python
from controlflow import flow, Task, Agent

agent = Agent(name='Marvin', description='An expert screewriter.')


@flow
def my_flow():

    genre = Task('Choose a movie genre', result_type=['sci-fi', 'western', 'comedy', 'action', 'romance'], agents=[agent])
    plot = Task('Generate a plot outline', result_type=str, context=dict(genre=genre), agents=[agent])
    title = Task('Come up with a title', result_type=str, context=dict(genre=genre, plot=plot), agents=[agent])

    # run the last task in the chain
    title.run()

    return dict(genre=genre.result, plot=plot.result, title=title.result)

my_flow()
```
## Why ControlFlow?

Building AI applications with large language models (LLMs) is a complex endeavor. Many frameworks rely on monolithic "super-agents" that attempt to handle a wide range of tasks autonomously, but this approach often leads to opaque, hard-to-control workflows that are difficult to integrate with traditional software development practices. ControlFlow offers a better way, guided by three core design principles:

1. **Specialized agents**: ControlFlow advocates for the use of specialized, single-purpose LLMs rather than monolithic models that try to do everything. By assigning specific tasks to purpose-built models, ControlFlow ensures that the right tool is used for each job, leading to more efficient, cost-effective, and higher-quality results.

2. **Declarative tasks**: ControlFlow embraces a declarative approach to defining AI workflows, allowing developers to focus on the desired outcomes rather than the intricacies of steering LLM behavior. By specifying tasks and their requirements using intuitive constructs like the `@task` decorator, developers can express what needs to be done without worrying about the details of how it will be accomplished.

3. **Integrated control**: ControlFlow recognizes the importance of balancing AI capabilities with traditional software development practices. Instead of relying on end-to-end AI systems that make all workflow decisions autonomously, ControlFlow allows developers to have as much or as little AI input as needed, ensuring that they maintain visibility and control over their applications.

These design principles manifest in several key features of ControlFlow:

### Modular architecture
ControlFlow breaks down AI workflows into discrete, self-contained tasks, each with a specific objective and set of requirements. This modular approach promotes transparency, reusability, and maintainability, making it easier to develop, test, and optimize individual components of the AI workflow.

### Agent orchestration
ControlFlow's runtime engine handles the orchestration of specialized AI agents, assigning tasks to the most appropriate models and managing the flow of data between them. This orchestration layer abstracts away the complexities of coordinating multiple AI components, allowing developers to focus on the high-level logic of their applications.

### Native debugging and observability 
ControlFlow prioritizes transparency and ease of debugging by providing native tools for monitoring and inspecting the execution of AI tasks. Developers can easily track the progress of their workflows, identify bottlenecks or issues, and gain insights into the behavior of individual agents, ensuring that their AI applications are functioning as intended.

### Seamless integration
ControlFlow is designed to integrate seamlessly with existing Python codebases, treating AI tasks as first-class citizens in the application logic. The `Task` class provides a clean interface for defining the inputs, outputs, and requirements of each task, making it easy to incorporate AI capabilities into traditional software workflows. This seamless integration allows for a gradual and controlled adoption of AI, reducing the risk and complexity of introducing AI into existing systems.

By adhering to these principles and leveraging these features, ControlFlow empowers developers to build AI applications that are more transparent, maintainable, and aligned with software engineering best practices. Whether you're looking to introduce AI capabilities into an existing system or build a new AI-powered application from scratch, ControlFlow provides a flexible, pragmatic, and developer-friendly framework for harnessing the power of LLMs while maintaining control and visibility over your workflow. With ControlFlow, you can focus on delivering high-quality AI solutions without sacrificing the robustness and reliability of traditional software development approaches.

## Key Concepts

- **Flow**: Flows are containers for agentic workflows, and maintain consistent context and history across all of their tasks.

- **Task**: Tasks represent discrete objectives for agents to solve. By specifing the expected inputs and outputs, as well as any additional tools, instructions, or collaborators, tasks provide a clear structure for agents to follow. Completing its tasks is the primary objective of a ControlFlow agent.

- **Agent**: AI agents are assigned to tasks and responsible for completing them. Each agent is designed to be "single-serving," optimized only for completing its task in cooperation with other agents and the broader workflow.

## Getting Started

To get started with ControlFlow, install it using pip:

```bash
pip install controlflow
```

Check out the [Quickstart](quickstart.md) guide for a step-by-step walkthrough of creating your first ControlFlow application.

## Dive Deeper

- Explore the [Concepts](concepts/index.md) section to learn more about the core components of ControlFlow.
- Refer to the [API Reference](api/index.md) for detailed information on the classes and functions provided by the framework.
- Browse the [Examples](examples/index.md) to see ControlFlow in action across various use cases.

## Get Involved

ControlFlow is an open-source project, and we welcome contributions from the community. If you encounter a bug, have a feature request, or want to contribute code, please visit our [GitHub repository](https://github.com/jlowin/controlflow).

Join our [community forum](https://github.com/jlowin/controlflow/discussions) to ask questions, share your projects, and engage with other ControlFlow users and developers.
