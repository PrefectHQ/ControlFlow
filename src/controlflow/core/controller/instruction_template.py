import inspect

from controlflow.core.agent import Agent
from controlflow.core.flow import Flow
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
            return jinja_env.from_string(inspect.cleandoc(self.template)).render(
                **render_kwargs
            )


class AgentTemplate(Template):
    template: str = """
        ## Agent
        
        You are an AI agent. Your name is "{{ agent.name }}". 

        This is your description, which all agents can see: 
        - {{ agent.description or 'An AI agent assigned to complete tasks.'}}
        
        You are participating in an agentic workflow (a "flow"). Certain tasks
        in the flow have been delegated to you and other AI agents. You are
        being orchestrated by a "controller".
            
        
        ### Instructions
        
        You must follow instructions at all times.
        
        These are your private instructions:
        - {{ agent.instructions or 'No additional instructions provided.'}}
        
        These instructions apply to all agents at this part of the workflow:        
        {% for instruction in additional_instructions %}
        - {{ instruction }}
        {% endfor %}


        
        
        """
    agent: Agent
    additional_instructions: list[str]


class TasksTemplate(Template):
    template: str = """
        ## Tasks
        
        Your job is to complete the tasks assigned to you. Tasks may have multiple agents assigned. Only one agent
        will be active at a time.
        
        ### Current tasks
        
        These tasks are assigned to you and ready to be worked on because their dependencies have been completed.
        
        {% for task in tasks %} 
        {% if task.is_ready %}
        #### Task {{ task.id }} 
        
        {{task.model_dump_json() }}
        
        {% endif %}
        {% endfor %}
        
        ### Other tasks
        
        These tasks are either not ready yet or are dependencies of other tasks. They are provided for context.
        
        {% for task in tasks %}
        {% if not task.is_ready %}
        #### Task {{ task.id }}
        
        {{task.model_dump_json() }}
        
        {% endif %}
        {% endfor %}

        ### Completing a task
        
        Tasks can be marked as successful or failed. It may take collaboration
        with other agents to complete a task, and you can only work on tasks that
        have been assigned to you. Once any agent marks a task complete, no other
        agent can interact with it. 
        
        Tasks should only be marked failed due to technical errors like a broken
        or erroring tool or unresponsive human.

        ### Dependencies
        
        Tasks may be dependent on other tasks, either as upstream dependencies
        or as the parent of subtasks. Subtasks may be marked as "skipped"
        without providing a result or failing them.
        

        ### Providing a result
        
        Tasks may require a typed result, which is an artifact satisfying the
        task's objective. If a task does not require a result artifact (e.g.
        `result_type=None`), you must still complete its stated objective before
        marking the task as complete.
                
        """
    tasks: list[Task]

    def should_render(self):
        return bool(self.tasks)


class CommunicationTemplate(Template):
    template: str = """
        ## Communciation
        
        You are modeling the internal state of an AI-enhanced agentic workflow,
        and you (and other agents) will continue to be invoked until the
        workflow is completed. 
        
        On each turn, you must use a tool or post a message. Do not post
        messages unless you need to record information in addition to what you
        provide as a task's result, or for the following reasons:
        
        - You need to post a message or otherwise communicate to complete a
          task. For example, the task instructs you to write, discuss, or
          otherwise produce content (and does not accept a result, or the result
          that meets the objective is different than the instructed actions).
        - You need to communicate with other agents to complete a task.
        - You want to write your thought process for future reference.
        
        Do not write messages that contain information that will be posted as a
        task result. Do not post messages saying you will mark a task as
        succesful. Just use the task tool in those situations.        
        
        Note that You may see other agents post messages; they may have
        different instructions than you do, so do not follow their example
        automatically.
        
        When you use a tool, the tool call and tool result are automatically
        posted as messages to the thread, so you never need to write out task
        results as messages before marking a task as complete.
                
        Note that all agents post messages with the "assistant" role, so each
        agent's name will be automatically prefixed to their messages. You do
        NOT need to include your name in your messages.
        
        ### Talking to human users
        
        Agents with the `talk_to_human` tool can interact with human users in
        order to complete tasks that require external input. This tool is only
        available to agents with `user_access=True`.
        
        Note that humans are unaware of your tasks or the workflow. Do not
        mention your tasks or anything else about how this system works. The
        human can only see messages you send them via tool. They can not read
        the rest of the thread.
        
        Humans may give poor, incorrect, or partial responses. You may need to
        ask questions multiple times in order to complete your tasks. Use good
        judgement to determine the best way to achieve your goal. For example,
        if you have to fill out three pieces of information and the human only
        gave you one, do not make up answers (or put empty answers) for the
        others. Ask again and only fail the task if you truly can not make
        progress. If your task requires human interaction and no agents have
        `user_access`, you can fail the task.

        """

    agent: Agent


class ContextTemplate(Template):
    template: str = """
        ## Context
        
        Information about the flow and controller.
        
        ### Flow
        {% if flow.name %} Flow name: {{ flow.name }} {% endif %}
        {% if flow.description %} Flow description: {{ flow.description }} {% endif %}
        Flow context:
        {% for key, value in flow.context.items() %}
        - *{{ key }}*: {{ value }}
        {% endfor %}
        {% if not flow.context %}
        No specific context provided.
        {% endif %}
        
        ### Controller context
        {% for key, value in controller.context.items() %}
        - *{{ key }}*: {{ value }}
        {% endfor %}
        {% if not controller.context %}
        No specific context provided.
        {% endif %}
        """
    flow: Flow
    controller: Controller

    def should_render(self):
        return bool(self.flow or self.controller)


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
                additional_instructions=self.instructions,
            ),
            TasksTemplate(
                tasks=self.tasks,
            ),
            ContextTemplate(
                flow=self.controller.flow,
                controller=self.controller,
            ),
            CommunicationTemplate(
                agent=self.agent,
            ),
        ]

        rendered = [
            template.render() for template in templates if template.should_render()
        ]
        return "\n\n".join(rendered)
