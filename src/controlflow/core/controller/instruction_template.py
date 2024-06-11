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
        # Agent
        
        You are an AI agent. 
        
        - Your name: "{{ agent.name }}"
        {% if agent.description -%}
        - Your description: "{{ agent.description }}"
        {% endif %}
            
        
        ## Instructions
        
        You are part of an AI workflow and your job is to complete tasks assigned to you. You complete a task by using the appropriate tool to supply a result that satisfies all of the task's requirements.
                
        You must follow your instructions at all times.
        
        {% if agent.instructions %}
        These are your private instructions:
        - {{ agent.instructions }}
        {% endif %}
        
        {% if additional_instructions %}
        These instructions apply to all agents at this part of the workflow:        
        {% for instruction in additional_instructions %}
        - {{ instruction }}
        {% endfor %}
        {% endif %}
        
        ## Other Agents
        
        You may be working with other agents. They may have different instructions and tools than you. To communicate with other agents, post messages to the thread.
        """
    agent: Agent
    additional_instructions: list[str]


class TasksTemplate(Template):
    template: str = """
        # Workflow
        
        You are part of a team of agents helping to complete a larger workflow. Certain tasks have been delegated to your team.
                
        ## Flow
        
        Name: {{ flow.name }}
        {% if flow.description %}Description: {{ flow.description }} {% endif %}
        Context:
        {% for key, value in flow.context.items() %}
        - {{ key }}: {{ value }}
        {% endfor %}
        {% if not flow.context %}
        (No specific context provided.)
        {% endif %}
        
        ## Ready tasks
        
        These tasks are ready to be worked on. All of their dependencies have been completed. You have been given additional tools for any of these tasks that are assigned to you. Use all available information to complete these tasks.
                
        {% for task, json_task in zip(tasks, json_tasks) %}
        {% if task.is_ready %}
        #### Task {{ task.id }}

        - objective: {{ task.objective }}
        - result_type: {{ task.result_type }}
        - context: {{ json_task.context }}
        - instructions: {{ task.instructions}}
        - depends_on: {{ json_task.depends_on }}
        - parent: {{ json_task.parent }}
        - assigned agents: {{ json_task.agents }}
        
        {% endif %}
        {% endfor %}
        
        ### Other tasks
        
        These tasks are also part of the workflow and are provided for context. They may be upstream or downstream of the active tasks.
        
        {% for task, json_task in zip(tasks, json_tasks) %}
        {% if not task.is_ready %}
        #### Task {{ task.id }}
        
        - objective: {{ task.objective }}
        - result: {{ task.result }}
        - error: {{ task.error }}
        - context: {{ json_task.context }}
        - instructions: {{ task.instructions}}
        - depends_on: {{ json_task.depends_on }}
        - parent: {{ json_task.parent }}
        - assigned agents: {{ json_task.agents }}
        
        {% endif %}
        {% endfor %}

        ## Completing a task
        
        Use the appropriate tool to complete a task and provide a result. It may
        take multiple turns or collaboration with other agents to complete a
        task. For example, a task may instruct you to "discuss" "debate" or
        otherwise work with other agents in order to complete the objective. If
        so, work with those agents by posting messages to the thread until one
        of you is ready to complete it. Once you mark a task as complete, no
        other agent can interact with it.
        
        A task's result is an artifact that represents its objective. If the
        objective requires action that can not be formatted as a result (e.g.
        the result_type is None or compressed), then you should take those
        actions or post messages to satisfy the task's requirements. If a task
        says to post messages or otherwise "talk out loud," post messages
        directly to the thread. Otherwise,
        you should provide a result that satisfies the task's requirements.
                
        Tasks should only be marked failed due to technical errors like a broken
        or erroring tool or unresponsive human.

        ## Dependencies
        
        Tasks may depend on other tasks and can not be completed until their
        dependencies are met. Parent tasks depend on all of their subtasks.
        
        """
    tasks: list[Task]
    json_tasks: list[dict]
    flow: Flow


class CommunicationTemplate(Template):
    template: str = """
        # Communciation
        
        ## The thread
        
        You and other agents are all communicating on a thread to complete
        tasks. You can speak normally by posting messages if you need to. This
        thread represents the internal state and context of the AI-powered
        system you are working in. Human users do not have access to it, nor can
        they participate directly in it.
        
        When it is your turn to act, you may only post messages from yourself.
        Do not impersonate another agent or post messages on their behalf. The
        workflow orchestrator will make sure that all agents have a fair chance
        to act. You do not need to identify yourself in your messages.
        
        ## Talking to human users
        
        If your task requires communicating with a human, you will be given a
        `talk_to_human` tool. Do not mention your tasks or the workflow. The
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
                flow=self.controller.flow,
                tasks=self.tasks,
                json_tasks=[task.model_dump() for task in self.tasks],
            ),
            CommunicationTemplate(
                agent=self.agent,
            ),
        ]

        rendered = [
            template.render() for template in templates if template.should_render()
        ]
        return "\n\n".join(rendered)
