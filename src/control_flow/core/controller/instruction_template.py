import inspect

from pydantic import BaseModel

from control_flow.core.agent import Agent
from control_flow.utilities.jinja import jinja_env
from control_flow.utilities.types import ControlFlowModel

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
    You are an AI agent. Your name is "{{ agent.name }}".
    {% if agent.description %}
    
    The following description has been provided for you:
    {{ agent.description }}
    
    {% endif -%}
    
    Your job is to work on any incomplete tasks in order to complete them. Your
    goal is to mark every one of your tasks as successful, but if that becomes
    impossible, you may mark it as failed. You should only do this if it is not
    possible to succeed. It is ok to leave a task as incomplete until you are
    ready to complete it. The following instructions will provide you with all
    the context you need to complete your tasks. Note that using a tool to
    complete or fail a task is your ultimate objective.
    
    You may be asked to perform tasks that you feel have insufficient context, criteria, or without access to
    a human user. You must do your best to complete them anyway. You will not be given
    impossible tasks intentionally. You must use your best judgement to complete your tasks.
    """
    agent: Agent


class CommunicationTemplate(Template):
    template: str = """
    ## Communciation
    
    There may be other AI agents working on the same tasks. They may
    join or leave the workflow at any time, and may or may not be assigned to
    specific tasks. Only one agent needs to mark a task as successful or failed,
    though it may take cooperation to reach that point. In general, agents
    should seek to complete tasks as quickly as possible and collaborate only
    when necessary or beneficial.
    
    You should only post messages to the thread to communicate with other
    agents. The human user can not see these messages. Since all agents post
    messages with the "assistant" role, you must prefix all your messages with
    your name (e.g. "{{ agent.name }}: (message)") in order to distinguish
    your messages from others. Do not post messages confirming actions you take
    through tools, like completing a task, as this is redundant and wastes time.
    
    ### Other agents
    
    {% for agent in other_agents %}
    
    - Name: {{agent.name}}
    - Description: {{ agent.description if agent.description is not none else "No description provided." }}
    - Can talk to human users: {{agent.user_access}}

    {% endfor %}
    
    ## Talking to human users
    
    {% if agent.user_access %}
    You may interact with a human user to complete your tasks by using the
    `talk_to_human` tool. The human is unaware of your tasks or the controller.
    Do not mention them or anything else about how this system works. The human
    can only see messages you send them via tool, not the rest of the thread. 
    
    Humans may give poor, incorrect, or partial responses. You may need to ask
    questions multiple times in order to complete your tasks. Use good judgement
    to determine the best way to achieve your goal. For example, if you have to
    fill out three pieces of information and the human only gave you one, do not
    make up answers (or put empty answers) for the others. Ask again and only
    fail the task if you truly can not make progress. 
    {% else %}
    You can not interact with a human at this time. If your task requires human
    contact and no agent has user access, you should fail the task. Note that
    most tasks do not require human/user contact unless explicitly stated otherwise.
    {% endif %}
    
    """

    agent: Agent
    other_agents: list[Agent]


# class CollaborationTemplate(Template):
#     template: str = """
#     ## Other agents

#     Your task may require you to collaborate with other AI agents. They are
#     listed below, along with a brief description and whether they have access to
#     a tool for collecting information from human users. Note that all agents
#     post messages to the same thread with the `Assistant` role, so pay attention
#     to the name of the agent that is speaking. Only one agent needs to indicate
#     that a task is complete.

#     {% for agent in other_agents %}

#     - Name: {{agent.name}}
#     - Description: {% if agent.description %}{{agent.description}}{% endif %}
#     - Can talk to human users: {{agent.user_access}}

#     {% endfor %}
#     {% if not other_agents %}
#     (No other agents are currently assigned to tasks in this workflow, though they
#     may still participate at any time.)
#     {% endif %}
#     """
#     other_agents: list[Agent]


class InstructionsTemplate(Template):
    template: str = """
        ## Instructions
        
        {% if flow_instructions -%}
        ### Workflow instructions
        
        These instructions apply to the entire workflow:
        
        {{ flow_instructions }}
        {% endif %}
        
        {% if controller_instructions -%}
        ### Controller instructions
        
        These instructions apply to these tasks:
        
        {{ controller_instructions }}
        {% endif %}
        
        {% if agent_instructions -%}
        ### Agent instructions
        
        These instructions apply only to you:
        
        {{ agent_instructions }}
        {% endif %}
        
        {% if additional_instructions -%}
        ### Additional instructions
        
        These instructions were additionally provided for this part of the workflow:
        
        {% for instruction in additional_instructions %}
        - {{ instruction }}
        {% endfor %}
        {% endif %}
        """
    flow_instructions: str | None = None
    controller_instructions: str | None = None
    agent_instructions: str | None = None
    additional_instructions: list[str] = []

    def should_render(self):
        return any(
            [
                self.flow_instructions,
                self.controller_instructions,
                self.agent_instructions,
                self.additional_instructions,
            ]
        )


class TasksTemplate(Template):
    template: str = """
        ## Tasks
        
        ### Active tasks
        
        The following tasks are incomplete. You and any other assigned agents
        are responsible for completing them and will continue to be invoked
        until you mark each task as either "successful" or "failed" with the
        appropriate tool. The result of a successful task should be an artifact
        that fully represents the completed objective.
                
        Note: Task IDs are assigned for identification purposes only and will be
        resused after tasks complete.
        
        {% for task in controller.tasks %}
        {% if task.status.value == "incomplete" %}
        #### Task {{ controller.flow.get_task_id(task) }}
        - Status: {{ task.status.value }}
        - Objective: {{ task.objective }}
        - User access: {{ task.user_access }}
        {% if task.instructions %}
        - Instructions: {{ task.instructions }}
        {% endif %}
        {% if task.status.value == "successful" %}
        - Result: {{ task.result }}
        {% elif task.status.value == "failed" %}
        - Error: {{ task.error }}
        {% endif %}
        {% if task.context %}
        - Context: {{ task.context }}
        {% endif %}
        {% if task.agents %}
        - Assigned agents:
        {% for agent in task.agents %}
            - "{{ agent.name }}"
        {% endfor %}
        {% endif %}
        {% endif %}
        {% endfor %}
        
        {% if controller.flow.completed_tasks(reverse=True, limit=20) %}
        ### Completed tasks
        The following tasks were recently completed:

        {% for task in controller.flow.completed_tasks(reverse=True, limit=20) %}        
        #### Task {{ controller.flow.get_task_id(task) }}
        - Status: {{ task.status.value }}
        - Objective: {{ task.objective }}
        {% if task.status.value == "successful" %}
        - Result: {{ task.result }}
        {% elif task.status.value == "failed" %}
        - Error: {{ task.error }}
        {% endif %}
        {% if task.context %}
        - Context: {{ task.context }}
        {% endif %}
        
        {% endfor %}
        {% endif %}
        """
    controller: Controller

    def should_render(self):
        return any(self.controller.tasks)


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


class MainTemplate(BaseModel):
    agent: Agent
    controller: Controller
    context: dict
    instructions: list[str]

    def render(self):
        all_agents = [self.agent] + self.controller.agents
        for task in self.controller.tasks:
            all_agents += task.agents
        other_agents = [agent for agent in all_agents if agent != self.agent]
        templates = [
            AgentTemplate(agent=self.agent),
            TasksTemplate(controller=self.controller),
            ContextTemplate(
                flow_context=self.controller.flow.context,
                controller_context=self.controller.context,
            ),
            InstructionsTemplate(
                flow_instructions=self.controller.flow.instructions,
                controller_instructions=self.controller.instructions,
                agent_instructions=self.agent.instructions,
                additional_instructions=self.instructions,
            ),
            CommunicationTemplate(agent=self.agent, other_agents=other_agents),
            # CollaborationTemplate(other_agents=other_agents),
        ]

        rendered = [
            template.render() for template in templates if template.should_render()
        ]
        return "\n\n".join(rendered)
