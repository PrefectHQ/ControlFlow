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
    
    Your job is to work on any pending tasks until you can mark them as either
    `complete` or `failed`. The following instructions will provide you with all
    the context you need to complete your tasks. Note that using a tool to
    complete or fail a task is your ultimate objective, and you should not post
    any messages to the thread unless you have a specific reason to do so.
    """
    agent: Agent


class CommunicationTemplate(Template):
    template: str = """
    ## Communciation
    
    ### Posting messages to the thread
    
    You have been created by a Controller in the Python library ControlFlow in
    order to complete various tasks or instructions. All messages in this thread
    are either from the controller or from AI agents like you. Note that all
    agents post to the thread with the `Assistant` role, so if you do need to
    post a message, preface with your name (e.g. "{{ agent.name }}: Hello!") in
    order to distinguish your messages. 
    
    The controller CAN NOT and WILL NOT read your messages, so DO NOT post
    messages unless you need to send information to another agent. DO NOT post
    messages about information already captured by your tool calls, such as the
    tool call itself, its result, human responses, or task completion. 
    
    ### Talking to humans
    
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
    contact and no agent has user access, you should fail the task.
    {% endif %}
    
    """

    agent: Agent


class CollaborationTemplate(Template):
    template: str = """
    ## Collaboration
    
    You are collaborating with other AI agents. They are listed below by name,
    along with a brief description. Note that all agents post messages to the
    same thread with the `Assistant` role, so pay attention to the name of the
    agent that is speaking. Only one agent needs to indicate that a task is
    complete.
    
    ### Agents
    {% for agent in other_agents %}
    
    #### "{{agent.name}}"
    Can talk to humans: {{agent.user_access}}
    Description: {% if agent.description %}{{agent.description}}{% endif %}
    
    {% endfor %}
    {% if not other_agents %}
    (No other agents are currently participating in this workflow)
    {% endif %}
    """
    other_agents: list[Agent]


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
        
        The following tasks are pending. You and any other agents are responsible
        for completing them and will continue to be invoked until you mark each
        task as either "completed" or "failed" with the appropriate tool. The
        result of a complete task should be an artifact that fully represents
        the completed objective.
                
        Note: Task IDs are assigned for identification purposes only and will be
        resused after tasks complete.
        
        {% for task in controller.tasks %}
        {% if task.status.value == "pending" %}
        #### Task {{ controller.flow.get_task_id(task) }}
        - Status: {{ task.status.value }}
        - Objective: {{ task.objective }}
        - User access: {{ task.user_access }}
        {% if task.instructions %}
        - Instructions: {{ task.instructions }}
        {% endif %}
        {% if task.status.value == "completed" %}
        - Result: {{ task.result }}
        {% elif task.status.value == "failed" %}
        - Error: {{ task.error }}
        {% endif %}
        {% if task.context %}
        - Context: {{ task.context }}
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
        {% if task.status.value == "completed" %}
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
        ## Context
        
        {% if flow_context %}
        ### Flow context
        {% for key, value in flow_context.items() %}
        - *{{ key }}*: {{ value }}
        {% endfor %}
        {% endif %}
        
        {% if controller_context %}
        ### Controller context
        {% for key, value in controller_context.items() %}
        - *{{ key }}*: {{ value }}
        {% endfor %}
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
            CommunicationTemplate(agent=self.agent),
            CollaborationTemplate(
                other_agents=[a for a in self.controller.agents if a != self.agent]
            ),
        ]

        rendered = [
            template.render() for template in templates if template.should_render()
        ]
        return "\n\n".join(rendered)
