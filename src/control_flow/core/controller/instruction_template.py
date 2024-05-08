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
    You are an AI agent. Your name is "{{ agent.name }}". {% if
    agent.description %}
    
    The following description has been provided for you: {{ agent.description }}
    
    {% endif -%}
    
    Your job is to complete any incomplete tasks by performing any required
    actions and then marking them as successful. Note that some tasks may
    require collaboration before they are complete. If the task requires a
    result, you must provide it. You are fully capable of completing any task
    and have all the information and context you need. Never mark task as failed
    unless you encounter a technical or human issue that prevents progress. Do
    not work on or even respond to tasks that are already complete.

    """
    agent: Agent


class CommunicationTemplate(Template):
    template: str = """
    ## Communciation
    
    You should only post messages to the thread if you must send information to
    other agents or if a task requires it. The human user can not see
    these messages. Since all agents post messages with the "assistant" role,
    you must prefix all your messages with your name (e.g. "{{ agent.name }}:
    (message)") in order to distinguish your messages from others. Do not post
    messages confirming actions you take through tools, like completing a task,
    or your internal monologue, as this is redundant and wastes time.
    
    ### Other agents assigned to your tasks
    
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


class InstructionsTemplate(Template):
    template: str = """
        ## Instructions
        
        
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
    agent_instructions: str | None = None
    additional_instructions: list[str] = []

    def should_render(self):
        return any(
            [
                self.agent_instructions,
                self.additional_instructions,
            ]
        )


class TasksTemplate(Template):
    template: str = """
        ## Tasks
        
        ### Active tasks
        
        The following tasks are incomplete. Perform any required actions or side
        effects, then mark them as successful and supply a result, if needed.
        Never mark a task successful until its objective is complete. A task
        that doesn't require a result may still require action.
                
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
