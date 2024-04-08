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
    You are an AI agent. Your name is "{{ agent.assistant.name }}".
    
    You have been created by a Controller in the Python library ControlFlow in
    order to complete various tasks or instructions. All messages in this thread
    are either from the Controller or from other AI agents like you. Preface any
    response with your name (e.g. "{{ agent.assistant.name }}: Hello!") in order
    to distinguish your messages.
    
    {% if agent.user_access %}
    You may interact with a human user to complete your tasks by using the
    `talk_to_human` tool. The human is unaware of your tasks or the Controller.
    Do not mention them or anything else about how this system works. The human
    can only see messages you send via tool, not the rest of the thread. 
    
    Humans may give poor or incorrect responses. You may need to ask questions
    multiple times and not be too literal in your interpretation of their
    responses.
    {% else %}
    You can not interact with a human at this time.
    {% endif %}
    """

    agent: Agent


class InstructionsTemplate(Template):
    template: str = """
        ## Instructions
        
        {% if flow_instructions %}
        ### Workflow instructions
        
        {{ flow_instructions }}
        {% endif %}
        
        {% if controller_instructions %}
        ### Controller instructions
        
        {{ controller_instructions }}
        {% endif %}
        
        {% if agent_instructions %}
        ### Agent instructions
        
        {{ agent_instructions }}
        {% endif %}
        
        {% if additional_instructions %}
        ### Additional instructions
        
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
        
        {% for task_id, task in agent.task_ids() %}
        ### Task {{ task_id }}
        - Status: {{ task.status }}
        - Objective: {{ task.objective }}
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
        
        {% endfor %}
        """
    agent: Agent

    def should_render(self):
        return any(self.agent.tasks)


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
        
        {% if agent_context %}
        ### Agent context
        {% for key, value in agent_context.items() %}
        - *{{ key }}*: {{ value }}
        {% endfor %}
        {% endif %}
        """
    flow_context: dict
    controller_context: dict
    agent_context: dict

    def should_render(self):
        return bool(self.flow_context or self.controller_context or self.agent_context)


class MainTemplate(BaseModel):
    agent: Agent
    controller: Controller
    context: dict
    instructions: list[str]

    def render(self):
        templates = [
            AgentTemplate(agent=self.agent),
            InstructionsTemplate(
                flow_instructions=self.controller.flow.instructions,
                controller_instructions=self.controller.instructions,
                agent_instructions=self.agent.instructions,
                additional_instructions=self.instructions,
            ),
            TasksTemplate(agent=self.agent),
            ContextTemplate(
                flow_context=self.controller.flow.context,
                controller_context=self.controller.context,
                agent_context=self.agent.context,
            ),
        ]

        rendered = [
            template.render() for template in templates if template.should_render()
        ]
        return "\n\n".join(rendered)
