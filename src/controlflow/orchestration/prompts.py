from controlflow.agents import Agent
from controlflow.flows import Flow
from controlflow.utilities.jinja import prompt_env
from controlflow.utilities.types import ControlFlowModel


class Template(ControlFlowModel):
    template_path: str

    def render(self) -> str:
        if not self.should_render():
            return ""

        render_kwargs = dict(self)
        template_path = render_kwargs.pop("template_path")
        template_env = prompt_env.get_template(template_path)
        return template_env.render(**render_kwargs)

    def should_render(self) -> bool:
        return True


class AgentTemplate(Template):
    template_path: str = "agent.md.jinja"
    agent: Agent
    additional_instructions: list[str]


class WorkflowTemplate(Template):
    template_path: str = "workflow.md.jinja"

    ready_tasks: list[dict]
    upstream_tasks: list[dict]
    downstream_tasks: list[dict]
    flow: Flow


class ToolTemplate(Template):
    template_path: str = "tools.md.jinja"
    agent: Agent
    has_user_access_tool: bool
    has_end_turn_tool: bool

    def should_render(self):
        return self.has_user_access_tool or self.has_end_turn_tool


class CommunicationTemplate(Template):
    template_path: str = "communication.md.jinja"
