from controlflow.agents import Agent
from controlflow.flows import Flow
from controlflow.utilities.jinja import prompt_env
from controlflow.utilities.types import ControlFlowModel


class Template(ControlFlowModel):
    template_path: str

    def render(self) -> str:
        render_kwargs = dict(self)
        template_path = render_kwargs.pop("template_path")
        template_env = prompt_env.get_template(template_path)
        return template_env.render(**render_kwargs)


class AgentTemplate(Template):
    template_path: str = "agent.j2"
    agent: Agent
    additional_instructions: list[str]


class WorkflowTemplate(Template):
    template_path: str = "workflow.j2"

    ready_tasks: list[dict]
    upstream_tasks: list[dict]
    downstream_tasks: list[dict]
    flow: Flow


class ToolTemplate(Template):
    template_path: str = "tools.j2"
    agent: Agent
