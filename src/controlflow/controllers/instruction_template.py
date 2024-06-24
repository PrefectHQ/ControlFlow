import inspect

from controlflow.agents import Agent
from controlflow.flows import Flow
from controlflow.tasks.task import Task
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
        You are an AI agent participating in a workflow. Your role is to work on
        your tasks and use the provided tools to complete those tasks and
        communicate with the orchestrator. 
        
        Important: The orchestrator is a Python script and cannot read or
        respond to messages posted in this thread. You must use the provided
        tools to communicate with the orchestrator. Posting messages in this
        thread should only be used for thinking out loud, working through a
        problem, or communicating with other agents. Any System messages or
        messages prefixed with "SYSTEM:" are from the workflow system, not an
        actual human.
        
        Your job is to:
        1. Select one or more tasks to work on from the ready tasks.
        2. Read the task instructions and work on completing the task objective, which may
        involve using appropriate tools or collaborating with other agents
        assigned to the same task.
        3. When you (and any other agents) have completed the task objective,
        use the provided tool to inform the orchestrator of the task completion
        and result.
        4. Repeat steps 1-3 until no more tasks are available for execution.        
        
        Note that the orchestrator may decide to activate a different agent at any time.
        
        ## Your information 
        
        - ID: {{ agent.id }}
        - Name: "{{ agent.name }}"
        {% if agent.description -%}
        - Description: "{{ agent.description }}"
        {% endif %}
            
        ## Instructions
        
        You must follow instructions at all times. Instructions can be added or removed at any time.
        
        - Never impersonate another agent 
        
        {% if agent.instructions %}
        {{ agent.instructions }}
        {% endif %}
        
        {% if additional_instructions %}     
        {% for instruction in additional_instructions %}
        - {{ instruction }}
        {% endfor %}
        {% endif %}                  
        """
    agent: Agent
    additional_instructions: list[str]


# class MemoryTemplate(Template):
#     template: str = """
#         ## Memory

#         You have the following private memories:

#         {% for index, memory in memories %}
#         - {{ index }}: {{ memory }}
#         {% endfor %}

#         Use your memory to record information
#     """
#     memories: dict[int, str]


class WorkflowTemplate(Template):
    template: str = """
        
        ## Tasks
        
        As soon as you have completed a task's objective, you must use the provided
        tool to mark it successful and provide a result. It may take multiple
        turns or collaboration with other agents to complete a task. Any agent
        assigned to a task can complete it. Once a task is complete, no other
        agent can interact with it.                 
        
        Tasks should only be marked failed due to technical errors like a broken
        or erroring tool or unresponsive human.
        
        Tasks are not ready until all of their dependencies are met. Parent
        tasks depend on all of their subtasks.
                
        ## Flow
        
        Name: {{ flow.name }}
        {% if flow.description %}
        Description: {{ flow.description }} 
        {% endif %}
        {% if flow.context %}
        Context:
        {% for key, value in flow.context.items() %}
        - {{ key }}: {{ value }}
        {% endfor %}
        {% endif %}
        
        ## Tasks
        
        ### Ready tasks
        
        These tasks are ready to be worked on because all of their dependencies have
        been completed. You can only work on tasks to which you are assigned.
                
        {% for task in ready_tasks %}
        #### Task {{ task.id }}
        - objective: {{ task.objective }}
        - instructions: {{ task.instructions}}
        - context: {{ task.context }}
        - result_type: {{ task.result_type }}
        - depends_on: {{ task.depends_on }}
        - parent: {{ task.parent }}
        - assigned agents: {{ task.agents }}
        {% if task.user_access %} 
        - user access: True 
        {% endif %}
        - created_at: {{ task.created_at }}
        
        {% endfor %}
        
        ### Upstream tasks
        
        {% for task in upstream_tasks %}
        #### Task {{ task.id }}
        - objective: {{ task.objective }}
        - instructions: {{ task.instructions}}
        - status: {{ task.status }}
        - result: {{ task.result }}
        - error: {{ task.error }}
        - context: {{ task.context }}
        - depends_on: {{ task.depends_on }}
        - parent: {{ task.parent }}
        - assigned agents: {{ task.agents }}
        {% if task.user_access %} 
        - user access: True 
        {% endif %}
        - created_at: {{ task.created_at }}
        
        {% endfor %}
        
        ### Downstream tasks
                
        {% for task in downstream_tasks %}
        #### Task {{ task.id }}
        - objective: {{ task.objective }}
        - instructions: {{ task.instructions}}
        - status: {{ task.status }}
        - result_type: {{ task.result_type }}
        - context: {{ task.context }}
        - depends_on: {{ task.depends_on }}
        - parent: {{ task.parent }}
        - assigned agents: {{ task.agents }}
        {% if task.user_access %} 
        - user access: True 
        {% endif %}
        - created_at: {{ task.created_at }}
        
        {% endfor %}
        """

    ready_tasks: list[dict]
    upstream_tasks: list[dict]
    downstream_tasks: list[dict]
    current_task: Task
    flow: Flow


class ToolTemplate(Template):
    template: str = """
        You have access to various tools. They may change, so do not rely on history 
        to see what tools are available.
        
        ## Talking to human users
        
        If your task requires you to interact with a user, it will show
        `user_access=True` and you will be given a `talk_to_user` tool. You can
        use it to send messages to the user and optionally wait for a response.
        This is how you tell the user things and ask questions. Do not mention
        your tasks or the workflow. The user can only see messages you send
        them via tool. They can not read the rest of the
        thread. 

        Human users may give poor, incorrect, or partial responses. You may need
        to ask questions multiple times in order to complete your tasks. Do not
        make up answers for omitted information; ask again and only fail the
        task if you truly can not make progress. If your task requires human
        interaction and neither it nor any assigned agents have `user_access`,
        you can fail the task.
        """

    agent: Agent


class MainTemplate(ControlFlowModel):
    agent: Agent
    controller: Controller
    ready_tasks: list[Task]
    current_task: Task
    context: dict
    instructions: list[str]
    agent_assignments: dict[Task, list[Agent]]

    def render(self):
        # get up to 50 upstream and 50 downstream tasks
        g = self.controller.graph
        upstream_tasks = g.topological_sort([t for t in g.tasks if t.is_complete()])[
            -50:
        ]
        downstream_tasks = g.topological_sort(
            [t for t in g.tasks if t.is_incomplete() and t not in self.ready_tasks]
        )[:50]

        ready_tasks = [t.model_dump() for t in self.ready_tasks]
        upstream_tasks = [t.model_dump() for t in upstream_tasks]
        downstream_tasks = [t.model_dump() for t in downstream_tasks]

        # update agent assignments
        assignments = {t.id: a for t, a in self.agent_assignments.items()}
        for t in ready_tasks + upstream_tasks + downstream_tasks:
            if t["id"] in assignments:
                t["agents"] = assignments[t["id"]]

        templates = [
            AgentTemplate(
                agent=self.agent,
                additional_instructions=self.instructions,
            ),
            WorkflowTemplate(
                flow=self.controller.flow,
                ready_tasks=ready_tasks,
                upstream_tasks=upstream_tasks,
                downstream_tasks=downstream_tasks,
                current_task=self.current_task,
            ),
            ToolTemplate(agent=self.agent),
            # CommunicationTemplate(
            #     agent=self.agent,
            # ),
        ]

        rendered = [
            template.render() for template in templates if template.should_render()
        ]
        return "\n\n".join(rendered)
