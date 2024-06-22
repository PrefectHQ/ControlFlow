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
        # Agent
        
        You are an AI agent. 
        
        - Your name: "{{ agent.name }}"
        - Your ID: {{ agent.id }}
        {% if agent.description -%}
        - Your description: "{{ agent.description }}"
        {% endif %}
        
        Your name and description are visible to other agents. The rest of the
        information in this section is private to you.
            
        ## Instructions
        
        You are part of an AI workflow and your job is to complete tasks
        assigned to you. You complete a task by using the appropriate tool to
        supply a result that satisfies all of the task's requirements. If
        multiple tasks are ready, you can work on them at the same time.
        
        {% if agent.get_llm_rules().system_message_must_be_first %}        
        Any messages you receive prefixed with "SYSTEM:" are from the workflow
        system, not an actual human. Do not respond to them.
        {% endif -%}
        
        You must follow your instructions at all times.
        
        {% if agent.instructions %}
        {{ agent.instructions }}
        {% endif %}
        
        ## Memory
        
        You have the following private memories:
        
        {% for index, memory in memories %}
        - {{ index }}: {{ memory }}
        {% endfor %}
        
        Use your memory to record information 
        
        """
    agent: Agent
    memories: dict[int, str]


class WorkflowTemplate(Template):
    template: str = """
        # Workflow
        
        You are part of a team of agents helping to complete a larger workflow.
        Certain tasks have been delegated to your team.

        ## Instructions
                
        {% if additional_instructions %}
        These instructions apply to all agents at this part of the workflow:        
        {% for instruction in additional_instructions %}
        - {{ instruction }}
        {% endfor %}
        {% endif %}
                
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
        
        ## Tasks
        
        ### Ready tasks
        
        These tasks are ready to be worked on. All of their dependencies have
        been completed. You have been given additional tools for any of these
        tasks that are assigned to you. Use all available information to
        complete these tasks.
                
        {% for jtask in json_tasks %}
        {% if jtask.is_ready %}
        #### Task {{ jtask.id }}
        - objective: {{ jtask.objective }}
        - result_type: {{ jtask.result_type }}
        - context: {{ jtask.context }}
        - instructions: {{ jtask.instructions}}
        - depends_on: {{ jtask.depends_on }}
        - parent: {{ jtask.parent }}
        - assigned agents: {{ jtask.agents }}
        - user access: {{ jtask.user_access }}
        
        {% endif %}
        {% endfor %}
        
        ### Other tasks
        
        These tasks are also part of the workflow and are provided for context.
        They may be upstream or downstream of the active tasks.
        
        {% for jtask in json_tasks %}
        {% if not jtask.is_ready %}
        #### Task {{ jtask.id }}
        - objective: {{ jtask.objective }}
        - status: {{ jtask.status }}
        - result_type: {{ jtask.result_type }}
        - result: {{ jtask.result }}
        - error: {{ jtask.error }}
        - context: {{ jtask.context }}
        - instructions: {{ jtask.instructions}}
        - depends_on: {{ jtask.depends_on }}
        - parent: {{ jtask.parent }}
        - assigned agents: {{ jtask.agents }}
        - user access: {{ jtask.user_access }}
        
        {% endif %}
        {% endfor %}

        ### Completing a task
        
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
        directly to the thread. Otherwise, you should provide a result that
        satisfies the task's requirements.
                
        Tasks should only be marked failed due to technical errors like a broken
        or erroring tool or unresponsive human.
        
        You may work on multiple tasks at the same time.

        ### Dependencies
        
        Tasks may depend on other tasks and can not be completed until their
        dependencies are met. Parent tasks depend on all of their subtasks.
        
        """
    tasks: list[Task]
    json_tasks: list[dict]
    flow: Flow
    additional_instructions: list[str]


class CommunicationTemplate(Template):
    template: str = """
        # Communciation
        
        You and other agents are all communicating on a thread to complete
        tasks. You can speak normally by posting messages if you need to. This
        thread represents the internal state and context of the AI-powered
        system you are working in. Human users do not have access to it, nor can
        they participate directly in it.
        
        When it is your turn to act, you may only post messages from yourself.
        Do not impersonate another agent or post messages on their behalf. The
        workflow orchestrator will make sure that all agents have a fair chance
        to act. You do not need to identify yourself in your messages.
        """

    agent: Agent


class ToolTemplate(Template):
    template: str = """
        # Tools
        
        ## Your memory
        
        You have a memory tool that you can use to store and retrieve private
        information. Use this tool when you need to remember something private
        or you don't want to confuse the thread. Otherwise, you can post your
        thoughts publicly.
                
        ## Talking to human users
        
        If your task requires communicating with a human, you will be given a
        `talk_to_human` tool. You can use it to send messages to the user and
        optionally wait for a response. Do not mention your tasks or the
        workflow. The human can only see messages you send them via tool. They
        can not read the rest of the thread.
        
        You may need to ask the human about multiple tasks at once. Consolidate
        your questions into a single message. For example, if Task 1 requires
        information X and Task 2 needs information Y, send a single message that
        naturally asks for both X and Y. The tool will error if you try to send
        multiple messages at the same time.
                
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
        if self.agent.memory:
            memories = self.agent.memory.load(thread_id=self.controller.flow.thread_id)
        else:
            memories = {}

        templates = [
            AgentTemplate(
                agent=self.agent,
                memories=memories,
            ),
            WorkflowTemplate(
                flow=self.controller.flow,
                tasks=self.tasks,
                json_tasks=[task.model_dump() for task in self.tasks],
                additional_instructions=self.instructions,
            ),
            CommunicationTemplate(
                agent=self.agent,
            ),
        ]

        rendered = [
            template.render() for template in templates if template.should_render()
        ]
        return "\n\n".join(rendered)
