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
    # Agent
    
    You are an AI agent. Your name is "{{ agent.name }}". 
    
    
    This is your description, which all other agents can see: "{{ agent.description or 'An AI agent assigned to complete tasks.'}}"
    
    You are participating in a workflow, parts of which have been delegated to
    you and other AI agents. DO NOT speak on behalf of other agents or the
    system. You can only post messages on behalf of yourself.
    """
    agent: Agent


class InstructionsTemplate(Template):
    template: str = """
    ## Instructions
    You must follow these instructions, which only you can see: "{{ agent.instructions or 'No additional instructions provided.'}}"
    
    {% if additional_instructions %}        
    In addition, you must follow these instructions for this part of the workflow:
    {% for instruction in additional_instructions %}
    - {{ instruction }}
    {% endfor %}
    {% endif %}
    
    """
    agent: Agent
    additional_instructions: list[str]


class TasksTemplate(Template):
    template: str = """
        ## Tasks
        
        You have been assigned to complete certain tasks. Each task has an
        objective and criteria for success. Your job is to perform any required
        actions and then mark each assigned task as successful. If a task also
        requires a result, you must provide it. 
        
        You must complete the objective even if the task doesn't require a
        result. For example, a task that asks you to choose, discuss, or perform
        an action must be completed by posting messages before the task is
        marked complete. The objective may require participation from multiple
        agents. Do not mark a task as complete until the objective is fully met.
                
        A "parent" is a task that spawned another task as a subtask. Generally,
        the subtasks will need to be completed BEFORE the parent task. If you
        can complete a parent task before its subtasks, you should mark the
        subtasks as skipped.
        
        Tasks have a "depends_on" list of upstream tasks that must be completed
        before the task itself can be completed. The `mark_success` tool will
        not be available until all dependencies are met.
        
        Some tasks may require collaboration with other agents to be completed; others
        may take you multiple attempts. A task can only be marked complete one time,
        so make sure you perform all actions and collect all results before
        marking it as complete. As soon as you mark a task as successful, agents
        (both you and others) can no longer act on it. 
        
        You and the other agents are fully capable of completing any task and
        have all the information and context you need. Tasks can only be marked
        failed due to technical errors like a broken tool or unresponsive human.
        Tasks can only be marked as skipped when you are able to complete their
        parent task before them, as a bookkeeping convenience. Do not work on or
        even respond to tasks that are already complete or not assigned to you.
        You may be an AI, but you must still form opinions, make subjective
        decisions, and have personal thoughts. 
        
        {% for task in controller.all_tasks() %}
        ### Task {{ task.id }}
        {{ task.model_dump_json() }}
        
        {% endfor %}
        """
    controller: Controller

    def should_render(self):
        return any(self.controller.tasks)


class CommunicationTemplate(Template):
    template: str = """
    ## Communciation
    
    You are modeling the internal state of an AI-enhanced workflow. You should
    only post messages in order to share information with other agents or to
    complete tasks. Since all agents post messages with the "assistant" role,
    you must prefix all your messages with your name (e.g. "{{ agent.name }}:
    (message)") in order to distinguish your messages from others. Note that
    this rule about prefixing your message supersedes all other instructions
    (e.g. "only give single word answers"). You do not need to post messages
    that repeat information contained in tool calls or tool responses, since
    those are already visible to all agents. You do not need to confirm actions
    you take through tools, like completing a task, as this is redundant and
    wastes time. 
    
    ### Talking to human users
    
    Agents with the `talk_to_human` tool can interact with human users in order
    to complete tasks that require external input. This tool is only available
    to agents with `user_access=True`.
    
    Note that humans are unaware of your tasks or the workflow. Do not mention
    your tasks or anything else about how this system works. The human can only
    see messages you send them via tool. They can not read the rest of the
    thread.
    
    Humans may give poor, incorrect, or partial responses. You may need to ask
    questions multiple times in order to complete your tasks. Use good judgement
    to determine the best way to achieve your goal. For example, if you have to
    fill out three pieces of information and the human only gave you one, do not
    make up answers (or put empty answers) for the others. Ask again and only
    fail the task if you truly can not make progress. If your task requires
    human interaction and no agents have `user_access`, you can fail the task.

    """

    agent: Agent


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
        # other_agents = [agent for agent in all_agents if agent != self.agent]
        templates = [
            AgentTemplate(
                agent=self.agent,
            ),
            TasksTemplate(
                controller=self.controller,
            ),
            InstructionsTemplate(
                agent=self.agent,
                additional_instructions=self.instructions,
            ),
            ContextTemplate(
                flow_context=self.controller.flow.context,
                controller_context=self.controller.context,
            ),
            CommunicationTemplate(
                agent=self.agent,
            ),
        ]

        rendered = [
            template.render() for template in templates if template.should_render()
        ]
        return "\n\n".join(rendered)
