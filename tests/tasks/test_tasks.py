import controlflow
import pytest
from controlflow.agents import Agent
from controlflow.flows import Flow
from controlflow.instructions import instructions
from controlflow.orchestration.agent_context import AgentContext
from controlflow.tasks.task import (
    COMPLETE_STATUSES,
    INCOMPLETE_STATUSES,
    Task,
    TaskStatus,
)
from controlflow.utilities.context import ctx
from controlflow.utilities.testing import SimpleTask


def test_status_coverage():
    assert INCOMPLETE_STATUSES | COMPLETE_STATUSES == set(TaskStatus)


def test_context_open_and_close():
    assert ctx.get("tasks") == []
    with SimpleTask() as ta:
        assert ctx.get("tasks") == [ta]
        with SimpleTask() as tb:
            assert ctx.get("tasks") == [ta, tb]
        assert ctx.get("tasks") == [ta]
    assert ctx.get("tasks") == []


def test_task_requires_objective():
    with pytest.raises(ValueError):
        Task()


def test_task_initialization():
    task = Task(objective="Test objective")
    assert task.objective == "Test objective"
    assert task.status == TaskStatus.PENDING
    assert task.result is None
    assert task.error is None


def test_stable_id():
    t1 = Task(objective="Test Objective")
    t2 = Task(objective="Test Objective")
    t3 = Task(objective="Test Objective+")
    assert t1.id == t2.id == "e90eaf1f"
    assert t3.id == "3cc39696"


def test_task_mark_successful_and_mark_failed():
    task = SimpleTask()
    task.mark_successful(result=None)
    assert task.status == TaskStatus.SUCCESSFUL
    task.mark_failed(reason="test error")
    assert task.status == TaskStatus.FAILED


def test_task_loads_instructions_at_creation():
    with instructions("test instruction"):
        task = SimpleTask()

    assert "test instruction" in task.instructions


def test_task_dependencies():
    task1 = SimpleTask()
    task2 = SimpleTask(depends_on=[task1])
    assert task1 in task2.depends_on
    assert task2 in task1._downstreams


def test_task_context_dependencies():
    task1 = SimpleTask()
    task2 = SimpleTask(context=dict(a=task1))
    assert task1 in task2.depends_on
    assert task2 in task1._downstreams


def test_task_context_complex_dependencies():
    task1 = SimpleTask()
    task2 = SimpleTask()
    task3 = SimpleTask(context=dict(a=[task1], b=dict(c=[task2])))
    assert task1 in task3.depends_on
    assert task2 in task3.depends_on
    assert task3 in task1._downstreams
    assert task3 in task2._downstreams


def test_task_subtasks():
    task1 = SimpleTask()
    task2 = SimpleTask(parent=task1)
    assert task2 in task1._subtasks
    assert task2.parent is task1


def test_task_parent_context():
    with SimpleTask() as task1:
        with SimpleTask() as task2:
            task3 = SimpleTask()

    assert task3.parent is task2
    assert task2.parent is task1
    assert task1.parent is None

    assert task1._subtasks == {task2}
    assert task2._subtasks == {task3}
    assert task3._subtasks == set()


def test_task_agent_assignment():
    agent = Agent(name="Test Agent")
    task = SimpleTask(agent=agent)
    assert task.agent is agent


def test_task_bad_agent_assignment():
    with pytest.raises(ValueError):
        SimpleTask(agent=5)


def test_task_loads_agent_from_parent():
    agent = Agent(name="Test Agent")
    with SimpleTask(agent=agent):
        child = SimpleTask()

    assert child.agents is None
    assert child.get_agent() == agent


def test_task_loads_agent_from_flow():
    def_agent = controlflow.defaults.agent
    agent = Agent(name="Test Agent")
    with Flow(agent=agent):
        task = SimpleTask()

        assert task.agents is None
        assert task.get_agent() == agent

    # outside the flow context, pick up the default agent
    assert task.get_agent() == def_agent


def test_task_loads_agent_from_default_if_none_otherwise():
    agent = controlflow.defaults.agent
    task = SimpleTask()

    assert task.agents is None
    assert task.get_agent() == agent


def test_task_loads_agent_from_parent_before_flow():
    agent1 = Agent(name="Test Agent 1")
    agent2 = Agent(name="Test Agent 2")
    with Flow(agent=agent1):
        with SimpleTask(agent=agent2):
            child = SimpleTask()

    assert child.agent is None
    assert child.get_agent() == agent2


class TestWarning:
    def test_warn_on_steps_without_flow(self, default_fake_llm, caplog):
        default_fake_llm.set_responses(["Hi."])
        task = SimpleTask()
        task.run(steps=1)
        assert (
            "Running a task with a steps argument but no flow is not recommended"
            in caplog.text
        )

    async def test_warn_on_steps_without_flow_async(self, default_fake_llm, caplog):
        default_fake_llm.set_responses(["Hi."])
        task = SimpleTask()
        await task.run_async(steps=1)
        assert (
            "Running a task with a steps argument but no flow is not recommended"
            in caplog.text
        )


class TestFlowRegistration:
    def test_task_tracking(self):
        with Flow() as flow:
            task = SimpleTask()
            assert task in flow.tasks

    def test_task_tracking_on_call(self):
        task = SimpleTask()
        with Flow() as flow:
            task.run(steps=1)
        assert task in flow.tasks

    def test_parent_child_tracking(self):
        with Flow() as flow:
            with SimpleTask() as parent:
                with SimpleTask() as child:
                    grandchild = SimpleTask()

        assert parent in flow.tasks
        assert child in flow.tasks
        assert grandchild in flow.tasks

        assert len(flow.graph.edges) == 2


class TestTaskStatus:
    def test_task_status_transitions(self):
        task = SimpleTask()
        assert task.is_incomplete()
        assert not task.is_running()
        assert not task.is_complete()
        assert not task.is_successful()
        assert not task.is_failed()
        assert not task.is_skipped()

        task.mark_successful()
        assert not task.is_incomplete()
        assert not task.is_running()
        assert task.is_complete()
        assert task.is_successful()
        assert not task.is_failed()
        assert not task.is_skipped()

        task = SimpleTask()
        task.mark_failed()
        assert not task.is_incomplete()
        assert task.is_complete()
        assert not task.is_successful()
        assert task.is_failed()
        assert not task.is_skipped()

        task = SimpleTask()
        task.mark_skipped()
        assert not task.is_incomplete()
        assert task.is_complete()
        assert not task.is_successful()
        assert not task.is_failed()
        assert task.is_skipped()

    def test_validate_upstream_dependencies_on_success(self):
        task1 = SimpleTask()
        task2 = SimpleTask(depends_on=[task1])
        with pytest.raises(ValueError, match="cannot be marked successful"):
            task2.mark_successful()
        task1.mark_successful()
        task2.mark_successful()

    def test_validate_subtask_dependencies_on_success(self):
        task1 = SimpleTask()
        task2 = SimpleTask(parent=task1)
        with pytest.raises(ValueError, match="cannot be marked successful"):
            task1.mark_successful()
        task2.mark_successful()
        task1.mark_successful()

    def test_task_ready(self):
        task1 = SimpleTask()
        assert task1.is_ready()

    def test_task_not_ready_if_successful(self):
        task1 = SimpleTask()
        task1.mark_successful()
        assert not task1.is_ready()

    def test_task_not_ready_if_failed(self):
        task1 = SimpleTask()
        task1.mark_failed()
        assert not task1.is_ready()

    def test_task_not_ready_if_dependencies_are_ready(self):
        task1 = SimpleTask()
        task2 = SimpleTask(depends_on=[task1])
        assert task1.is_ready()
        assert not task2.is_ready()

    def test_task_ready_if_dependencies_are_ready(self):
        task1 = SimpleTask()
        task2 = SimpleTask(depends_on=[task1])
        task1.mark_successful()
        assert not task1.is_ready()
        assert task2.is_ready()

    def test_task_hash(self):
        task1 = SimpleTask()
        task2 = SimpleTask()
        assert hash(task1) != hash(task2)


class TestTaskPrompt:
    @pytest.fixture
    def agent_context(self) -> AgentContext:
        return AgentContext(agents=[Agent(name="Test Agent")], flow=Flow(), tasks=[])

    def test_default_prompt(self):
        task = SimpleTask()
        assert task.prompt is None

    def test_default_template(self, agent_context):
        task = SimpleTask()
        prompt = task.get_prompt(context=agent_context)
        assert prompt.startswith("## Task")

    def test_custom_prompt(self, agent_context):
        task = SimpleTask(prompt="Custom Prompt")
        prompt = task.get_prompt(context=agent_context)
        assert prompt == "Custom Prompt"

    def test_custom_templated_prompt(self, agent_context):
        task = SimpleTask(prompt="{{ task.objective }}", objective="abc")
        prompt = task.get_prompt(context=agent_context)
        assert prompt == "abc"
