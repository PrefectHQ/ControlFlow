from enum import Enum
from typing import Annotated, Any, Dict, List, Literal

import pytest
from pydantic import BaseModel

import controlflow
from controlflow.agents import Agent
from controlflow.events.base import Event
from controlflow.events.events import AgentMessage
from controlflow.flows import Flow
from controlflow.instructions import instructions
from controlflow.orchestration.handler import Handler
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
    assert ctx.get("tasks") is None
    with SimpleTask() as ta:
        assert ctx.get("tasks") == [ta]
        with SimpleTask() as tb:
            assert ctx.get("tasks") == [ta, tb]
        assert ctx.get("tasks") == [ta]
    assert ctx.get("tasks") is None


def test_task_requires_objective():
    with pytest.raises(ValueError):
        Task()


def test_task_initialization():
    task = Task(objective="Test objective")
    assert task.objective == "Test objective"
    assert task.status == TaskStatus.PENDING
    assert task.result is None


@pytest.mark.skip(reason="IDs are not stable right now")
def test_stable_id():
    t1 = Task(objective="Test Objective")
    t2 = Task(objective="Test Objective")
    t3 = Task(objective="Test Objective+")
    assert t1.id == t2.id == "9663272a"  # Update this line with the new ID
    assert t3.id != t1.id  # Ensure different objectives produce different IDs


def test_task_mark_successful_and_mark_failed():
    task = Task(objective="Test Objective", result_type=int)
    task.mark_successful(result=5)
    assert task.result == 5
    assert task.status == TaskStatus.SUCCESSFUL
    task.mark_failed(reason="test error")
    assert task.status == TaskStatus.FAILED
    assert task.result == "test error"


def test_task_loads_instructions_at_creation():
    with instructions("test instruction"):
        task = SimpleTask()

    assert "test instruction" in task.instructions


def test_task_dependencies():
    task1 = SimpleTask()
    task2 = SimpleTask(depends_on=[task1])
    assert task1 in task2.depends_on
    assert task2 in task1._downstreams


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
    agent = Agent()
    task = SimpleTask(agents=[agent])
    assert task.agents == [agent]
    assert task.get_agents() == [agent]


def test_task_bad_agent_assignment():
    with pytest.raises(ValueError):
        SimpleTask(agents=5)


def test_task_loads_agent_from_parent():
    agent = Agent()
    with SimpleTask(agents=[agent]):
        child = SimpleTask()

    assert child.agents is None
    assert child.get_agents() == [agent]


def test_task_loads_agent_from_flow():
    def_agent = controlflow.defaults.agent
    agent = Agent()
    with Flow(default_agent=agent):
        task = SimpleTask()

        assert task.agents is None
        assert task.get_agents() == [agent]

    # outside the flow context, pick up the default agent
    assert task.get_agents() == [def_agent]


def test_task_loads_agent_from_default_if_none_otherwise():
    agent = controlflow.defaults.agent
    task = SimpleTask()

    assert task.agents is None
    assert task.get_agents() == [agent]


def test_task_loads_agent_from_parent_before_flow():
    agent1 = Agent()
    agent2 = Agent()
    with Flow(default_agent=agent1):
        with SimpleTask(agents=[agent2]):
            child = SimpleTask()

    assert child.agents is None
    assert child.get_agents() == [agent2]


def test_completion_agents_default():
    task = Task(objective="Test task")
    assert task.completion_agents is None


def test_completion_agents_set():
    agent1 = Agent(name="Agent 1")
    agent2 = Agent(name="Agent 2")
    task = Task(objective="Test task", completion_agents=[agent1, agent2])
    assert task.completion_agents == [agent1, agent2]


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
    def test_default_prompt(self):
        task = SimpleTask()
        assert task.prompt is None

    def test_default_template(self):
        task = SimpleTask()
        prompt = task.get_prompt()
        assert prompt.startswith("- id:")
        assert "- objective: test" in prompt
        assert "- context:" in prompt

    def test_custom_prompt(self):
        task = SimpleTask(prompt="Custom Prompt")
        prompt = task.get_prompt()
        assert prompt == "Custom Prompt"

    def test_custom_templated_prompt(self):
        task = SimpleTask(prompt="{{ task.objective }}", objective="abc")
        prompt = task.get_prompt()
        assert prompt == "abc"


class TestResultType:
    def test_int_result(self):
        task = Task("choose 5", result_type=int)
        task.mark_successful(result=5)
        assert task.result == 5

    def test_str_result(self):
        task = Task("choose 5", result_type=str)
        task.mark_successful(result="5")
        assert task.result == "5"

    def test_typed_dict_result(self):
        task = Task("", result_type=dict[str, int])
        task.mark_successful(result={"a": 5, "b": "6"})
        assert task.result == {"a": 5, "b": 6}

    def test_special_list_type_result(self):
        # test capitalized List type
        task = Task("", result_type=List[int])
        task.mark_successful(result=[5, 6])
        assert task.result == [5, 6]

    def test_special_dict_type_result(self):
        # test capitalized Dict type
        task = Task("", result_type=Dict[str, int])
        task.mark_successful(result={"a": 5, "b": "6"})
        assert task.result == {"a": 5, "b": 6}

    def test_pydantic_result(self):
        class Name(BaseModel):
            first: str
            last: str

        task = Task("The character said his name is John Doe", result_type=Name)
        task.run()
        assert task.result == Name(first="John", last="Doe")

    def test_annotated_result(self):
        task = Task(
            "generate any result that satisfies the result type",
            result_type=Annotated[str, "a 5 digit zip code"],
        )
        task.run()
        assert len(task.result) == 5
        assert int(task.result)


class TestResultTypeConstrainedChoice:
    class Letter(BaseModel):
        letter: str

        def __hash__(self):
            return id(self)

    A = Letter(letter="a")
    B = Letter(letter="b")
    C = Letter(letter="c")

    def test_tuple_of_ints_result(self):
        task = Task("choose 5", result_type=(4, 5, 6))
        task.mark_successful(result=5)
        assert task.result == 5

    def test_tuple_of_ints_validates(self):
        task = Task("choose 5", result_type=(4, 5, 6))
        with pytest.raises(ValueError):
            task.mark_successful(result=7)

    def test_list_of_strings_result(self):
        # test list of strings result
        task = Task(
            "Choose the second letter of the alphabet", result_type=["b", "c", "a"]
        )
        task.run()
        assert task.result == "b"

    def test_list_of_objects_result(self):
        # test list of strings result
        task = Task(
            "Choose the second letter of the alphabet",
            result_type=[self.A, self.C, self.B],
        )
        task.run()
        assert task.result is self.B

    def test_tuple_of_objects_result(self):
        # test list of strings result
        task = Task(
            "Choose the second letter of the alphabet",
            result_type=(self.A, self.C, self.B),
        )
        task.run()
        assert task.result is self.B

    def test_set_of_objects_result(self):
        # test list of strings result
        task = Task(
            "Choose the second letter of the alphabet",
            result_type={self.A, self.C, self.B},
        )
        task.run()
        assert task.result is self.B

    def test_literal_string_result(self):
        task = Task(
            "Choose the second letter of the alphabet",
            result_type=Literal["a", "c", "b"],
        )
        task.run()
        assert task.result == "b"

    def test_enum_result(self):
        class Letters(Enum):
            A = "a"
            B = "b"
            C = "c"

        task = Task("Choose the second letter of the alphabet", result_type=Letters)
        task.run()
        assert task.result is Letters.B

    def test_literal_object_result(self):
        # this is bad syntax, but works
        task = Task(
            "Choose the second letter of the alphabet",
            result_type=Literal[self.A, self.B, self.C],  # noqa
        )
        task.run()
        assert task.result is self.B

    def test_list_of_literals_result(self):
        task = Task(
            "Choose the second and third letters of the alphabet",
            result_type=list[Literal["a", "b", "c"]],
        )
        task.run()
        assert task.result == ["b", "c"]

    def test_map_labels_to_values(self):
        task = Task(
            "Choose the right label, in order provided in context",
            context=dict(goals=["the second letter", "the first letter"]),
            result_type=list[Literal["a", "b", "c"]],
        )
        task.run()
        assert task.result == ["b", "a"]


class TestResultValidator:
    def test_result_validator(self):
        def validate_even(value: int) -> int:
            if value % 2 != 0:
                raise ValueError("Value must be even")
            return value

        task = Task(
            "choose an even number", result_type=int, result_validator=validate_even
        )
        task.mark_successful(result=4)
        assert task.result == 4

        with pytest.raises(ValueError, match="Value must be even"):
            task.mark_successful(result=5)

    def test_result_validator_with_constraints(self):
        def validate_range(value: int) -> int:
            if not 10 <= value <= 20:
                raise ValueError("Value must be between 10 and 20")
            return value

        task = Task("choose a number", result_type=int, result_validator=validate_range)
        task.mark_successful(result=15)
        assert task.result == 15

        with pytest.raises(ValueError, match="Value must be between 10 and 20"):
            task.mark_successful(result=5)

    def test_result_validator_with_modification(self):
        def round_to_nearest_ten(value: int) -> int:
            return round(value, -1)

        task = Task(
            "choose a number", result_type=int, result_validator=round_to_nearest_ten
        )
        task.mark_successful(result=44)
        assert task.result == 40

        task.mark_successful(result=46)
        assert task.result == 50

    def test_result_validator_with_pydantic_model(self):
        class User(BaseModel):
            name: str
            age: int

        def validate_adult(user: User) -> User:
            if user.age < 18:
                raise ValueError("User must be an adult")
            return user

        task = Task(
            "create an adult user", result_type=User, result_validator=validate_adult
        )
        task.mark_successful(result={"name": "John", "age": 25})
        assert task.result == User(name="John", age=25)

        with pytest.raises(ValueError, match="User must be an adult"):
            task.mark_successful(result={"name": "Jane", "age": 16})

    def test_result_validator_applied_after_type_coercion(self):
        def always_return_none(value: Any) -> None:
            return None

        task = Task(
            "do something with no result",
            result_type=None,
            result_validator=always_return_none,
        )

        with pytest.raises(
            ValueError, match="Task has result_type=None, but a result was provided"
        ):
            task.mark_successful(result="anything")


class TestSuccessTool:
    def test_success_tool(self):
        task = Task("choose 5", result_type=int)
        tool = task.get_success_tool()
        tool.run(input=dict(result=5))
        assert task.is_successful()
        assert task.result == 5

    def test_success_tool_with_list_of_options(self):
        task = Task('choose "good"', result_type=["bad", "good", "medium"])
        tool = task.get_success_tool()
        tool.run(input=dict(result=1))
        assert task.is_successful()
        assert task.result == "good"

    def test_success_tool_with_list_of_options_requires_int(self):
        task = Task('choose "good"', result_type=["bad", "good", "medium"])
        tool = task.get_success_tool()
        with pytest.raises(ValueError):
            tool.run(input=dict(result="good"))

    def test_tuple_of_ints_result(self):
        task = Task("choose 5", result_type=(4, 5, 6))
        tool = task.get_success_tool()
        tool.run(input=dict(result=1))
        assert task.result == 5

    def test_tuple_of_pydantic_models_result(self):
        class Person(BaseModel):
            name: str
            age: int

        task = Task(
            "Who is the oldest?",
            result_type=(Person(name="Alice", age=30), Person(name="Bob", age=35)),
        )
        tool = task.get_success_tool()
        tool.run(input=dict(result=1))
        assert task.result == Person(name="Bob", age=35)
        assert isinstance(task.result, Person)


class TestHandlers:
    class ExampleHandler(Handler):
        def __init__(self):
            self.events = []
            self.agent_messages = []

        def on_event(self, event: Event):
            self.events.append(event)

        def on_agent_message(self, event: AgentMessage):
            self.agent_messages.append(event)

    def test_task_run_with_handlers(self, default_fake_llm):
        handler = self.ExampleHandler()
        task = Task(objective="Calculate 2 + 2", result_type=int)
        task.run(handlers=[handler], max_llm_calls=1)

        assert len(handler.events) > 0
        assert len(handler.agent_messages) == 1

    async def test_task_run_async_with_handlers(self, default_fake_llm):
        handler = self.ExampleHandler()
        task = Task(objective="Calculate 2 + 2", result_type=int)
        await task.run_async(handlers=[handler], max_llm_calls=1)

        assert len(handler.events) > 0
        assert len(handler.agent_messages) == 1


class TestCompletionTools:
    def test_default_completion_tools(self):
        task = Task(objective="Test task")
        assert task.completion_tools is None
        tools = task.get_completion_tools()
        assert len(tools) == 2
        assert any(t.name == f"mark_task_{task.id}_successful" for t in tools)
        assert any(t.name == f"mark_task_{task.id}_failed" for t in tools)

    def test_only_succeed_tool(self):
        task = Task(objective="Test task", completion_tools=["SUCCEED"])
        tools = task.get_completion_tools()
        assert len(tools) == 1
        assert tools[0].name == f"mark_task_{task.id}_successful"

    def test_only_fail_tool(self):
        task = Task(objective="Test task", completion_tools=["FAIL"])
        tools = task.get_completion_tools()
        assert len(tools) == 1
        assert tools[0].name == f"mark_task_{task.id}_failed"

    def test_no_completion_tools(self):
        task = Task(objective="Test task", completion_tools=[])
        tools = task.get_completion_tools()
        assert len(tools) == 0

    def test_invalid_completion_tool(self):
        with pytest.raises(ValueError):
            Task(objective="Test task", completion_tools=["INVALID"])

    def test_manual_success_tool(self):
        task = Task(objective="Test task", completion_tools=[], result_type=int)
        success_tool = task.get_success_tool()
        success_tool.run(input=dict(result=5))
        assert task.is_successful()
        assert task.result == 5

    def test_manual_fail_tool(self):
        task = Task(objective="Test task", completion_tools=[])
        fail_tool = task.get_fail_tool()
        assert fail_tool.name == f"mark_task_{task.id}_failed"
        fail_tool.run(input=dict(reason="test error"))
        assert task.is_failed()
        assert task.result == "test error"

    def test_completion_tools_with_run(self):
        task = Task("Calculate 2 + 2", result_type=int, completion_tools=["SUCCEED"])
        result = task.run(max_llm_calls=1)
        assert result == 4
        assert task.is_successful()

    def test_no_completion_tools_with_run(self):
        task = Task("Calculate 2 + 2", result_type=int, completion_tools=[])
        task.run(max_llm_calls=1)
        assert task.is_incomplete()

    async def test_completion_tools_with_run_async(self):
        task = Task("Calculate 2 + 2", result_type=int, completion_tools=["SUCCEED"])
        result = await task.run_async(max_llm_calls=1)
        assert result == 4
        assert task.is_successful()

    async def test_no_completion_tools_with_run_async(self):
        task = Task("Calculate 2 + 2", result_type=int, completion_tools=[])
        await task.run_async(max_llm_calls=1)
        assert task.is_incomplete()
