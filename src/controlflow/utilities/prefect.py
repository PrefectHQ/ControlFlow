import inspect
import json
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
)
from uuid import UUID

import prefect.tasks
from prefect import get_client as get_prefect_client
from prefect.artifacts import ArtifactRequest
from prefect.client.orchestration import SyncPrefectClient, get_client
from prefect.client.schemas import State, TaskRun
from prefect.context import (
    FlowRunContext,
    TaskRunContext,
)
from prefect.events.schemas.events import Event
from prefect.results import ResultFactory
from prefect.states import (
    Cancelled,
    Completed,
    Failed,
    Running,
    return_value_to_state,
)
from prefect.utilities.asyncutils import run_coro_as_sync
from prefect.utilities.engine import (
    emit_task_run_state_change_event,
    propose_state_sync,
)
from pydantic import TypeAdapter

from controlflow.utilities.types import ControlFlowModel

if TYPE_CHECKING:
    from controlflow.llm.tools import Tool


def create_markdown_artifact(
    key: str,
    markdown: str,
    description: str = None,
    task_run_id: UUID = None,
    flow_run_id: UUID = None,
) -> None:
    """
    Create a Markdown artifact.
    """

    tr_context = TaskRunContext.get()
    fr_context = FlowRunContext.get()

    if tr_context:
        task_run_id = task_run_id or tr_context.task_run.id
    if fr_context:
        flow_run_id = flow_run_id or fr_context.flow_run.id

    client = get_prefect_client()
    run_coro_as_sync(
        client.create_artifact(
            artifact=ArtifactRequest(
                key=key,
                data=markdown,
                description=description,
                type="markdown",
                task_run_id=task_run_id,
                flow_run_id=flow_run_id,
            )
        )
    )


def create_json_artifact(
    key: str,
    data: Any,
    description: str = None,
    task_run_id: UUID = None,
    flow_run_id: UUID = None,
) -> None:
    """
    Create a JSON artifact.
    """

    try:
        markdown = TypeAdapter(type(data)).dump_json(data, indent=2).decode()
        markdown = f"```json\n{markdown}\n```"
    except Exception:
        markdown = str(data)

    create_markdown_artifact(
        key=key,
        markdown=markdown,
        description=description,
        task_run_id=task_run_id,
        flow_run_id=flow_run_id,
    )


def create_python_artifact(
    key: str,
    code: str,
    description: str = None,
    task_run_id: UUID = None,
    flow_run_id: UUID = None,
) -> None:
    """
    Create a Python artifact.
    """

    create_markdown_artifact(
        key=key,
        markdown=f"```python\n{code}\n```",
        description=description,
        task_run_id=task_run_id,
        flow_run_id=flow_run_id,
    )


TOOL_CALL_FUNCTION_RESULT_TEMPLATE = inspect.cleandoc(
    """
    # Tool call: {name}
    
    **Description:** {description}
    
    ## Arguments
    
    ```json
    {args}
    ```
    
    ## Result
    
    ```
    {result}
    ```
    """
)


def wrap_prefect_tool(tool: "Tool") -> "Tool":
    """
    Wrap an Agent tool in a Prefect task.
    """

    # for functions, we modify the function to become a Prefect task and
    # publish an artifact that contains details about the function call

    if isinstance(tool.func, prefect.tasks.Task) or isinstance(
        tool.coroutine, prefect.tasks.Task
    ):
        return tool

    if tool.coroutine is not None:

        async def modified_coroutine(
            # provide args with default values to avoid a late-binding issue
            original_coroutine: Callable = tool.coroutine,
            tool: "Tool" = tool,
            **kwargs,
        ):
            # call fn
            result = await original_coroutine(**kwargs)

            # prepare artifact
            passed_args = inspect.signature(original_coroutine).bind(**kwargs).arguments
            try:
                # try to pretty print the args
                passed_args = json.dumps(passed_args, indent=2)
            except Exception:
                pass
            create_markdown_artifact(
                markdown=TOOL_CALL_FUNCTION_RESULT_TEMPLATE.format(
                    name=tool.name,
                    description=tool.description or "(none provided)",
                    args=passed_args,
                    result=result,
                ),
                key="tool-result",
            )

            # return result
            return result

        tool.coroutine = prefect.task(
            modified_coroutine,
            task_run_name=f"Tool call: {tool.name}",
        )

    def modified_fn(
        # provide args with default values to avoid a late-binding issue
        original_func: Callable = tool.func,
        tool: "Tool" = tool,
        **kwargs,
    ):
        # call fn
        result = original_func(**kwargs)

        # prepare artifact
        passed_args = inspect.signature(original_func).bind(**kwargs).arguments
        try:
            # try to pretty print the args
            passed_args = json.dumps(passed_args, indent=2)
        except Exception:
            pass
        create_markdown_artifact(
            markdown=TOOL_CALL_FUNCTION_RESULT_TEMPLATE.format(
                name=tool.name,
                description=tool.description or "(none provided)",
                args=passed_args,
                result=result,
            ),
            key="tool-result",
        )

        # return result
        return result

    # replace the function with the modified version
    tool.func = prefect.task(
        modified_fn,
        task_run_name=f"Tool call: {tool.name}",
    )

    return tool


class PrefectTrackingTask(ControlFlowModel):
    """
    A utility for creating a Prefect task that tracks the state of a task run
    without requiring a function or related Prefect machinery such as error
    handling. We use this to map ControlFlow Task objects onto Prefect states.

    While calling cf.Task.run() provides an opportunity to map a ControlFlow
    task directly onto an invocation of compute (and therefore a classic Prefect
    Task), when run as part of a flow we do not "invoke" ControlFlow tasks
    directly. Instead, an agent may decide at any time to work on any ready task
    and, ultimately, mark it as complete. We model this behavior into Prefect
    with this TrackingTask. The corresponding Prefect task starts "running" as
    soon as the CF Task is ready to run (e.g. as soon as it becomes eligible for
    an agent to work on it). It then transitions to a terminal state when the CF
    Task is complete. This gives us excellent visibility into the state of the
    CF Task within the Prefect UI, even though it doesn't correspond to a single
    invocation of compute.
    """

    name: str
    description: Optional[str] = None
    task_run_id: Optional[str] = None
    tags: Optional[list[str]] = None

    _task: prefect.Task = None
    _task_run: TaskRun = None
    _client: SyncPrefectClient = None
    _last_event: Optional[Event] = None
    is_started: bool = False

    _context: list = []

    def start(self, depends_on: list = None):
        if self.is_started:
            raise ValueError("Task already started")
        self.is_started = True
        self._client = get_client(sync_client=True)

        self._task = prefect.Task(
            fn=lambda: None,
            name=self.name,
            description=self.description,
            tags=self.tags,
        )

        self._task_run = run_coro_as_sync(
            self._task.create_run(
                id=self.task_run_id,
                parameters=dict(depends_on=depends_on),
                flow_run_context=FlowRunContext.get(),
            )
        )

        self._last_event = emit_task_run_state_change_event(
            task_run=self._task_run,
            initial_state=None,
            validated_state=self._task_run.state,
        )

        self.set_state(Running())

    def set_state(self, state: State) -> State:
        if not self.is_started:
            raise ValueError("Task not started")

        new_state = propose_state_sync(
            self._client, state, task_run_id=self._task_run.id, force=True
        )

        self._last_event = emit_task_run_state_change_event(
            task_run=self._task_run,
            initial_state=self._task_run.state,
            validated_state=new_state,
            follows=self._last_event,
        )
        self._task_run.state = new_state
        return new_state

    def succeed(self, result: Any):
        if result is not None:
            terminal_state = run_coro_as_sync(
                return_value_to_state(
                    result,
                    result_factory=run_coro_as_sync(
                        ResultFactory.from_autonomous_task(self._task)
                    ),
                )
            )
        else:
            terminal_state = Completed()
        self.set_state(terminal_state)

    def fail(self, error: Optional[str] = None):
        self.set_state(Failed(message=error))

    def skip(self):
        self.set_state(Cancelled(message="Task skipped"))


def prefect_task_context(**kwargs):
    @contextmanager
    @prefect.task(**kwargs)
    def task_context():
        yield

    return task_context()


def prefect_flow_context(**kwargs):
    @contextmanager
    @prefect.flow(**kwargs)
    def flow_context():
        yield

    return flow_context()
