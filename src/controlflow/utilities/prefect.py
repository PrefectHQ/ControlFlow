from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
)
from uuid import UUID

import prefect
import prefect.cache_policies
import prefect.serializers
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

import controlflow
from controlflow.utilities.types import ControlFlowModel

if TYPE_CHECKING:
    pass


def prefect_task(*args, **kwargs):
    """
    A decorator that creates a Prefect task with ControlFlow defaults
    """

    # TODO: only open in Flow context?

    kwargs.setdefault("log_prints", controlflow.settings.log_prints)
    kwargs.setdefault("cache_policy", prefect.cache_policies.NONE)
    kwargs.setdefault("result_serializer", prefect.serializers.JSONSerializer())

    return prefect.task(*args, **kwargs)


def prefect_flow(*args, **kwargs):
    """
    A decorator that creates a Prefect flow with ControlFlow defaults
    """

    kwargs.setdefault("log_prints", controlflow.settings.log_prints)

    return prefect.flow(*args, **kwargs)


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
    """
    Creates a Prefect task that starts when the context is entered and ends when
    it closes. This is useful for creating a Prefect task that is not tied to a
    specific function but governs a block of code. Note that some features, like
    retries and caching, will not work.
    """
    supported_kwargs = {
        "name",
        "description",
        "task_run_name",
        "tags",
        "version",
        "timeout_seconds",
        "log_prints",
        "on_completion",
        "on_failure",
    }
    unsupported_kwargs = set(kwargs.keys()) - set(supported_kwargs)
    if unsupported_kwargs:
        raise ValueError(
            f"Unsupported keyword arguments for a task context provided: "
            f"{unsupported_kwargs}. Consider using a @task-decorated function instead."
        )

    @contextmanager
    @prefect_task(**kwargs)
    def task_context():
        yield

    return task_context()


def prefect_flow_context(**kwargs):
    """
    Creates a Prefect flow that starts when the context is entered and ends when
    it closes. This is useful for creating a Prefect flow that is not tied to a
    specific function but governs a block of code. Note that some features, like
    retries and caching, will not work.
    """

    supported_kwargs = {
        "name",
        "description",
        "flow_run_name",
        "tags",
        "version",
        "timeout_seconds",
        "log_prints",
        "on_completion",
        "on_failure",
    }
    unsupported_kwargs = set(kwargs.keys()) - set(supported_kwargs)
    if unsupported_kwargs:
        raise ValueError(
            f"Unsupported keyword arguments for a flow context provided: "
            f"{unsupported_kwargs}. Consider using a @flow-decorated function instead."
        )

    @contextmanager
    @prefect_flow(**kwargs)
    def flow_context():
        yield

    return flow_context()
