import datetime
import inspect
import json
from functools import partial, update_wrapper
from typing import Any, Callable, Literal, Optional, Union

import litellm
import pydantic
from marvin.beta.assistants import Assistant, Thread
from marvin.beta.assistants.assistants import AssistantTool
from marvin.types import FunctionTool
from marvin.utilities.asyncio import ExposeSyncMethodsMixin
from pydantic import BaseModel, Field, PrivateAttr
from traitlets import default

# flag for unset defaults
NOTSET = "__NOTSET__"


ToolType = Union[FunctionTool, AssistantTool, Callable]


class ControlFlowModel(BaseModel):
    model_config = dict(validate_assignment=True, extra="forbid")


class PandasDataFrame(ControlFlowModel):
    """Schema for a pandas dataframe"""

    data: Union[
        list[list[Union[str, int, float, bool]]],
        dict[str, list[Union[str, int, float, bool]]],
    ]
    columns: list[str] = None
    index: list[str] = None
    dtype: dict[str, str] = None


class PandasSeries(ControlFlowModel):
    """Schema for a pandas series"""

    data: list[Union[str, int, float]]
    index: list[str] = None
    name: str = None
    dtype: str = None


class ToolFunction(ControlFlowModel):
    name: str
    parameters: dict
    description: str = ""


class Tool(ControlFlowModel):
    type: Literal["function"] = "function"
    function: ToolFunction
    _fn: Callable = PrivateAttr()

    def __init__(self, *, _fn: Callable, **kwargs):
        super().__init__(**kwargs)
        self._fn = _fn

    @classmethod
    def from_function(
        cls, fn: Callable, name: Optional[str] = None, description: Optional[str] = None
    ):
        return cls(
            function=ToolFunction(
                name=name or fn.__name__,
                description=inspect.cleandoc(description or fn.__doc__ or ""),
                parameters=pydantic.TypeAdapter(
                    fn, config=pydantic.ConfigDict(arbitrary_types_allowed=True)
                ).json_schema(),
            ),
            _fn=fn,
        )

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class ToolResult(ControlFlowModel):
    model_config = dict(allow_arbitrary_types=True)
    tool_call_id: str
    tool_name: str
    tool: Tool
    args: dict
    is_error: bool
    result: Any = Field(None, exclude=True)


class Message(litellm.Message):
    model_config = dict(validate_assignment=True)
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )

    tool_result: Optional[ToolResult] = None

    def __init__(
        self, content: str, *, role: str = None, tool_result: Any = None, **kwargs
    ):
        super().__init__(content=content, role=role, **kwargs)
        self.tool_result = tool_result
