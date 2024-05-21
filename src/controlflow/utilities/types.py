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
from pydantic import BaseModel, PrivateAttr

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


class ToolCall(ControlFlowModel):
    model_config = dict(allow_arbitrary_types=True)
    tool_call_id: str
    tool_name: str
    tool: Tool
    args: dict
    output: Any


class Message(litellm.Message):
    _tool_call: ToolCall = PrivateAttr()

    def __init__(self, *args, tool_output: Any = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._tool_output = tool_output
