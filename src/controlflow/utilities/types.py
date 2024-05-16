from typing import Callable, Union

from marvin.beta.assistants import Assistant, Thread
from marvin.beta.assistants.assistants import AssistantTool
from marvin.types import FunctionTool
from marvin.utilities.asyncio import ExposeSyncMethodsMixin
from pydantic import BaseModel

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
