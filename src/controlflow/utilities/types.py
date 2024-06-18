from typing import Optional, Union

import prefect
from pydantic import BaseModel, ConfigDict

# flag for unset defaults
NOTSET = "__NOTSET__"


class ControlFlowModel(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        ignored_types=(prefect.Flow, prefect.Task),
    )


class PandasDataFrame(ControlFlowModel):
    """Schema for a pandas dataframe"""

    data: Union[
        list[list[Union[str, int, float, bool]]],
        dict[str, list[Union[str, int, float, bool]]],
    ]
    columns: Optional[list[str]] = None
    index: Optional[list[str]] = None
    dtype: Optional[dict[str, str]] = None


class PandasSeries(ControlFlowModel):
    """Schema for a pandas series"""

    data: list[Union[str, int, float]]
    index: Optional[list[str]] = None
    name: Optional[str] = None
    dtype: Optional[str] = None


class _OpenAIBaseType(ControlFlowModel):
    model_config = ConfigDict(extra="allow")


__all__ = ["ControlFlowModel", "PandasDataFrame", "PandasSeries", "_OpenAIBaseType"]
