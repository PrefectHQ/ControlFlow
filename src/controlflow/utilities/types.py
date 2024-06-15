from typing import Union

import prefect
from pydantic import BaseModel, ConfigDict

# flag for unset defaults
NOTSET = "__NOTSET__"


class ControlFlowModel(BaseModel):
    model_config: dict = ConfigDict(
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
    columns: list[str] = None
    index: list[str] = None
    dtype: dict[str, str] = None


class PandasSeries(ControlFlowModel):
    """Schema for a pandas series"""

    data: list[Union[str, int, float]]
    index: list[str] = None
    name: str = None
    dtype: str = None


class _OpenAIBaseType(ControlFlowModel):
    model_config = dict(extra="allow")

    # ---- end openai fields
