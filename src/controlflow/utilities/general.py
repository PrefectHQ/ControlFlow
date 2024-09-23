import hashlib
import json
import re
import textwrap
from typing import Optional, Union

import prefect
from pydantic import BaseModel, ConfigDict

# flag for unset defaults
NOTSET = "__NOTSET__"


def hash_objects(input_data: tuple, len: int = 8) -> str:
    """
    Generates a fast, stable MD5 hash for the given tuple of input data.

    Args:
        input_data (tuple): The tuple of data to hash.

    Returns:
        str: The hexadecimal digest of the MD5 hash.
    """
    # Serialize the tuple into a JSON string
    serialized_data = json.dumps(input_data, sort_keys=True)

    # Create an MD5 hash object
    hasher = hashlib.md5()

    # Update the hash object with the serialized string
    hasher.update(serialized_data.encode("utf-8"))

    # Return the hexadecimal digest of the hash
    return hasher.hexdigest()[:len]


def unwrap(text: str) -> str:
    """
    Given a multi-line string, dedent, remove newlines within paragraphs, but keep paragraph breaks.
    """
    # Dedent the text
    dedented_text = textwrap.dedent(text)

    # Remove newlines within paragraphs, but keep paragraph breaks
    cleaned_text = re.sub(r"(?<!\n)\n(?!\n)", " ", dedented_text)

    # Remove leading and trailing whitespace
    cleaned_text = cleaned_text.strip()

    return cleaned_text


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
