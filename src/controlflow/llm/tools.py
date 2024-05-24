import functools
import inspect
from functools import PrivateAttr, partial, update_wrapper
from typing import Any, Callable, Literal, Optional, Union

import pydantic

from controlflow.llm.messages import (
    AssistantMessage,
    ControlFlowMessage,
    ToolCall,
    ToolMessage,
)
from controlflow.utilities.types import ControlFlowModel


class ToolFunction(ControlFlowModel):
    name: str
    parameters: dict
    description: str = ""


class Tool(ControlFlowModel):
    type: Literal["function"] = "function"
    function: ToolFunction
    _fn: Callable = PrivateAttr()
    _metadata: dict = PrivateAttr(default_factory=dict)

    def __init__(self, *, _fn: Callable, _metadata: dict = None, **kwargs):
        super().__init__(**kwargs)
        self._fn = _fn
        self._metadata = _metadata or {}

    @classmethod
    def from_function(
        cls,
        fn: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        if name is None and fn.__name__ == "<lambda>":
            name = "__lambda__"

        return cls(
            function=ToolFunction(
                name=name or fn.__name__,
                description=inspect.cleandoc(description or fn.__doc__ or ""),
                parameters=pydantic.TypeAdapter(
                    fn, config=pydantic.ConfigDict(arbitrary_types_allowed=True)
                ).json_schema(),
            ),
            _fn=fn,
            _metadata=metadata or getattr(fn, "__metadata__", {}),
        )

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


def tool(
    fn: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Tool:
    if fn is None:
        return partial(tool, name=name, description=description, metadata=metadata)
    return Tool.from_function(fn, name=name, description=description, metadata=metadata)


def annotate_fn(
    fn: Callable,
    name: Optional[str] = None,
    description: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Callable:
    """
    Annotate a function with a new name and description without modifying the
    original. Useful when you want to provide a custom name and description for
    a tool, but without creating a new tool object.
    """
    new_fn = functools.partial(fn)
    new_fn.__name__ = name or fn.__name__
    new_fn.__doc__ = description or fn.__doc__
    new_fn.__metadata__ = getattr(fn, "__metadata__", {}) | metadata
    return new_fn


def as_tools(tools: list[Union[Tool, Callable]]) -> list[Tool]:
    tools = [t if isinstance(t, Tool) else tool(t) for t in tools]
    if len({t.function.name for t in tools}) != len(tools):
        duplicates = {t.function.name for t in tools if tools.count(t) > 1}
        raise ValueError(
            f"Tool names must be unique, but found duplicates: {', '.join(duplicates)}"
        )
    return tools


def as_tool_lookup(tools: list[Union[Tool, Callable]]) -> dict[str, Tool]:
    return {t.function.name: t for t in as_tools(tools)}


def custom_partial(func: Callable, **fixed_kwargs: Any) -> Callable:
    """
    Returns a new function with partial application of the given keyword arguments.
    The new function has the same __name__ and docstring as the original, and its
    signature excludes the provided kwargs.
    """

    # Define the new function with a dynamic signature
    def wrapper(**kwargs):
        # Merge the provided kwargs with the fixed ones, prioritizing the former
        all_kwargs = {**fixed_kwargs, **kwargs}
        return func(**all_kwargs)

    # Update the wrapper function's metadata to match the original function
    update_wrapper(wrapper, func)

    # Modify the signature to exclude the fixed kwargs
    original_sig = inspect.signature(func)
    new_params = [
        param
        for param in original_sig.parameters.values()
        if param.name not in fixed_kwargs
    ]
    wrapper.__signature__ = original_sig.replace(parameters=new_params)

    return wrapper


def output_to_string(output: Any) -> str:
    """
    Function outputs must be provided as strings
    """
    if output is None:
        output = ""
    elif not isinstance(output, str):
        try:
            output = pydantic.TypeAdapter(type(output)).dump_json(output).decode()
        except Exception:
            output = str(output)
    return output


def get_tool_calls(
    messages: list[ControlFlowMessage],
) -> list[ToolCall]:
    if not isinstance(messages, list):
        messages = [messages]
    return [
        tc
        for m in messages
        if isinstance(m, AssistantMessage) and m.tool_calls
        for tc in m.tool_calls
    ]


def handle_tool_call(tool_call: ToolCall, tools: list[dict, Callable]) -> ToolMessage:
    tool_lookup = as_tool_lookup(tools)
    fn_name = tool_call.function.name
    fn_args = None
    metadata = {}
    try:
        if fn_name not in tool_lookup:
            fn_output = f'Function "{fn_name}" not found.'
            metadata["is_failed"] = True
        else:
            tool = tool_lookup[fn_name]
            metadata.update(tool._metadata)
            fn_args = tool_call.function.json_arguments()
            fn_output = tool(**fn_args)
    except Exception as exc:
        fn_output = f'Error calling function "{fn_name}": {exc}'
        metadata["is_failed"] = True
    return ToolMessage(
        content=output_to_string(fn_output),
        tool_call_id=tool_call.id,
        tool_call=tool_call,
        tool_result=fn_output,
        tool_metadata=metadata,
    )


async def handle_tool_call_async(
    tool_call: ToolCall, tools: list[dict, Callable]
) -> ToolMessage:
    tool_lookup = as_tool_lookup(tools)
    fn_name = tool_call.function.name
    fn_args = None
    metadata = {}
    try:
        if fn_name not in tool_lookup:
            fn_output = f'Function "{fn_name}" not found.'
            metadata["is_failed"] = True
        else:
            tool = tool_lookup[fn_name]
            metadata = tool._metadata
            fn_args = tool_call.function.json_arguments()
            fn_output = tool(**fn_args)
            if inspect.is_awaitable(fn_output):
                fn_output = await fn_output
    except Exception as exc:
        fn_output = f'Error calling function "{fn_name}": {exc}'
        metadata["is_failed"] = True
    return ToolMessage(
        content=output_to_string(fn_output),
        tool_call_id=tool_call.id,
        tool_call=tool_call,
        tool_result=fn_output,
        tool_metadata=metadata,
    )
