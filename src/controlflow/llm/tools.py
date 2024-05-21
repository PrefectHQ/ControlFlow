import inspect
import json
from functools import partial, update_wrapper
from typing import Any, Callable, Literal, Optional, Union

import litellm
import pydantic
from pydantic import PrivateAttr

from controlflow.utilities.types import ControlFlowModel, Message


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


def tool(
    fn: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Tool:
    if fn is None:
        return partial(tool, name=name, description=description)
    return Tool.from_function(fn, name=name, description=description)


def as_tools(tools: list[Union[Tool, Callable]]) -> list[Tool]:
    return [t if isinstance(t, Tool) else tool(t) for t in tools]


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


def has_tool_calls(response: litellm.ModelResponse) -> bool:
    """
    Check if the model response contains tool calls.
    """
    return bool(response.choices[0].message.get("tool_calls"))


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


def handle_tool_calls(response: litellm.ModelResponse, tools: list[dict, Callable]):
    messages = []
    tool_lookup = as_tool_lookup(tools)

    response_message = response.choices[0].message
    tool_calls: list[litellm.utils.ChatCompletionMessageToolCall] = (
        response_message.tool_calls
    )

    for tool_call in tool_calls:
        fn_name = tool_call.function.name
        try:
            if fn_name not in tool_lookup:
                raise ValueError(f'Function "{fn_name}" not found.')
            tool = tool_lookup[fn_name]
            fn_args = json.loads(tool_call.function.arguments)
            fn_output = tool(**fn_args)
        except Exception as exc:
            fn_output = f'Error calling function "{fn_name}": {exc}'
        messages.append(
            Message(
                role="tool",
                name=fn_name,
                content=output_to_string(fn_output),
                tool_call_id=tool_call.id,
            )
        )

    return messages


async def handle_tool_calls_async(
    response: litellm.ModelResponse, tools: list[dict, Callable]
):
    messages = []
    tool_lookup = as_tool_lookup(tools)

    response_message = response.choices[0].message
    tool_calls: list[litellm.utils.ChatCompletionMessageToolCall] = (
        response_message.tool_calls
    )

    for tool_call in tool_calls:
        fn_name = tool_call.function.name
        try:
            if fn_name not in tool_lookup:
                raise ValueError(f'Function "{fn_name}" not found.')
            tool = tool_lookup[fn_name]
            fn_args = json.loads(tool_call.function.arguments)
            fn_output = tool(**fn_args)
            if inspect.isawaitable(fn_output):
                fn_output = await fn_output
        except Exception as exc:
            fn_output = f'Error calling function "{fn_name}": {exc}'
        messages.append(
            Message(
                role="tool",
                name=fn_name,
                content=output_to_string(fn_output),
                tool_call_id=tool_call.id,
            )
        )

    return messages
