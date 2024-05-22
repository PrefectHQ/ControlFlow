import functools
import inspect
import json
from functools import partial, update_wrapper
from typing import Any, Callable, Optional, Union

import litellm
import pydantic

from controlflow.utilities.types import Message, Tool, ToolResult


def tool(
    fn: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Tool:
    if fn is None:
        return partial(tool, name=name, description=description)
    return Tool.from_function(fn, name=name, description=description)


def annotate_fn(
    fn: Callable, name: Optional[str], description: Optional[str]
) -> Callable:
    """
    Annotate a function with a new name and description without modifying the
    original. Useful when you want to provide a custom name and description for
    a tool, but without creating a new tool object.
    """
    new_fn = functools.partial(fn)
    new_fn.__name__ = name or fn.__name__
    new_fn.__doc__ = description or fn.__doc__
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


def get_tool_calls(
    response: litellm.ModelResponse,
) -> list[litellm.utils.ChatCompletionMessageToolCall]:
    return response.choices[0].message.get("tool_calls", [])


def handle_tool_call(
    tool_call: litellm.utils.ChatCompletionMessageToolCall, tools: list[dict, Callable]
) -> Message:
    tool_lookup = as_tool_lookup(tools)
    fn_name = tool_call.function.name
    fn_args = (None,)
    try:
        is_error = False
        if fn_name not in tool_lookup:
            fn_output = f'Function "{fn_name}" not found.'
            is_error = True
        else:
            tool = tool_lookup[fn_name]
            fn_args = json.loads(tool_call.function.arguments)
            fn_output = tool(**fn_args)
    except Exception as exc:
        fn_output = f'Error calling function "{fn_name}": {exc}'
        is_error = True
    return Message(
        role="tool",
        name=fn_name,
        content=output_to_string(fn_output),
        tool_call_id=tool_call.id,
        tool_result=ToolResult(
            tool_call_id=tool_call.id,
            tool_name=fn_name,
            tool=tool,
            args=fn_args,
            result=fn_output,
        ),
    )


async def handle_tool_call_async(
    tool_call: litellm.utils.ChatCompletionMessageToolCall, tools: list[dict, Callable]
) -> Message:
    tool_lookup = as_tool_lookup(tools)
    fn_name = tool_call.function.name
    fn_args = (None,)
    try:
        is_error = False
        if fn_name not in tool_lookup:
            fn_output = f'Function "{fn_name}" not found.'
            is_error = True
        else:
            tool = tool_lookup[fn_name]
            fn_args = json.loads(tool_call.function.arguments)
            fn_output = tool(**fn_args)
            if inspect.is_awaitable(fn_output):
                fn_output = await fn_output
    except Exception as exc:
        fn_output = f'Error calling function "{fn_name}": {exc}'
        is_error = True
    return Message(
        role="tool",
        name=fn_name,
        content=output_to_string(fn_output),
        tool_call_id=tool_call.id,
        tool_result=ToolResult(
            tool_call_id=tool_call.id,
            tool_name=fn_name,
            tool=tool,
            args=fn_args,
            is_error=is_error,
            result=fn_output,
        ),
    )
