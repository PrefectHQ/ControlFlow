import functools
import inspect
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import langchain_core
import langchain_core.tools
import pydantic
import pydantic.v1
from langchain_core.messages import InvalidToolCall, ToolCall
from prefect.utilities.asyncutils import run_coro_as_sync
from pydantic import Field, create_model

from controlflow.llm.messages import InvalidToolMessage

if TYPE_CHECKING:
    from controlflow.llm.messages import ToolMessage


def pydantic_model_from_function(fn: Callable):
    sig = inspect.signature(fn)
    fields = {}
    for name, param in sig.parameters.items():
        annotation = (
            param.annotation if param.annotation is not inspect.Parameter.empty else Any
        )
        default = param.default if param.default is not inspect.Parameter.empty else ...
        fields[name] = (annotation, Field(default=default))
    return create_model(fn.__name__, **fields)


def _sync_wrapper(coro):
    """
    Wrapper that runs a coroutine as a synchronous function with deffered args
    """

    @functools.wraps(coro)
    def wrapper(*args, **kwargs):
        return run_coro_as_sync(coro(*args, **kwargs))

    return wrapper


class Tool(langchain_core.tools.StructuredTool):
    """
    A subclass of StructuredTool that is compatible with functions whose
    signatures include either Pydantic v1 models (which Langchain uses) or v2
    models (which ControlFlow users).

    Note that THIS class is a Pydantic v1 model because it subclasses the Langchain
    class.
    """

    tags: dict[str, Any] = pydantic.v1.Field(default_factory=dict)
    args_schema: Optional[type[Union[pydantic.v1.BaseModel, pydantic.BaseModel]]]

    @classmethod
    def from_function(cls, fn=None, *args, **kwargs):
        args_schema = pydantic_model_from_function(fn)
        if inspect.iscoroutinefunction(fn):
            fn, coro = _sync_wrapper(fn), fn
        else:
            coro = None
        return super().from_function(
            *args, func=fn, coroutine=coro, args_schema=args_schema, **kwargs
        )


def tool(
    fn: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[dict] = None,
) -> Tool:
    if fn is None:
        return functools.partial(tool, name=name, description=description, tags=tags)
    return Tool.from_function(fn, name=name, description=description, tags=tags or {})


def as_tools(tools: list[Union[Callable, Tool]]) -> list[Tool]:
    new_tools = []
    for t in tools:
        if not isinstance(t, Tool):
            t = Tool.from_function(t)
        new_tools.append(t)
    return new_tools


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


def handle_tool_call(tool_call: ToolCall, tools: list[Tool]) -> "ToolMessage":
    tool_lookup = {t.name: t for t in tools}
    fn_name = tool_call["name"]
    metadata = {}
    try:
        if fn_name not in tool_lookup:
            fn_output = f'Function "{fn_name}" not found.'
            metadata["is_failed"] = True
        else:
            tool = tool_lookup[fn_name]
            fn_args = tool_call["args"]
            fn_output = tool.invoke(input=fn_args)
            if inspect.isawaitable(fn_output):
                fn_output = run_coro_as_sync(fn_output)
    except Exception as exc:
        fn_output = f'Error calling function "{fn_name}": {exc}'
        metadata["is_failed"] = True

    from controlflow.llm.messages import ToolMessage

    return ToolMessage(
        content=output_to_string(fn_output),
        tool_call_id=tool_call["id"],
        tool_call=tool_call,
        tool_result=fn_output,
        tool_metadata=metadata,
    )


async def handle_tool_call_async(
    tool_call: ToolCall, tools: list[Tool]
) -> "ToolMessage":
    tool_lookup = {t.name: t for t in tools}
    fn_name = tool_call["name"]
    metadata = {}
    try:
        if fn_name not in tool_lookup:
            fn_output = f'Function "{fn_name}" not found.'
            metadata["is_failed"] = True
        else:
            tool = tool_lookup[fn_name]
            fn_args = tool_call["args"]
            fn_output = await tool.ainvoke(input=fn_args)
    except Exception as exc:
        fn_output = f'Error calling function "{fn_name}": {exc}'
        metadata["is_failed"] = True

    from controlflow.llm.messages import ToolMessage

    return ToolMessage(
        content=output_to_string(fn_output),
        tool_call_id=tool_call["id"],
        tool_call=tool_call,
        tool_result=fn_output,
        tool_metadata=metadata,
    )


def handle_invalid_tool_call(tool_call: InvalidToolCall) -> "ToolMessage":
    return InvalidToolMessage(
        content=tool_call["error"] or "",
        tool_call_id=tool_call["id"],
        tool_call=tool_call,
        tool_result=tool_call["error"],
        tool_metadata=dict(is_failed=True, is_invalid=True),
    )
