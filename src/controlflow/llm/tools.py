import functools
import inspect
import typing
from typing import TYPE_CHECKING, Annotated, Any, Callable, Optional, Union

import langchain_core
import langchain_core.tools
import pydantic
import pydantic.v1
from langchain_core.messages import InvalidToolCall, ToolCall
from prefect.utilities.asyncutils import run_coro_as_sync
from pydantic import TypeAdapter

import controlflow
from controlflow.llm.messages import InvalidToolMessage
from controlflow.utilities.prefect import wrap_prefect_tool

if TYPE_CHECKING:
    from controlflow.llm.messages import ToolMessage


class FnArgsSchema:
    """
    A dropin replacement for LangChain's StructuredTool args_schema objects
    that can be created from any function that has type hints.
    Used in ControlFlow to create Tool objects from functions.
    """

    def __init__(self, fn):
        self.fn = fn

    def schema(self) -> dict:
        schema = TypeAdapter(self.fn).json_schema()

        # load parameter descriptions
        for param in inspect.signature(self.fn).parameters.values():
            # handle Annotated type hints
            if (
                # param.annotation is not inspect.Parameter.empty
                # and
                typing.get_origin(param.annotation) is Annotated
            ):
                description = " ".join(
                    str(a) for a in typing.get_args(param.annotation)[1:]
                )

            # handle pydantic Field descriptions
            elif param.default is not inspect.Parameter.empty and isinstance(
                param.default, pydantic.fields.FieldInfo
            ):
                description = param.default.description
            else:
                continue

            schema["properties"][param.name]["description"] = description

        return schema

    def parse_obj(self, args_dict: dict):
        # use validate call to parse the args
        # using a function with the same signature as the real one, but just returning the kwargs
        @pydantic.validate_call
        @functools.wraps(self.fn)
        def get_kwargs(**kwargs):
            return kwargs

        return get_kwargs(**args_dict)


def _sync_wrapper(coro):
    """
    Wrapper that runs a coroutine as a synchronous function with deferred args
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

    args_schema: Optional[
        Union[type[pydantic.v1.BaseModel], type[pydantic.BaseModel], FnArgsSchema]
    ]
    metadata: dict[str, Any] = pydantic.v1.Field(default_factory=dict)

    @classmethod
    def from_function(cls, fn=None, *args, **kwargs):
        args_schema = FnArgsSchema(fn)
        if not fn.__doc__ and not kwargs.get("description"):
            kwargs["description"] = fn.__name__
        if inspect.iscoroutinefunction(fn):
            fn, coro = _sync_wrapper(fn), fn
        else:
            coro = None

        return super().from_function(
            *args, func=fn, coroutine=coro, args_schema=args_schema, **kwargs
        )

    def _parse_input(self, tool_input: Union[str, dict]) -> Union[str, dict[str, Any]]:
        if isinstance(self.args_schema, FnArgsSchema):
            return self.args_schema.parse_obj(tool_input)
        else:
            return super()._parse_input(tool_input)


def tool(
    fn: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Tool:
    """
    Decorator for turning a function into a Tool
    """
    if fn is None:
        return functools.partial(
            tool, name=name, description=description, metadata=metadata
        )
    return Tool.from_function(
        fn, name=name, description=description, metadata=metadata or {}
    )


def as_tools(
    tools: list[Union[Callable, Tool]], wrap_prefect: bool = True
) -> list[Tool]:
    """
    Converts a list of tools (either Tool objects or callables) into a list of
    Tool objects.

    If duplicate tools are found, where the name, function, and coroutine are
    the same, only one is kept.
    """
    seen = set()
    new_tools = []
    for t in tools:
        if not isinstance(t, Tool):
            t = Tool.from_function(t)
        if (t.name, t.func, t.coroutine) in seen:
            continue
        if wrap_prefect:
            t = wrap_prefect_tool(t)
        new_tools.append(t)
        seen.add((t.name, t.func, t.coroutine))
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


def handle_tool_call(
    tool_call: ToolCall,
    tools: list[Tool],
    error: str = None,
    agent_id: str = None,
) -> "ToolMessage":
    tool_lookup = {t.name: t for t in tools}
    fn_name = tool_call["name"]
    metadata = {}
    try:
        if error:
            fn_output = error
            metadata["is_failed"] = True
        elif fn_name not in tool_lookup:
            fn_output = f'Function "{fn_name}" not found.'
            metadata["is_failed"] = True
        else:
            tool = tool_lookup[fn_name]
            metadata.update(getattr(tool, "metadata", {}))
            fn_args = tool_call["args"]
            fn_output = tool.invoke(input=fn_args)
            if inspect.isawaitable(fn_output):
                fn_output = run_coro_as_sync(fn_output)
    except Exception as exc:
        fn_output = f'Error calling function "{fn_name}": {exc}'
        metadata["is_failed"] = True
        if controlflow.settings.raise_on_tool_error:
            raise

    from controlflow.llm.messages import ToolMessage

    return ToolMessage(
        content=output_to_string(fn_output),
        tool_call_id=tool_call["id"],
        tool_call=tool_call,
        tool_result=fn_output,
        tool_metadata=metadata,
        agent_id=agent_id,
    )


async def handle_tool_call_async(
    tool_call: ToolCall,
    tools: list[Tool],
    error: str = None,
    agent_id: str = None,
) -> "ToolMessage":
    tool_lookup = {t.name: t for t in tools}
    fn_name = tool_call["name"]
    metadata = {}
    try:
        if error:
            fn_output = error
            metadata["is_failed"] = True
        elif fn_name not in tool_lookup:
            fn_output = f'Function "{fn_name}" not found.'
            metadata["is_failed"] = True
        else:
            tool = tool_lookup[fn_name]
            metadata.update(getattr(tool, "metadata", {}))
            fn_args = tool_call["args"]
            fn_output = await tool.ainvoke(input=fn_args)
    except Exception as exc:
        fn_output = f'Error calling function "{fn_name}": {exc}'
        metadata["is_failed"] = True
        if controlflow.settings.raise_on_tool_error:
            raise

    from controlflow.llm.messages import ToolMessage

    return ToolMessage(
        content=output_to_string(fn_output),
        tool_call_id=tool_call["id"],
        tool_call=tool_call,
        tool_result=fn_output,
        tool_metadata=metadata,
        agent_id=agent_id,
    )


def handle_invalid_tool_call(
    tool_call: InvalidToolCall, agent_id: str = None
) -> "ToolMessage":
    return InvalidToolMessage(
        content=tool_call["error"] or "",
        tool_call_id=tool_call["id"],
        tool_call=tool_call,
        tool_result=tool_call["error"],
        tool_metadata=dict(is_failed=True, is_invalid=True),
        agent_id=agent_id,
    )
