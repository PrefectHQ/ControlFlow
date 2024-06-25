import functools
import inspect
import json
import typing
from typing import TYPE_CHECKING, Annotated, Any, Callable, Optional, Union

import langchain_core.tools
import pydantic
import pydantic.v1
from langchain_core.messages import ToolCall
from prefect.utilities.asyncutils import run_coro_as_sync
from pydantic import Field, TypeAdapter

import controlflow
from controlflow.llm.messages import ToolMessage
from controlflow.utilities.prefect import create_markdown_artifact, prefect_task
from controlflow.utilities.types import ControlFlowModel

if TYPE_CHECKING:
    from controlflow.agents import Agent


TOOL_CALL_FUNCTION_RESULT_TEMPLATE = """
# Tool call: {name}

**Description:** {description}

## Arguments

```json
{args}
```

## Result

```
{result}
```
"""


class Tool(ControlFlowModel):
    name: str
    description: str
    parameters: dict
    metadata: dict = Field({}, exclude_none=True)
    fn: Callable = Field(None, exclude=True)

    def to_lc_tool(self) -> dict:
        return self.model_dump(include={"name", "description", "parameters"})

    @prefect_task(task_run_name="Tool call: {self.name}")
    def run(self, input: dict):
        result = self.fn(**input)
        if inspect.isawaitable(result):
            result = run_coro_as_sync(result)

        # prepare artifact
        passed_args = inspect.signature(self.fn).bind(**input).arguments
        try:
            # try to pretty print the args
            passed_args = json.dumps(passed_args, indent=2)
        except Exception:
            pass
        create_markdown_artifact(
            markdown=TOOL_CALL_FUNCTION_RESULT_TEMPLATE.format(
                name=self.name,
                description=self.description or "(none provided)",
                args=passed_args,
                result=result,
            ),
            key="tool-result",
        )
        return result

    @classmethod
    def from_function(
        cls, fn: Callable, name: str = None, description: str = None, **kwargs
    ):
        description = description or fn.__doc__ or "(No description provided)"

        signature = inspect.signature(fn)
        parameters = TypeAdapter(fn).json_schema()

        # load parameter descriptions
        for param in signature.parameters.values():
            # handle Annotated type hints
            if typing.get_origin(param.annotation) is Annotated:
                param_description = " ".join(
                    str(a) for a in typing.get_args(param.annotation)[1:]
                )
            # handle pydantic Field descriptions
            elif param.default is not inspect.Parameter.empty and isinstance(
                param.default, pydantic.fields.FieldInfo
            ):
                param_description = param.default.description
            else:
                param_description = None

            if param_description:
                parameters["properties"][param.name]["description"] = param_description

        # Handle return type description
        return_type = signature.return_annotation
        if return_type is not inspect._empty:
            return_schema = TypeAdapter(return_type).json_schema()
            description += f"\n\nReturn value schema: {return_schema}"

        return cls(
            name=name or fn.__name__,
            description=description,
            parameters=parameters,
            fn=fn,
            **kwargs,
        )

    @classmethod
    def from_lc_tool(cls, tool: langchain_core.tools.BaseTool, **kwargs):
        if isinstance(tool, langchain_core.tools.StructuredTool):
            fn = tool.func
        else:
            fn = lambda *a, **k: None  # noqa
        return cls(
            name=tool.name,
            description=tool.description,
            parameters=tool.args_schema.schema(),
            fn=fn,
            **kwargs,
        )


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
    tools: list[Union[Callable, langchain_core.tools.BaseTool, Tool]],
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
        if isinstance(t, Tool):
            pass
        elif isinstance(t, langchain_core.tools.BaseTool):
            t = Tool.from_lc_tool(t)
        elif inspect.isfunction(t):
            t = Tool.from_function(t)
        elif isinstance(t, dict):
            t = Tool(**t)
        else:
            raise ValueError(f"Invalid tool: {t}")

        if (t.name, t.description, t.fn) in seen:
            continue
        new_tools.append(t)
        seen.add((t.name, t.description, t.fn))
    return new_tools


def output_to_string(output: Any) -> str:
    """
    Function outputs must be provided as strings
    """
    if output is None:
        return ""
    elif isinstance(output, str):
        return output
    try:
        return pydantic.TypeAdapter(type(output)).dump_json(output).decode()
    except Exception:
        return str(output)


def handle_tool_call(
    tool_call: ToolCall,
    tools: list[Tool],
    error: str = None,
    agent: "Agent" = None,
) -> ToolMessage:
    tool_lookup = {t.name: t for t in tools}
    fn_name = tool_call["name"]
    is_error = False
    metadata = {}
    try:
        if error:
            fn_output = error
            is_error = True
        elif fn_name not in tool_lookup:
            fn_output = f'Function "{fn_name}" not found.'
            is_error = True
        else:
            tool = tool_lookup[fn_name]
            metadata.update(getattr(tool, "metadata", {}))
            fn_args = tool_call["args"]
            if isinstance(tool, Tool):
                fn_output = tool.run(input=fn_args)
            elif isinstance(tool, langchain_core.tools.BaseTool):
                fn_output = tool.invoke(input=fn_args)
    except Exception as exc:
        fn_output = f'Error calling function "{fn_name}": {exc}'
        is_error = True
        if controlflow.settings.tools_raise_on_error:
            raise

    from controlflow.llm.messages import ToolMessage

    return ToolMessage(
        content=output_to_string(fn_output),
        tool_call_id=tool_call["id"],
        tool_call=tool_call,
        tool_result=fn_output,
        tool_metadata=metadata,
        is_error=is_error,
        agent=agent,
    )
