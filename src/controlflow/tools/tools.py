import functools
import inspect
import json
import typing
from typing import Annotated, Any, Callable, Optional, Union

import langchain_core.tools
import pydantic
from langchain_core.messages import InvalidToolCall, ToolCall
from prefect.utilities.asyncutils import run_coro_as_sync
from pydantic import Field, PydanticSchemaGenerationError, TypeAdapter

import controlflow
from controlflow.utilities.general import ControlFlowModel, unwrap
from controlflow.utilities.prefect import create_markdown_artifact, prefect_task

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
    name: str = Field(description="The name of the tool")
    description: str = Field(
        description="A description of the tool, which is provided to the LLM"
    )
    instructions: Optional[str] = Field(
        None,
        description="Optional instructions to display to the agent as part of the system prompt"
        " when this tool is available. Tool descriptions have a 1024 "
        "character limit, so this is a way to provide extra detail about behavior.",
    )
    parameters: dict = Field(
        description="The JSON schema for the tool's input parameters"
    )
    metadata: dict = {}

    fn: Callable = Field(None, exclude=True)
    _lc_tool: Optional[langchain_core.tools.BaseTool] = None

    def to_lc_tool(self) -> dict:
        payload = self.model_dump(include={"name", "description", "parameters"})
        return payload

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

    @prefect_task(task_run_name="Tool call: {self.name}")
    async def run_async(self, input: dict):
        result = self.fn(**input)
        if inspect.isawaitable(result):
            result = await result

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
        cls,
        fn: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        instructions: Optional[str] = None,
        include_param_descriptions: bool = True,
        include_return_description: bool = True,
        metadata: Optional[dict] = None,
        **kwargs,
    ):
        name = name or fn.__name__
        description = description or fn.__doc__ or ""
        signature = inspect.signature(fn)

        # If parameters are provided in kwargs, use those instead of generating them
        if "parameters" in kwargs:
            parameters = kwargs.pop("parameters")  # Custom parameters are respected
        else:
            try:
                parameters = TypeAdapter(fn).json_schema()
            except PydanticSchemaGenerationError:
                raise ValueError(
                    f'Could not generate a schema for tool "{name}". '
                    "Tool functions must have type hints that are compatible with Pydantic."
                )

        # load parameter descriptions
        if include_param_descriptions:
            for param in signature.parameters.values():
                # ensure we only try to add descriptions for parameters that exist in the schema
                if param.name not in parameters.get("properties", {}):
                    continue

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
                    parameters["properties"][param.name]["description"] = (
                        param_description
                    )

        # Handle return type description

        if (
            include_return_description
            and signature.return_annotation is not inspect._empty
        ):
            return_schema = {}
            try:
                return_schema.update(
                    TypeAdapter(signature.return_annotation).json_schema()
                )
            except PydanticSchemaGenerationError:
                pass
            finally:
                if typing.get_origin(signature.return_annotation) is Annotated:
                    return_schema["annotation"] = " ".join(
                        str(a) for a in typing.get_args(signature.return_annotation)[1:]
                    )

            if return_schema:
                description += f"\n\nReturn value schema: {return_schema}"

        if not description:
            description = "(No description provided)"

        if len(description) > 1024:
            raise ValueError(
                unwrap(f"""
                    {name}: The tool's description exceeds 1024
                    characters. Please provide a shorter description, fewer
                    annotations, or pass
                    `include_param_descriptions=False` or
                    `include_return_description=False` to `from_function`.
                    """)
            )

        return cls(
            name=name,
            description=description,
            parameters=parameters,
            fn=fn,
            instructions=instructions,
            metadata=metadata or {},
            **kwargs,
        )

    @classmethod
    def from_lc_tool(cls, tool: langchain_core.tools.BaseTool, **kwargs):
        fn = tool._run
        return cls(
            name=tool.name,
            description=tool.description,
            parameters=tool.args_schema.schema(),
            fn=fn,
            **kwargs,
        )

    def serialize_for_prompt(self) -> dict:
        return self.model_dump(include={"name", "description", "metadata"})


def tool(
    fn: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    instructions: Optional[str] = None,
    include_param_descriptions: bool = True,
    include_return_description: bool = True,
    metadata: Optional[dict] = None,
    **kwargs,
) -> Tool:
    """
    Decorator for turning a function into a Tool
    """
    kwargs.update(
        instructions=instructions,
        include_param_descriptions=include_param_descriptions,
        include_return_description=include_return_description,
        metadata=metadata or {},
    )
    if fn is None:
        return functools.partial(tool, name=name, description=description, **kwargs)
    return Tool.from_function(fn, name=name, description=description, **kwargs)


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
        elif inspect.isfunction(t) or inspect.ismethod(t):
            t = Tool.from_function(t)
        elif isinstance(t, dict):
            t = Tool(**t)
        else:
            raise ValueError(f"Invalid tool: {t}")

        if (t.name, t.description) in seen:
            continue
        new_tools.append(t)
        seen.add((t.name, t.description))
    return new_tools


def as_lc_tools(
    tools: list[Union[Callable, langchain_core.tools.BaseTool, Tool]],
) -> list[langchain_core.tools.BaseTool]:
    new_tools = []
    for t in tools:
        if isinstance(t, langchain_core.tools.BaseTool):
            pass
        elif isinstance(t, Tool):
            t = t.to_lc_tool()
        elif inspect.isfunction(t) or inspect.ismethod(t):
            t = langchain_core.tools.StructuredTool.from_function(t)
        else:
            raise ValueError(f"Invalid tool: {t}")
        new_tools.append(t)
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


class ToolResult(ControlFlowModel):
    tool_call: Union[ToolCall, InvalidToolCall]
    tool: Optional[Tool] = None
    result: Any = Field(exclude=True, repr=False)
    str_result: str = Field(repr=False)
    is_error: bool = False


def handle_tool_call(
    tool_call: Union[ToolCall, InvalidToolCall], tools: list[Tool]
) -> ToolResult:
    """
    Given a ToolCall and set of available tools, runs the tool call and returns
    a ToolResult object
    """
    is_error = False
    tool = None
    tool_lookup = {t.name: t for t in tools}
    fn_name = tool_call["name"]

    if fn_name not in tool_lookup:
        fn_output = f'There is no tool called "{fn_name}".'
        is_error = True
        if controlflow.settings.tools_raise_on_error:
            raise ValueError(fn_output)

    if not is_error:
        try:
            tool = tool_lookup[fn_name]
            fn_args = tool_call["args"]
            if isinstance(tool, Tool):
                fn_output = tool.run(input=fn_args)
            elif isinstance(tool, langchain_core.tools.BaseTool):
                fn_output = tool.invoke(input=fn_args)
            else:
                raise ValueError(f"Invalid tool: {tool}")
        except Exception as exc:
            fn_output = f'Error calling function "{fn_name}": {exc}'
            is_error = True
            if controlflow.settings.tools_raise_on_error:
                raise exc

    return ToolResult(
        tool_call=tool_call,
        tool=tool,
        result=fn_output,
        str_result=output_to_string(fn_output),
        is_error=is_error,
    )


async def handle_tool_call_async(tool_call: ToolCall, tools: list[Tool]) -> ToolResult:
    """
    Given a ToolCall and set of available tools, runs the tool call and returns
    a ToolResult object
    """
    is_error = False
    tool = None
    tool_lookup = {t.name: t for t in tools}
    fn_name = tool_call["name"]

    if fn_name not in tool_lookup:
        fn_output = f'There is no tool called "{fn_name}".'
        is_error = True
        if controlflow.settings.tools_raise_on_error:
            raise ValueError(fn_output)

    if not is_error:
        try:
            tool = tool_lookup[fn_name]
            fn_args = tool_call["args"]
            if isinstance(tool, Tool):
                fn_output = await tool.run_async(input=fn_args)
            elif isinstance(tool, langchain_core.tools.BaseTool):
                fn_output = await tool.ainvoke(input=fn_args)
            else:
                raise ValueError(f"Invalid tool: {tool}")
        except Exception as exc:
            fn_output = f'Error calling function "{fn_name}": {exc}'
            is_error = True
            if controlflow.settings.tools_raise_on_error:
                raise exc

    return ToolResult(
        tool_call=tool_call,
        tool=tool,
        result=fn_output,
        str_result=output_to_string(fn_output),
        is_error=is_error,
    )
