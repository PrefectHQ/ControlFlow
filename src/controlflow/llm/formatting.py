import datetime
import inspect

from rich import box
from rich.console import Group
from rich.markdown import Markdown
from rich.panel import Panel

from controlflow.llm.messages import (
    AIMessage,
    MessageType,
    ToolMessage,
)

ROLE_COLORS = {
    "system": "gray",
    "user": "green",
    "assistant": "blue",
}
ROLE_NAMES = {
    "system": "System",
    "assistant": "Agent",
    "user": "User",
}


def format_timestamp(timestamp: datetime.datetime) -> str:
    return timestamp.strftime("%l:%M:%S %p")


def format_message(
    message: MessageType,
) -> Panel:
    if isinstance(message, ToolMessage):
        return format_tool_message(message)
    elif isinstance(message, AIMessage) and message.tool_calls:
        return format_ai_message_with_tool_calls(message)
    else:
        return format_text_message(message)


def format_text_message(message: MessageType) -> Panel:
    if message.role == "assistant" and message.name:
        title = f"Agent: {message.name}"
    else:
        title = ROLE_NAMES.get(message.role, "Agent")

    return Panel(
        Markdown(message.content or ""),
        title=f"[bold]{title}[/]",
        subtitle=f"[italic]{format_timestamp(message.timestamp)}[/]",
        title_align="left",
        subtitle_align="right",
        border_style=ROLE_COLORS.get(message.role, "red"),
        box=box.ROUNDED,
        width=100,
        expand=True,
        padding=(1, 2),
    )


def format_ai_message_with_tool_calls(message: AIMessage) -> Group:
    panels = []
    for tool_call in message.tool_calls:
        if message.role == "assistant" and message.name:
            title = f"Tool Call: {message.name}"
        else:
            title = "Tool Call"

        content = Markdown(
            inspect.cleandoc("""
                🦾 Calling `{name}` with the following arguments:
                
                ```json
                {args}
                ```
                """).format(name=tool_call["name"], args=tool_call["args"])
        )

        panels.append(
            Panel(
                content,
                title=f"[bold]{title}[/]",
                subtitle=f"[italic]{format_timestamp(message.timestamp)}[/]",
                title_align="left",
                subtitle_align="right",
                border_style=ROLE_COLORS.get(message.role, "red"),
                box=box.ROUNDED,
                width=100,
                expand=True,
                padding=(1, 2),
            )
        )
    return Group(*panels)


def format_tool_message(message: ToolMessage) -> Panel:
    if message.tool_metadata.get("is_failed"):
        content = (
            f"❌ The tool call to [markdown.code]{message.tool_call['name']}[/] failed."
        )
    elif not message.tool_metadata.get("is_task_status_tool"):
        content_type = "json" if isinstance(message.tool_result, (dict, list)) else ""
        content = Group(
            f"✅ Received output from the [markdown.code]{message.tool_call['name']}[/] tool.\n",
            Markdown(f"```{content_type}\n{message.content or ''}\n```"),
        )
    else:
        return ""

    return Panel(
        content,
        title="Tool Call Result",
        subtitle=f"[italic]{format_timestamp(message.timestamp)}[/]",
        title_align="left",
        subtitle_align="right",
        border_style="blue",
        box=box.ROUNDED,
        width=100,
        expand=True,
        padding=(1, 2),
    )
