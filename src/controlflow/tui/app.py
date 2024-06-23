import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Union

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widgets import Footer, Header, Label

import controlflow
import controlflow.utilities
import controlflow.utilities.asyncio
from controlflow.llm.messages import AIMessage, ToolMessage, UserMessage

from .basic import Column, Row
from .task import TUITask
from .thread import TUIMessage, TUIToolMessage

if TYPE_CHECKING:
    import controlflow


class TUIApp(App):
    CSS_PATH = "app.tcss"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("h", "toggle_hold", "Hold"),
    ]

    agent: reactive["controlflow.Agent"] = reactive(None)
    hold: reactive[bool] = reactive(False)

    def __init__(self, flow: "controlflow.Flow", **kwargs):
        self._flow = flow
        self._tasks = flow.tasks
        self._is_ready = False
        super().__init__(**kwargs)

    @asynccontextmanager
    async def run_context(
        self,
        run: bool = True,
        inline: bool = True,
        inline_stay_visible: bool = True,
        headless: bool = None,
        hold: bool = False,
    ):
        if headless is None:
            headless = controlflow.settings.run_tui_headless

        if run:
            controlflow.utilities.asyncio.create_task(
                self.run_async(
                    inline=inline,
                    inline_no_clear=inline_stay_visible,
                    headless=headless,
                )
            )

            while not self._is_ready:
                await asyncio.sleep(0.01)

            if hold is not None:
                self.hold = hold

        try:
            yield self
        except Exception:
            self.hold = False
            raise
        finally:
            if run:
                while self.hold:
                    await asyncio.sleep(0.01)
                self.exit()

    def exit(self, *args, **kwargs):
        self.hold = False
        self._is_ready = False
        return super().exit(*args, **kwargs)

    def action_toggle_hold(self):
        self.hold = not self.hold

    def watch_hold(self, hold: bool):
        try:
            if hold:
                self.query_one("#hold-banner").display = "block"
            else:
                self.query_one("#hold-banner").display = "none"
        except NoMatches:
            pass

    def on_mount(self):
        if self._flow.name:
            self.title = f"ControlFlow: {self._flow.name}"
        else:
            self.title = "ControlFlow"
        # self.sub_title = "With title and sub-title"
        self._is_ready = True

    # ---------------------------
    #
    # Interaction methods
    #
    # ---------------------------

    def update_task(self, task: "controlflow.Task"):
        try:
            component = self.query_one(f"#task-{task.id}", TUITask)
            component.task = task
            component.scroll_visible()
        except NoMatches:
            new_task = TUITask(task=task, id=f"task-{task.id}")
            self.query_one("#tasks-container", Column).mount(new_task)
            new_task.scroll_visible()

    def update_message(self, message: Union[UserMessage, AIMessage]):
        try:
            component = self.query_one(f"#message-{message.id}", TUIMessage)
            component.message = message
            component.scroll_visible()
        except NoMatches:
            new_message = TUIMessage(message=message, id=f"message-{message.id}")
            self.query_one("#thread-container", Column).mount(new_message)
            new_message.scroll_visible()

    def update_tool_result(self, message: ToolMessage):
        try:
            component = self.query_one(f"#message-{message.id}", TUIToolMessage)
            component.message = message
            component.scroll_visible()
        except NoMatches:
            new_step = TUIToolMessage(message=message, id=f"message-{message.id}")
            self.query_one("#thread-container", Column).mount(new_step)
            new_step.scroll_visible()

    def set_agent(self, agent: "controlflow.Agent"):
        self.agent = agent

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="app-container"):
            with Row(id="hold-banner"):
                yield Label(
                    "TUI will stay running after the run ends. Press 'h' to toggle."
                )
            with Row(id="main-container"):
                with Column(id="tasks"):
                    yield Label("Tasks", id="tasks-title", classes="title")
                    with Column(id="tasks-container"):
                        for task in self._tasks.values():
                            yield TUITask(task=task, id=f"task-{task.id}")
                yield Column(id="separator")
                with Column(id="thread"):
                    yield Label("Thread", id="thread-title", classes="title")
                    yield Column(id="thread-container")

        yield Footer()

    # async def get_input(self, message: str, container: list):
    #     self.query_one("#input-container").display = "block"
    #     self._container = container

    # @on(Button.Pressed, "#submit-input")
    # def submit_input(self):
    #     text = self.query_one("#input", TextArea).text
    #     self._container.append(text)
    #     self.query_one("#input-container").display = "none"
