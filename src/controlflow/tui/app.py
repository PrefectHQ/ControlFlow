import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import openai.types.beta.threads
import openai.types.beta.threads.runs.run_step
from textual.app import App as TextualApp
from textual.app import ComposeResult
from textual.containers import Container
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widgets import Footer, Header, Label

import controlflow
from controlflow.utilities.context import ctx

from .basic import Column, Row
from .task import Task
from .thread import Message, RunStep

if TYPE_CHECKING:
    import controlflow


class App(TextualApp):
    CSS_PATH = "app.tcss"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("h", "toggle_hold", "Hold"),
    ]

    agent: reactive["controlflow.Agent"] = reactive(None)
    hold: reactive[bool] = reactive(False)

    def __init__(self, flow: "controlflow.Flow", **kwargs):
        self._flow = flow
        self._tasks = flow._tasks
        self._is_ready = False
        super().__init__(**kwargs)

    @asynccontextmanager
    async def run_context(self, run: bool = True, inline=True, inline_no_clear=True):
        with ctx(tui=self):
            if run:
                asyncio.create_task(
                    self.run_async(inline=inline, inline_no_clear=inline_no_clear)
                )

                while not self._is_ready:
                    await asyncio.sleep(0.01)

            try:
                yield self
            finally:
                if run:
                    while self.hold:
                        await asyncio.sleep(0.01)
                    self.exit()

    def exit(self, *args, **kwargs):
        self._is_ready = False
        return super().exit(*args, **kwargs)

    def action_toggle_hold(self):
        self.hold = not self.hold

    def watch_hold(self, hold: bool):
        if hold:
            self.query_one("#hold-banner").display = "block"
        else:
            self.query_one("#hold-banner").display = "none"

    def on_mount(self):
        self.title = self._flow.name or "ControlFlow"
        # self.sub_title = "With title and sub-title"
        self._is_ready = True
        # for task in self._tasks.values():
        #     self.update_task(task)

    def update_task(self, task: "controlflow.Task"):
        try:
            component = self.query_one(f"#task-{task.id}", Task)
            component.task = task
            component.scroll_visible()
        except NoMatches:
            self.add_task(task)

    def add_task(self, task: "controlflow.Task"):
        if not self._is_ready:
            return
        new_task = Task(task=task, id=f"task-{task.id}")
        self.query_one("#tasks", Column).mount(new_task)
        new_task.scroll_visible()

    def update_message(self, message: openai.types.beta.threads.Message):
        try:
            component = self.query_one(f"#message-{message.id}", Message)
            component.message = message
            component.scroll_visible()
        except NoMatches:
            self.add_message(message)

    def add_message(self, message: openai.types.beta.threads.Message):
        if not self._is_ready:
            return
        new_message = Message(message=message, id=f"message-{message.id}")
        self.query_one("#thread", Column).mount(new_message)
        new_message.scroll_visible()

    def update_step(self, step: openai.types.beta.threads.runs.run_step.RunStep):
        try:
            component = self.query_one(f"#step-{step.id}", RunStep)
            component.step = step
            component.scroll_visible()
        except NoMatches:
            self.add_step(step)

    def add_step(self, step: openai.types.beta.threads.runs.run_step.RunStep):
        if not self._is_ready:
            return
        new_step = RunStep(step=step, id=f"step-{step.id}")
        self.query_one("#thread", Column).mount(new_step)
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
                    for task in self._tasks.values():
                        yield Task(task=task, id=f"task-{task.id}")

                yield Column(id="thread")
            # with Vertical(id="input-container"):
            #     yield TextArea(id="input")
            #     yield Button("Submit", variant="primary", id="submit-input")

        yield Footer()

    # async def get_input(self, message: str, container: list):
    #     self.query_one("#input-container").display = "block"
    #     self._container = container

    # @on(Button.Pressed, "#submit-input")
    # def submit_input(self):
    #     text = self.query_one("#input", TextArea).text
    #     self._container.append(text)
    #     self.query_one("#input-container").display = "none"
