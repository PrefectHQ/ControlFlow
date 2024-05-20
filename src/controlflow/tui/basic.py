from textual.containers import Horizontal, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Label


class Row(Horizontal):
    DEFAULT_CLASSES = "row"
    DEFAULT_CSS = """
        Row {
            height: auto;
        }
    """


class Column(VerticalScroll):
    DEFAULT_CLASSES = "column"
    DEFAULT_CSS = """
        Column {
            height: auto;
        }
    """


class ReactiveLabel(Label):
    value = reactive(None)

    def render(self):
        return str(self.value)
