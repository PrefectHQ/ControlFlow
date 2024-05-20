from textual.containers import Horizontal, VerticalScroll


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
