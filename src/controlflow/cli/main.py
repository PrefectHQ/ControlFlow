import platform
from pathlib import Path

import langchain_core
import prefect
import typer
from rich.table import Table
from typer import Context, Exit

from controlflow import __version__
from controlflow.utilities.rich import console

from .dev import dev_app

app = typer.Typer(no_args_is_help=True)

app.add_typer(dev_app, name="dev")


@app.command()
def version(ctx: Context):
    if ctx.resilient_parsing:
        return

    info = {
        "ControlFlow version": __version__,
        "Prefect version": prefect.__version__,
        "LangChain Core version": langchain_core.__version__,
        "Python version": platform.python_version(),
        "Platform": platform.platform(),
        "Path": Path(__file__).resolve().parents[3],
    }

    g = Table.grid(padding=(0, 1))
    g.add_column(justify="right")
    g.add_column()
    for k, v in info.items():
        g.add_row(k + ":", str(v).replace("\n", " "))
    console.print(g)

    raise Exit()


# this callback only exists to force `version` to be called as `controlflow
# version` instead of as the default command, which is the default behavior when
# there's only one command. It can be deleted if/when more commands are added.
@app.callback()
def callback():
    pass


if __name__ == "__main__":
    app()
