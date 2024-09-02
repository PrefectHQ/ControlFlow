import prefect
from typer.testing import CliRunner

import controlflow
from controlflow.cli.main import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert f"ControlFlow version: {controlflow.__version__}" in result.stdout
    assert f"Prefect version: {prefect.__version__}" in result.stdout
