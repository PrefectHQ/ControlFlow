import controlflow
import prefect
from controlflow.cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert f"ControlFlow version: {controlflow.__version__}" in result.stdout
    assert f"Prefect version: {prefect.__version__}" in result.stdout
