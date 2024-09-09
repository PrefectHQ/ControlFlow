import os
import subprocess
from pathlib import Path

import typer

dev_app = typer.Typer(no_args_is_help=True)


@dev_app.command()
def generate_ai_files(
    output_path: str = typer.Option(
        ".",
        "--output",
        "-o",
        help="The path where output files will be written. Defaults to current directory.",
    ),
):
    """
    Generates two markdown files that contain all of ControlFlow's source code and documentation,
    which can be used to provide context to an AI.
    """
    try:
        # Get the absolute path of the ControlFlow main repo
        repo_root = Path(__file__).resolve().parents[3]
        src_path = repo_root / "src"
        docs_path = repo_root / "docs"
        output_dir = Path(output_path).resolve()

        typer.echo(f"Repo root: {repo_root}")
        typer.echo(f"src_path: {src_path}")
        typer.echo(f"docs_path: {docs_path}")
        typer.echo(f"output_dir: {output_dir}")

        def generate_file_content(file_paths, output_file):
            with open(output_dir / output_file, "w") as f:
                for file_path in file_paths:
                    f.write(f"# ControlFlow Source File: {file_path.absolute()}\n\n")
                    f.write(file_path.read_text())
                    f.write("\n\n")

        code_files = list(src_path.rglob("*.py")) + list(src_path.rglob("*.jinja"))
        doc_files = list(docs_path.rglob("*.mdx")) + list(docs_path.glob("mint.json"))

        generate_file_content(code_files, "all_code.md")
        generate_file_content(doc_files, "all_docs.md")

        typer.echo(f"Generated all_code.md and all_docs.md in {output_dir}")
    except Exception as e:
        typer.echo(f"An error occurred: {str(e)}", err=True)
        raise typer.Exit(code=1)


@dev_app.command()
def docs():
    """
    This is equivalent to 'cd docs && mintlify dev' from the ControlFlow root.
    """
    try:
        # Get the absolute path of the ControlFlow main repo
        repo_root = Path(__file__).resolve().parents[3]
        docs_path = repo_root / "docs"

        if not docs_path.exists():
            typer.echo(f"Error: Docs directory not found at {docs_path}", err=True)
            raise typer.Exit(code=1)

        typer.echo(f"Changing directory to: {docs_path}")
        os.chdir(docs_path)

        typer.echo("Running 'mintlify dev'...")
        subprocess.run(["mintlify", "dev"], check=True)

    except subprocess.CalledProcessError as e:
        typer.echo(f"Error running 'mintlify dev': {str(e)}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"An error occurred: {str(e)}", err=True)
        raise typer.Exit(code=1)
