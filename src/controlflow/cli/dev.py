import os
import subprocess
from pathlib import Path

import typer

dev_app = typer.Typer(no_args_is_help=True)


@dev_app.command()
def ai_files(
    output_path: str = typer.Option(
        ".",
        "--output",
        "-o",
        help="The path where output files will be written. Defaults to current directory.",
    ),
):
    """
    Generates three markdown files that contain all of ControlFlow's source code, documentation,
    and LLM guides, which can be used to provide context to an AI.
    """
    try:
        # Get the absolute path of the ControlFlow main repo
        repo_root = Path(__file__).resolve().parents[3]
        src_path = repo_root / "src"
        docs_path = repo_root / "docs"
        llm_guides_path = docs_path / "llm-guides"
        output_dir = Path(output_path).resolve()

        def generate_file_content(file_paths, output_file):
            with open(output_dir / output_file, "w") as f:
                for file_path in file_paths:
                    f.write(f"# ControlFlow Source File: {file_path.absolute()}\n\n")
                    f.write(file_path.read_text())
                    f.write("\n\n")

        code_files = list(src_path.rglob("*.py")) + list(src_path.rglob("*.jinja"))
        doc_files = (
            list(docs_path.rglob("*.mdx"))
            + list(docs_path.glob("*.md"))
            + [docs_path / "mint.json", repo_root / "README.md"]
        )
        llm_guide_files = list(llm_guides_path.glob("*.md")) + list(
            llm_guides_path.glob("*.mdx")
        )

        generate_file_content(code_files, "all_code.md")
        generate_file_content(doc_files, "all_docs.md")
        generate_file_content(llm_guide_files, "llm_guides.md")

        typer.echo(
            f"Generated all_code.md, all_docs.md, and llm-guides.md in {output_dir}"
        )
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
