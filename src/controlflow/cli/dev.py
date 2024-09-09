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
