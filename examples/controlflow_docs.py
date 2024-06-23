from pathlib import Path

import controlflow as cf
from controlflow.tools import tool
from langchain_openai import OpenAIEmbeddings

try:
    from langchain_community.document_loaders import DirectoryLoader
    from langchain_community.vectorstores import LanceDB
    from langchain_text_splitters import (
        MarkdownTextSplitter,
        PythonCodeTextSplitter,
    )
except ImportError:
    raise ImportError(
        "Missing requirements: `pip install lancedb langchain-community langchain-text-splitters unstructured`"
    )


def create_code_db():
    # .py files
    py_loader = DirectoryLoader(
        Path(cf.__file__).parents[2] / "src/controlflow/", glob="**/*.py"
    )
    py_raw_documents = py_loader.load()
    py_splitter = PythonCodeTextSplitter(chunk_size=1400, chunk_overlap=200)
    documents = py_splitter.split_documents(py_raw_documents)
    return LanceDB.from_documents(documents, OpenAIEmbeddings())


def create_docs_db():
    # .mdx files
    mdx_loader = DirectoryLoader(Path(cf.__file__).parents[2] / "docs", glob="**/*.mdx")
    mdx_raw_documents = mdx_loader.load()
    mdx_splitter = MarkdownTextSplitter(chunk_size=1400, chunk_overlap=200)
    documents = mdx_splitter.split_documents(mdx_raw_documents)
    return LanceDB.from_documents(documents, OpenAIEmbeddings())


code_db = create_code_db()
docs_db = create_docs_db()


@tool
def search_code(query: str, n=50) -> list[dict]:
    """
    Semantic search over the current ControlFlow documentation

    Returns the top `n` results.
    """
    results = docs_db.similarity_search(query, k=n)
    return [
        dict(content=r.page_content, metadata=r.metadata["metadata"]) for r in results
    ]


@tool
def search_docs(query: str, n=50) -> list[dict]:
    """
    Semantic search over the current ControlFlow documentation

    Returns the top `n` results.
    """
    results = docs_db.similarity_search(query, k=n)
    return [
        dict(content=r.page_content, metadata=r.metadata["metadata"]) for r in results
    ]


@tool
def read_file(path: str) -> str:
    """
    Read a file from a path.
    """
    with open(path) as f:
        return f.read()


agent = cf.Agent(
    "DocsAgent",
    description="The agent for the ControlFlow documentation",
    instructions="Use your tools to explore the ControlFlow code and documentation. If you find something interesting but only see a snippet with the search tools, use the read_file tool to get the full text.",
    tools=[search_code, search_docs, read_file],
)


@cf.flow
def write_docs(topic: str):
    task = cf.Task(
        "Research the provided topic, then produce world-class documentation in the style of the existing docs.",
        context=dict(topic=topic),
        agents=[agent],
    )
    task.generate_subtasks()
    return task
