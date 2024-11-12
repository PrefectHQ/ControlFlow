import sys
from pathlib import Path

import controlflow as cf


@cf.task()
def write_poem(topic: str) -> str:
    """Write four lines that rhyme"""
    return f"The topic is {topic}"


@cf.flow()
def write_poems(topics: list[str]) -> list[str]:
    return write_poem.map(topics).result()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        write_poems.serve()
    elif len(sys.argv) > 1 and sys.argv[1] == "deploy":
        write_poems.from_source(
            source=str((p := Path(__file__)).parent.resolve()),
            entrypoint=f"{p.name}:something",
        ).deploy(name="some-deployment", work_pool_name="local-process-pool")
    else:
        write_poems(["roses", "violets", "sugar", "spice"])
