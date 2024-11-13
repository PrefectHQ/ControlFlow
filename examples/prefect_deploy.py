import os
import sys
from pathlib import Path

from prefect.docker import DockerImage
from prefect.runner.storage import GitRepository

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
    elif len(sys.argv) > 1 and sys.argv[1] == "local_deploy":
        write_poems.from_source(
            source=str((p := Path(__file__)).parent.resolve()),
            entrypoint=f"{p.name}:write_poem",
        ).deploy(name="local-deployment", work_pool_name="local-process-pool")
    elif len(sys.argv) > 1 and sys.argv[1] == "docker_deploy":
        repo = GitRepository(
            url="https://github.com/PrefectHQ/controlflow.git",
            branch="example-deploy",
        )
        write_poems.from_source(
            source=repo,
            entrypoint="examples/prefect_deploy.py:write_poems",
        ).deploy(
            name="docker-deployment",
            image=DockerImage(
                name="zzstoatzz/cf-test-deploy",
                tag="latest",
                dockerfile=str(
                    Path(__file__).parent.resolve() / "prefect-deploy.Dockerfile"
                ),
            ),
            work_pool_name="docker-work",
            parameters={"topics": ["roses", "violets", "sugar", "spice"]},
            job_variables={"env": {"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")}},
        )
    else:
        print(f"just running the code\n\n\n\n\n\n")
        write_poems(["roses", "violets", "sugar", "spice"])
