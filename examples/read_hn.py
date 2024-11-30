# /// script
# dependencies = ["controlflow"]
# ///

import os
import sys
from pathlib import Path
from typing import Annotated, TypedDict

import httpx
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
from prefect.docker import DockerImage
from prefect.runner.storage import GitCredentials, GitRepository
from pydantic import AnyHttpUrl, Field, TypeAdapter

import controlflow as cf


class HNArticleSummary(TypedDict):
    link: AnyHttpUrl
    title: str
    main_topics: Annotated[set[str], Field(min_length=1, max_length=5)]
    key_takeaways: Annotated[set[str], Field(min_length=1, max_length=5)]
    tech_domains: Annotated[set[str], Field(min_length=1, max_length=5)]


@cf.task(instructions="concise, main details")
def analyze_article(id: str) -> HNArticleSummary:
    """Analyze a HackerNews article and return structured insights"""
    content = httpx.get(f"https://hacker-news.firebaseio.com/v0/item/{id}.json").json()
    return f"here is the article content: {content}"  # type: ignore


@flow(retries=2)
def analyze_hn_articles(n: int = 5) -> list[HNArticleSummary]:
    top_article_ids = httpx.get(
        "https://hacker-news.firebaseio.com/v0/topstories.json"
    ).json()[:n]
    briefs = analyze_article.map(top_article_ids).result()
    create_markdown_artifact(
        key="hn-article-exec-summary",
        markdown=task(task_run_name=f"make summary of {len(briefs)} articles")(cf.run)(
            objective="markdown summary of all extracted article briefs",
            result_type=Annotated[str, Field(description="markdown summary")],
            context=dict(briefs=briefs),
        ),
        description="executive summary of all extracted article briefs",
    )
    return briefs


if __name__ == "__main__":
    EVERY_12_HOURS_CRON = "0 */12 * * *"
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        analyze_hn_articles.serve(
            parameters={"n": 5},
            cron=EVERY_12_HOURS_CRON,
        )
    elif len(sys.argv) > 1 and sys.argv[1] == "local_deploy":
        analyze_hn_articles.from_source(
            source=str((p := Path(__file__)).parent.resolve()),
            entrypoint=f"{p.name}:analyze_hn_articles",
        ).deploy(
            name="local-deployment",
            work_pool_name="local",
            cron=EVERY_12_HOURS_CRON,
        )
    elif len(sys.argv) > 1 and sys.argv[1] == "docker_deploy":
        repo = GitRepository(
            url="https://github.com/PrefectHQ/controlflow.git",
            branch="main",
            credentials=None,  # replace with `dict(username="", access_token="")` for private repos
        )
        analyze_hn_articles.from_source(
            source=repo,
            entrypoint="examples/read_hn.py:analyze_articles",
        ).deploy(
            name="docker-deployment",
            # image=DockerImage( # uncomment and replace with your own image if desired
            #     name="zzstoatzz/cf-read-hn",
            #     tag="latest",
            #     dockerfile=str(Path(__file__).parent.resolve() / "read-hn.Dockerfile"),
            # ),
            work_pool_name="docker-work",  # uv pip install -U prefect-docker prefect worker start --pool docker-work --type docker
            cron=EVERY_12_HOURS_CRON,
            parameters={"n": 5},
            job_variables={
                "env": {"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")},
                "image": "zzstoatzz/cf-read-hn:latest",  # publicly available image on dockerhub
            },
            build=False,
            push=False,
        )
    else:
        print(f"just running the code\n\n\n\n\n\n")
        briefs = analyze_hn_articles(5)  # type: ignore
        TypeAdapter(list[HNArticleSummary]).validate_python(briefs)
