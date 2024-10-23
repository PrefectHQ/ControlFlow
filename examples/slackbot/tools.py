import os
import re
from typing import Literal

import httpx
from langchain_google_community import GoogleSearchAPIWrapper
from raggy.vectorstores.chroma import query_collection
from settings import settings


def convert_md_links_to_slack(text) -> str:
    md_link_pattern = r"\[(?P<text>[^\]]+)]\((?P<url>[^\)]+)\)"

    def to_slack_link(match):
        return f'<{match.group("url")}|{match.group("text")}>'

    return re.sub(
        r"\*\*(.*?)\*\*", r"*\1*", re.sub(md_link_pattern, to_slack_link, text)
    )


async def post_slack_message(
    message: str,
    channel_id: str,
    attachments: list | None = None,
    thread_ts: str | None = None,
    auth_token: str | None = None,
) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://slack.com/api/chat.postMessage",
            headers={
                "Authorization": f"Bearer {auth_token or os.environ["SLACK_API_TOKEN"]}"
            },
            json={
                "channel": channel_id,
                "text": convert_md_links_to_slack(message),
                "attachments": attachments if attachments else [],
                **({"thread_ts": thread_ts} if thread_ts else {}),
            },
        )
    data = response.json()
    if data.get("ok") is not True:
        raise ValueError(f"Error posting Slack message: {data.get('error')}")
    return data


## tools
def search_internet(query: str) -> str:
    """Search the internet for information relevant to the query."""
    return GoogleSearchAPIWrapper(
        google_api_key=settings.google_api_key.get_secret_value(),
        google_cse_id=settings.google_cse_id.get_secret_value(),
    ).run(query)


async def search_knowledge_base(query: str, domain: Literal["docs"]) -> str:
    """Search the knowledge base for information relevant to the query."""
    return await query_collection(
        query_text=query,
        collection_name=domain,
        client_type=settings.chroma_client_type,
    )
