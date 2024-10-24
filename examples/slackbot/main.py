import asyncio
from typing import Any

from fastapi import FastAPI, Request
from prefect import flow, task

from controlflow import Agent, Memory
from controlflow import run as run_ai

from .moderation import moderate_event
from .settings import settings
from .tools import (
    post_slack_message,
    search_internet,
    search_knowledge_base,
)
from .types import SlackPayload

app = FastAPI()


## agent
agent = Agent(
    name="Marvin (from Hitchhiker's Guide to the Galaxy)",
    instructions="Use tools to assist users with their Prefect inquiries.",
    tools=[search_knowledge_base, search_internet],
)


@task
async def process_slack_event(payload: SlackPayload):
    assert (event := payload.event) is not None and (
        user_id := event.user
    ) is not None, "User not found"

    user_text, channel, thread_ts = moderate_event(event)
    user_memory = Memory(
        key=user_id,
        instructions=f"Store and retrieve information about user {user_id}.",
    )

    response = run_ai(
        user_text,
        instructions="Store relevant context on the user's stack and then query the knowledge base for an answer.",
        agents=[agent],
        memories=[user_memory],
    )

    await post_slack_message(
        message=response,
        channel_id=channel,
        thread_ts=thread_ts,
        auth_token=settings.slack_api_token.get_secret_value(),
    )


## routes
@app.post("/slack/events")
async def handle_events(request: Request):
    payload = SlackPayload(**await request.json())
    if payload.type == "url_verification":
        return {"challenge": payload.challenge}
    elif payload.type == "event_callback":
        asyncio.create_task(process_slack_event(payload))
        return {"message": "Request successfully queued"}
    else:
        return {"message": "Unknown event type"}


# Run with: uvicorn main:app --reload
