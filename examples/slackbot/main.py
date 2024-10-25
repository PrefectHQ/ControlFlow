import asyncio
from typing import Any

from agents import search_knowledgebase_and_refine_context
from custom_types import SlackPayload
from fastapi import FastAPI, Request
from moderation import moderate_event
from prefect import task
from settings import settings
from tools import post_slack_message

from controlflow import Memory
from controlflow import flow as cf_flow

app = FastAPI()


@task
async def process_slack_event(payload: SlackPayload):
    assert (event := payload.event) is not None and (
        slack_user_id := event.user
    ) is not None, "User not found"

    user_text, channel, thread_ts = moderate_event(event)
    user_memory = Memory(
        key=slack_user_id,
        instructions=f"Store and retrieve information about user {slack_user_id}.",
    )

    answer: str = await cf_flow(thread_id=slack_user_id)(
        search_knowledgebase_and_refine_context
    )(
        user_text,
        memories=[user_memory],
    )

    await post_slack_message(
        message=answer,
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", port=8000, reload=True)
