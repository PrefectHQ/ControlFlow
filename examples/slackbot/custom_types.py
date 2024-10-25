from pydantic import BaseModel


class EventBlockElement(BaseModel):
    type: str
    text: str | None = None
    user_id: str | None = None


class EventBlockElementGroup(BaseModel):
    type: str
    elements: list[EventBlockElement]


class EventBlock(BaseModel):
    type: str
    block_id: str
    elements: list[EventBlockElement | EventBlockElementGroup]


class SlackEvent(BaseModel):
    client_msg_id: str | None = None
    type: str
    text: str | None = None
    user: str | None = None
    ts: str | None = None
    team: str | None = None
    channel: str | None = None
    event_ts: str
    thread_ts: str | None = None
    parent_user_id: str | None = None
    blocks: list[EventBlock] | None = None


class EventAuthorization(BaseModel):
    enterprise_id: str | None = None
    team_id: str
    user_id: str
    is_bot: bool
    is_enterprise_install: bool


class SlackPayload(BaseModel):
    token: str
    type: str
    team_id: str | None = None
    api_app_id: str | None = None
    event: SlackEvent | None = None
    event_id: str | None = None
    event_time: int | None = None
    authorizations: list[EventAuthorization] | None = None
    is_ext_shared_channel: bool | None = None
    event_context: str | None = None
    challenge: str | None = None
