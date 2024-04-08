from marvin.beta.assistants import Assistant, Thread
from marvin.beta.assistants.assistants import AssistantTool
from marvin.utilities.asyncio import ExposeSyncMethodsMixin
from pydantic import BaseModel


class ControlFlowModel(BaseModel):
    model_config = dict(validate_assignment=True, extra="forbid")
