from typing import Literal

from controlflow.events.events import UnpersistedEvent
from controlflow.orchestration.controller import Controller
from controlflow.utilities.logging import get_logger

logger = get_logger(__name__)


class ControllerStart(UnpersistedEvent):
    event: Literal["controller-start"] = "controller-start"
    persist: bool = False
    controller: Controller


class ControllerEnd(UnpersistedEvent):
    event: Literal["controller-end"] = "controller-end"
    persist: bool = False
    controller: Controller


class ControllerError(UnpersistedEvent):
    event: Literal["controller-error"] = "controller-error"
    persist: bool = False
    controller: Controller
    error: Exception
