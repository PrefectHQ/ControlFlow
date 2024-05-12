import inspect
from datetime import datetime
from zoneinfo import ZoneInfo

from marvin.utilities.jinja import BaseEnvironment

jinja_env = BaseEnvironment(
    globals={
        "now": lambda: datetime.now(ZoneInfo("UTC")),
        "inspect": inspect,
        "id": id,
    }
)
