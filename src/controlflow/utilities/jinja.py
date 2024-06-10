import inspect
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from jinja2 import Environment as JinjaEnvironment
from jinja2 import StrictUndefined, select_autoescape

jinja_env = JinjaEnvironment(
    autoescape=select_autoescape(default_for_string=False),
    trim_blocks=True,
    lstrip_blocks=True,
    auto_reload=True,
    undefined=StrictUndefined,
)

jinja_env.globals.update(
    {
        "now": lambda: datetime.now(ZoneInfo("UTC")),
        "inspect": inspect,
        "getcwd": os.getcwd,
        "zip": zip,
    }
)
