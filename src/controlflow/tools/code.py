# 🚨 WARNING 🚨
# These functions allow ARBITRARY code execution and should be used with caution.

import json
import subprocess


def shell(command: str) -> str:
    """
    Executes a shell command and returns the output

    NOTE: this tool is not sandboxed and can execute arbitrary code. Be careful.
    """
    result = subprocess.run(command, shell=True, text=True, capture_output=True)

    # Output and error
    output = result.stdout
    error = result.stderr

    return json.dumps(dict(command_output=output, command_error=error))


def python(code: str) -> str:
    """
    Executes Python code on the local machine and returns the output.

    NOTE: this tool is not sandboxed and can execute arbitrary code. Be careful.
    """
    return str(eval(code))
