import glob
import os
import subprocess
from datetime import datetime
from typing import Annotated

from langchain_core.tools import BaseTool, tool
from pydantic import Field

from .config import BASH_TIMEOUT


@tool
def get_current_time() -> str:
    """Get the current local time. Returns a string like 2026-10-08 15:18:00."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def run_bash(
    command: Annotated[str, Field(description="Shell command to execute")],
) -> str:
    """Execute a command in the terminal. Returns stdout and stderr, or an error string."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=BASH_TIMEOUT
        )
        return (result.stdout + result.stderr).strip()
    except subprocess.TimeoutExpired:
        return "Error: Command timed out."
    except Exception as e:
        return f"Error: {e}"


@tool
def run_read(
    path: Annotated[str, Field(description="Path of the file to read")],
    limit: Annotated[
        int | None,
        Field(description="Maximum number of characters to return. Omit to read the whole file."),
    ] = None,
) -> str:
    """Read the content of a file. Returns the file text, truncated when limit is set."""
    with open(path, "r") as file:
        content = file.read()
    return content if limit is None else content[:limit]


@tool
def run_write(
    path: Annotated[str, Field(description="Path of the file to write")],
    content: Annotated[str, Field(description="Full text to write into the file")],
) -> str:
    """Write content to a file, replacing any existing text. Returns a success or error string."""
    with open(path, "w") as file:
        file.write(content)
    return "File written successfully."


@tool
def run_glob(
    pattern: Annotated[str, Field(description="Glob pattern, such as *.py or agent_loop/**/*.py")],
) -> str:
    """List files matching a pattern. Returns matching file paths, one per line."""
    try:
        matches = sorted(
            match for match in glob.glob(pattern, recursive=True) if os.path.isfile(match)
        )
        return "\n".join(matches)
    except Exception as e:
        return f"Error: {e}"


@tool
def run_delete(
    path: Annotated[str, Field(description="Path of the file to delete")],
) -> str:
    """Delete a file. Returns a success or error string."""
    try:
        os.remove(path)
        return "File deleted successfully."
    except Exception as e:
        return f"Error: {e}"


DEFAULT_TOOLS: list[BaseTool] = [
    get_current_time,
    run_bash,
    run_read,
    run_write,
    run_glob,
    run_delete,
]
