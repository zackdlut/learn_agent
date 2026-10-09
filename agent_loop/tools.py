import glob
import os
import json
import subprocess
from datetime import datetime
from typing import Annotated, Literal

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from .config import BASH_TIMEOUT
from .skills import get_skill_loader


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
TodoStatus = Literal["pending", "in_progress", "completed"]


class TodoTask(BaseModel):
    id: str = Field(description="Unique task identifier")
    content: str = Field(description="Task description")
    status: TodoStatus = Field(
        description="Task state: pending, in_progress, or completed"
    )


class TodoStore:
    """Session-scoped task list (one store per process)."""

    def __init__(self) -> None:
        self._items: dict[str, TodoTask] = {}

    def upsert_many(self, tasks: list[TodoTask]) -> list[TodoTask]:
        for task in tasks:
            self._items[task.id] = task
        return self.list_tasks()

    def list_tasks(self) -> list[TodoTask]:
        return sorted(self._items.values(), key=lambda t: t.id)


_todo_store = TodoStore()


def _parse_todo_payload(
    todos: list[TodoTask] | list[dict] | str,
) -> tuple[list[TodoTask] | None, str | None]:
    if isinstance(todos, str):
        try:
            todos = json.loads(todos)
        except json.JSONDecodeError as exc:
            return None, f"Invalid JSON: {exc}"
    if not isinstance(todos, list):
        return None, "todos must be a list of tasks"

    parsed: list[TodoTask] = []
    for index, item in enumerate(todos):
        try:
            if isinstance(item, TodoTask):
                parsed.append(item)
            else:
                parsed.append(TodoTask.model_validate(item))
        except Exception as exc:
            return None, f"todo[{index}]: {exc}"
    return parsed, None


@tool
def run_todo_write(
    todos: Annotated[
        list[TodoTask] | str,
        Field(
            description=(
                "Tasks to create or update. Each task has id, content, and status "
                "(pending | in_progress | completed). Same id updates the existing task."
            ),
        ),
    ],
) -> str:
    """Record or update the task breakdown for the current session. Returns the full task list."""
    tasks, error = _parse_todo_payload(todos)
    if error:
        return error
    all_tasks = _todo_store.upsert_many(tasks)
    payload = [task.model_dump() for task in all_tasks]
    return json.dumps(payload, ensure_ascii=False, indent=2)


@tool
def load_skills(
    names: Annotated[
        list[str],
        Field(description="Skill names from the system prompt catalog"),
    ],
) -> str:
    """Load full skill instructions by name. Use when a task matches a listed skill."""
    return get_skill_loader().load(names)


DEFAULT_TOOLS: list[BaseTool] = [
    get_current_time,
    run_bash,
    run_read,
    run_write,
    run_glob,
    run_delete,
    run_todo_write,
    load_skills,
]
