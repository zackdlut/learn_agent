from typing import Any, Callable

EXIT_COMMANDS = {"exit", "quit", "bye"}
SYSTEM_PROMPT = "You are a helpful assistant. Before starting any multi-step task, use todo_write to plan your steps.Update status as you go."
BASH_TIMEOUT = 120

DENY_COMMANDS = ["rm -rf /", "reboot", "shutdown", "kill -9", "chown"]
APPROVE_COMMANDS = ["chmod", "sudo", "mv"]
BASH_TOOL = "run_bash"
TODO_WRITE_TOOL = "run_todo_write"
TODO_WRITE_REMINDER_ROUNDS = 3
TODO_WRITE_REMINDER = (
    "Reminder: You have not called run_todo_write for several tool rounds. "
    "Break down the work into tasks and call run_todo_write with id, content, "
    "and status (pending | in_progress | completed)."
)
PATH_TOOLS = {"run_read", "run_write", "run_delete"}

Hook = Callable[..., Any]
