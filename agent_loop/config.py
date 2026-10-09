from typing import Any, Callable

EXIT_COMMANDS = {"exit", "quit", "bye"}
SYSTEM_PROMPT = "You are a helpful assistant."
BASH_TIMEOUT = 120

DENY_COMMANDS = ["rm -rf /", "reboot", "shutdown", "kill -9", "chown"]
APPROVE_COMMANDS = ["chmod", "sudo", "mv"]
BASH_TOOL = "run_bash"
PATH_TOOLS = {"run_read", "run_write", "run_delete"}

Hook = Callable[..., Any]
