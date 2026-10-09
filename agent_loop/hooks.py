import os

from .config import (
    APPROVE_COMMANDS,
    BASH_TOOL,
    DENY_COMMANDS,
    Hook,
    PATH_TOOLS,
)
from .http_logger import log


class HookManager:
    """Named hook lists. The first handler that returns a value stops the chain."""

    EVENTS = ("UserPromptSubmit", "PreToolCall", "PostToolCall", "Stop")

    def __init__(self):
        self._handlers: dict[str, list[Hook]] = {event: [] for event in self.EVENTS}

    def register(self, event: str, handler: Hook) -> Hook:
        if event not in self._handlers:
            raise ValueError(f"Unknown hook: {event}")
        self._handlers[event].append(handler)
        return handler

    def trigger(self, event: str, *args, **kwargs):
        for handler in self._handlers[event]:
            result = handler(*args, **kwargs)
            _log_hook_outcome(event, handler, args, result)
            if result is not None:
                return result
        return None


def _log_hook_outcome(event: str, handler: Hook, args: tuple, result) -> None:
    name = getattr(handler, "__name__", handler.__class__.__name__)
    if event == "UserPromptSubmit" and result is not None:
        log(f"hook {name}: user prompt augmented")
        return
    if event == "PreToolCall" and args:
        tool_call = args[0]
        label = _brief_tool_call(tool_call)
        if result is not None:
            log(f"hook {name}: blocked {label} — {result}")
        return
    if event == "Stop":
        if args:
            log(f"hook {name}: turn finished ({len(args[0])} messages in history)")
        return
    if result is not None:
        log(f"hook {name} ({event}): {result}")


def _brief_tool_call(tool_call) -> str:
    name = tool_call.get("name", "?")
    args = tool_call.get("args") or {}
    if not args:
        return name
    parts = []
    for key, value in args.items():
        text = repr(value)
        if len(text) > 60:
            text = f"{text[:57]}…"
        parts.append(f"{key}={text}")
    return f"{name}({', '.join(parts)})"


def _guard_bash(name: str, args: dict) -> str | None:
    command = args.get("command", "")
    if any(denied in command for denied in DENY_COMMANDS):
        log(f"Denied command: {command}")
        return f"Denied command: {command}"
    if any(flagged in command for flagged in APPROVE_COMMANDS):
        log(f"Checking approval for command: {command}")
        if input("Are you sure you want to proceed? (y/n): ").lower() != "y":
            reason = f"Not approved  by user for {name} with args {args}"
            log(reason)
            return reason
    return None


def _guard_path(path: str) -> str | None:
    if os.access(path, os.R_OK | os.W_OK):
        return None
    log(f"Permission denied for {path}")
    return f"Permission denied for {path}"


def permission_hook(tool_call) -> str | None:
    """Block a tool call. Return a reason string, or None to allow it."""
    name = tool_call["name"]
    args = tool_call["args"]
    if name == BASH_TOOL:
        return _guard_bash(name, args)
    if name in PATH_TOOLS:
        return _guard_path(args.get("path", ""))
    return None


def context_inject_hook(user_input: str) -> str:
    return "you should always return response in chinese " + user_input


def stop_hook(history) -> None:
    summary = "You have had " + str(len(history)) + " interactions with the LLM."
    print(f"Summary: {summary}")


DEFAULT_HOOKS: list[tuple[str, Hook]] = [
    ("UserPromptSubmit", context_inject_hook),
    ("PreToolCall", permission_hook),
    ("Stop", stop_hook),
]


def create_hooks(hooks: list[tuple[str, Hook]]) -> HookManager:
    manager = HookManager()
    for event, handler in hooks:
        manager.register(event, handler)
    return manager
