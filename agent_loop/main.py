#!/usr/bin/env python3

import glob
import os
import subprocess
from datetime import datetime
from typing import Annotated, Any, Callable

from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from pydantic import Field

from http_logger import clear_log, log, make_async_client, make_sync_client

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EXIT_COMMANDS = {"exit", "quit", "bye"}
SYSTEM_PROMPT = "You are a helpful assistant."
BASH_TIMEOUT = 120

DENY_COMMANDS = ["rm -rf /", "reboot", "shutdown", "kill -9", "chown"]
APPROVE_COMMANDS = ["chmod", "sudo", "mv"]
BASH_TOOL = "run_bash"
PATH_TOOLS = {"run_read", "run_write", "run_delete"}

Hook = Callable[..., Any]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def create_model(tools: list[BaseTool]):
    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        http_client=make_sync_client(),
        http_async_client=make_async_client(),
    )
    return model.bind_tools(tools)


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

class HookBus:
    """Named hook lists. The first handler that returns a value stops the chain."""

    EVENTS = ("UserPromptSubmit", "PreToolCall", "PostToolCall", "Stop")

    def __init__(self):
        self._handlers: dict[str, list[Hook]] = {event: [] for event in self.EVENTS}

    def on(self, event: str, handler: Hook) -> Hook:
        if event not in self._handlers:
            raise ValueError(f"Unknown hook: {event}")
        self._handlers[event].append(handler)
        return handler

    def emit(self, event: str, *args, **kwargs):
        for handler in self._handlers[event]:
            log(f"Triggering hook: {event} with args {args} and kwargs {kwargs}")
            result = handler(*args, **kwargs)
            log(f"Hook result: {result}")
            if result is not None:
                return result
        return None


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
    log(f"Checking permission for {name} with args {args}")

    if name == BASH_TOOL:
        return _guard_bash(name, args)
    if name in PATH_TOOLS:
        return _guard_path(args.get("path", ""))
    return None


def context_inject_hook(user_input: str) -> str:
    log(f"Injecting context into user_input: {user_input}")
    return "you should always return response in chinese " + user_input


def stop_hook(history) -> None:
    log(f"Stopping turn: {len(history)}")
    summary = "You have had " + str(len(history)) + " interactions with the LLM."
    print(f"Summary: {summary}")


DEFAULT_HOOKS: list[tuple[str, Hook]] = [
    ("UserPromptSubmit", context_inject_hook),
    ("PreToolCall", permission_hook),
    ("Stop", stop_hook),
]


def create_hook_bus(hooks: list[tuple[str, Hook]]) -> HookBus:
    bus = HookBus()
    for event, handler in hooks:
        bus.on(event, handler)
    return bus


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    def __init__(
        self,
        tools: list[BaseTool] = DEFAULT_TOOLS,
        hooks: list[tuple[str, Hook]] = DEFAULT_HOOKS,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        self.tools_by_name = {item.name: item for item in tools}
        self.model = create_model(tools)
        self.hooks = create_hook_bus(hooks)
        self.history: list[BaseMessage] = [SystemMessage(content=system_prompt)]

    def run(self) -> None:
        print("Agent Loop is starting...")
        log("Agent Loop is starting...")
        while True:
            user_input = input("You: ")
            log(f"User: {user_input}")
            if user_input.strip().lower() in EXIT_COMMANDS:
                break
            self.run_turn(user_input)
        log("Agent Loop is exiting...")

    def run_turn(self, user_input: str) -> None:
        log("--------------run_turn start------------------")
        user_input = self.hooks.emit("UserPromptSubmit", user_input) or user_input
        self.history.append(HumanMessage(content=user_input))

        while True:
            response = self.model.invoke(self.history)
            self.history.append(response)
            if not response.tool_calls:
                break
            for tool_call in response.tool_calls:
                self.history.append(self._handle_tool_call(tool_call))

        print(f"Assistant: {response.content}")
        self.hooks.emit("Stop", self.history)
        log("--------------run_turn end------------------")

    def _handle_tool_call(self, tool_call) -> ToolMessage:
        blocked = self.hooks.emit("PreToolCall", tool_call)
        if blocked:
            log(
                f"PreToolCall blocked {tool_call['name']} "
                f"with args {tool_call['args']}: {blocked}"
            )
            return ToolMessage(content=blocked, tool_call_id=tool_call["id"])

        message = self._execute_tool(tool_call)
        print(f"[tool] {tool_call['name']}({tool_call['args']}) -> {message.content}")
        self.hooks.emit("PostToolCall", tool_call, message)
        return message

    def _execute_tool(self, tool_call) -> ToolMessage:
        selected = self.tools_by_name.get(tool_call["name"])
        if selected is None:
            content = f"Unknown tool: {tool_call['name']}"
        else:
            try:
                return selected.invoke(tool_call)
            except Exception as exc:
                content = f"Error: {exc}"
        return ToolMessage(content=content, tool_call_id=tool_call["id"])


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    clear_log()
    agent = Agent()
    agent.run()


if __name__ == "__main__":
    main()
