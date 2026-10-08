#!/usr/bin/env python3

import os
import subprocess
from datetime import datetime
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import Field

from http_logger import make_async_client, make_sync_client

load_dotenv(override=True)

EXIT_COMMANDS = {"exit", "quit", "bye"}
SYSTEM_PROMPT = "You are a helpful assistant."


@tool
def get_current_time() -> str:
    """Get the current local time. Returns a string like 2026-10-08 15:18:00."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def run_bash(
    command: Annotated[str, Field(description="Shell command to execute")],
) -> str:
    """Execute a command in the terminal. Returns stdout and stderr, or an error string."""
    dangerous_commands = ["rm -rf /", "reboot", "shutdown", "sudo"]
    if any(cmd in command for cmd in dangerous_commands):
        return "Error: This command is dangerous and not allowed."
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
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
    if limit is not None:
        return content[:limit]
    return content


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
    import glob as g
    try:
        matches = sorted(match for match in g.glob(pattern) if g.path.isfile(match))
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


def init_tools():
    return [get_current_time, run_bash, run_read, run_write, run_glob, run_delete]


def init_model(tools):
    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        http_client=make_sync_client(),
        http_async_client=make_async_client(),
    )
    return model.bind_tools(tools)


def execute_tool_call(tools_by_name, tool_call):
    tool = tools_by_name.get(tool_call["name"])
    if tool is None:
        return ToolMessage(
            content=f"Unknown tool: {tool_call['name']}",
            tool_call_id=tool_call["id"],
        )
    try:
        return tool.invoke(tool_call)
    except Exception as exc:
        return ToolMessage(
            content=f"Error: {exc}",
            tool_call_id=tool_call["id"],
        )


def run_turn(model, tools_by_name, history, user_input):
    history.append(HumanMessage(content=user_input))
    while True:
        response = model.invoke(history)
        history.append(response)
        if not response.tool_calls:
            print(f"Assistant: {response.content}")
            return
        for tool_call in response.tool_calls:
            message = execute_tool_call(tools_by_name, tool_call)
            print(f"[tool] {tool_call['name']}({tool_call['args']}) -> {message.content}")
            history.append(message)


def main():
    print("Agent Loop is starting...")
    tools = init_tools()
    tools_by_name = {tool.name: tool for tool in tools}
    model = init_model(tools)
    history = [SystemMessage(content=SYSTEM_PROMPT)]
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in EXIT_COMMANDS:
            break
        run_turn(model, tools_by_name, history, user_input)


if __name__ == "__main__":
    main()
