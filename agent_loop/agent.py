from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from .config import (
    EXIT_COMMANDS,
    Hook,
    SYSTEM_PROMPT,
    TODO_WRITE_REMINDER,
    TODO_WRITE_REMINDER_ROUNDS,
    LOAD_SKILLS_TOOL,
    TODO_WRITE_TOOL,
)
from .env import ensure_env_loaded
from .hooks import DEFAULT_HOOKS, create_hooks
from .http_logger import log
from .model import create_model
from .skills import build_system_prompt
from .tools import DEFAULT_TOOLS


class Agent:
    def __init__(
        self,
        tools: list[BaseTool] = DEFAULT_TOOLS,
        hooks: list[tuple[str, Hook]] = DEFAULT_HOOKS,
        system_prompt: str | None = None,
    ):
        ensure_env_loaded()
        if system_prompt is None:
            system_prompt = build_system_prompt(SYSTEM_PROMPT)
        self.tools_by_name = {item.name: item for item in tools}
        self.model = create_model(tools)
        self.hooks = create_hooks(hooks)
        self.history: list[BaseMessage] = [SystemMessage(content=system_prompt)]
        self._tool_rounds_without_todo_write = 0

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
        user_input = self.hooks.trigger("UserPromptSubmit", user_input) or user_input
        self.history.append(HumanMessage(content=user_input))
        self._tool_rounds_without_todo_write = 0

        while True:
            response = self.model.invoke(self.history)
            self.history.append(response)
            if not response.tool_calls:
                break
            round_messages: list[ToolMessage] = []
            for tool_call in response.tool_calls:
                message = self._handle_tool_call(tool_call)
                round_messages.append(message)
                self.history.append(message)
            self._after_tool_round(response.tool_calls, round_messages)

        print(f"Assistant: {response.content}")
        self.hooks.trigger("Stop", self.history)
        log("--------------run_turn end------------------")

    def _after_tool_round(self, tool_calls, round_messages: list[ToolMessage]) -> None:
        used_todo_write = any(call["name"] == TODO_WRITE_TOOL for call in tool_calls)
        if used_todo_write:
            self._tool_rounds_without_todo_write = 0
            return

        self._tool_rounds_without_todo_write += 1
        if self._tool_rounds_without_todo_write < TODO_WRITE_REMINDER_ROUNDS:
            return

        last = round_messages[-1]
        reminded = ToolMessage(
            content=f"{last.content}\n\n{TODO_WRITE_REMINDER}",
            tool_call_id=last.tool_call_id,
        )
        round_messages[-1] = reminded
        self.history[-1] = reminded
        self._tool_rounds_without_todo_write = 0
        log(f"Appended {TODO_WRITE_TOOL} reminder after {TODO_WRITE_REMINDER_ROUNDS} rounds")

    def _handle_tool_call(self, tool_call) -> ToolMessage:
        blocked = self.hooks.trigger("PreToolCall", tool_call)
        if blocked:
            log(
                f"PreToolCall blocked {tool_call['name']} "
                f"with args {tool_call['args']}: {blocked}"
            )
            return ToolMessage(content=blocked, tool_call_id=tool_call["id"])

        message = self._execute_tool(tool_call)
        print(
            f"[tool] {tool_call['name']}({tool_call['args']}) -> "
            f"{self._tool_content_for_terminal(tool_call['name'], message.content)}"
        )
        self.hooks.trigger("PostToolCall", tool_call, message)
        return message

    @staticmethod
    def _tool_content_for_terminal(tool_name: str, content: str) -> str:
        """Full tool output still goes to the model; terminal log stays readable."""
        if tool_name == LOAD_SKILLS_TOOL:
            n = len(content)
            return f"<skill body loaded, {n} chars — not shown in terminal>"
        if len(content) > 400:
            return f"{content[:400]}… ({len(content)} chars total)"
        return content

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
