from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from .config import EXIT_COMMANDS, Hook, SYSTEM_PROMPT
from .env import ensure_env_loaded
from .hooks import DEFAULT_HOOKS, create_hooks
from .http_logger import log
from .model import create_model
from .tools import DEFAULT_TOOLS


class Agent:
    def __init__(
        self,
        tools: list[BaseTool] = DEFAULT_TOOLS,
        hooks: list[tuple[str, Hook]] = DEFAULT_HOOKS,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        ensure_env_loaded()
        self.tools_by_name = {item.name: item for item in tools}
        self.model = create_model(tools)
        self.hooks = create_hooks(hooks)
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
        user_input = self.hooks.trigger("UserPromptSubmit", user_input) or user_input
        self.history.append(HumanMessage(content=user_input))

        while True:
            response = self.model.invoke(self.history)
            self.history.append(response)
            if not response.tool_calls:
                break
            for tool_call in response.tool_calls:
                self.history.append(self._handle_tool_call(tool_call))

        print(f"Assistant: {response.content}")
        self.hooks.trigger("Stop", self.history)
        log("--------------run_turn end------------------")

    def _handle_tool_call(self, tool_call) -> ToolMessage:
        blocked = self.hooks.trigger("PreToolCall", tool_call)
        if blocked:
            log(
                f"PreToolCall blocked {tool_call['name']} "
                f"with args {tool_call['args']}: {blocked}"
            )
            return ToolMessage(content=blocked, tool_call_id=tool_call["id"])

        message = self._execute_tool(tool_call)
        print(f"[tool] {tool_call['name']}({tool_call['args']}) -> {message.content}")
        self.hooks.trigger("PostToolCall", tool_call, message)
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
