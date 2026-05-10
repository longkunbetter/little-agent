from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .llm import LLMClient, ModelOutput, ToolCall
from .prompts import SYSTEM_PROMPT
from .tools import ToolRegistry


@dataclass(frozen=True)
class AgentConfig:
    max_loops: int = 20


class StockAgent:
    def __init__(
        self,
        *,
        client: LLMClient,
        tools: ToolRegistry,
        config: AgentConfig,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.client = client
        self.tools = tools
        self.config = config
        self.system_prompt = system_prompt
        self.history: list[dict[str, Any]] = []

    def run_cli(self) -> None:
        print("little-agent stock assistant. Type 'exit' or 'quit' to stop.")
        while True:
            try:
                user_text = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return

            if not user_text:
                continue
            if user_text.lower() in {"exit", "quit"}:
                return

            self.history.append({"role": "user", "content": user_text})
            self.run_task()

    def run_task(self) -> None:
        for _ in range(self.config.max_loops):
            model_output = self.client.create_response(
                system_prompt=self.system_prompt,
                history=self.history,
                tools=self.tools.openai_tools(),
            )
            self.history.extend(model_output.history_items)

            if not model_output.tool_calls:
                if model_output.text:
                    print(model_output.text)
                return

            tool_outputs = self._execute_tool_calls(model_output.tool_calls)
            self.history.extend(tool_outputs)

        print("Reached the maximum loop count before the task completed.")

    def _execute_tool_calls(self, tool_calls: list[ToolCall]) -> list[dict[str, Any]]:
        tool_outputs: list[dict[str, Any]] = []
        for call in tool_calls:
            if call.name == "ask_user":
                result = self._ask_user(call.arguments)
            else:
                result = self.tools.execute(call.name, call.arguments)

            tool_outputs.append(
                {
                    "role": "tool",
                    "tool_call_id": call.call_id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
        return tool_outputs

    @staticmethod
    def _ask_user(arguments: dict[str, Any]) -> dict[str, Any]:
        question = str(arguments.get("question") or "Please provide more detail.")
        answer = input(f"{question}\n> ").strip()
        return {"answer": answer}
