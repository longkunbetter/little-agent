from __future__ import annotations

import json
import os
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from typing import Any

from .context_manager import ContextConfig, ConversationContext
from .llm import LLMClient, ModelOutput, ToolCall
from .prompts import COMPACTION_PROMPT, SYSTEM_PROMPT
from .tools import ToolRegistry


@dataclass(frozen=True)
class AgentConfig:
    max_loops: int = 20
    show_progress: bool = True
    show_model_reasoning: bool = True
    context_compaction_enabled: bool = True
    context_trigger_chars: int = 12000
    compact_recent_user_chars_budget: int = 3000
    max_compaction_retries: int = 2


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
        self.context = ConversationContext(
            ContextConfig(
                trigger_chars=config.context_trigger_chars,
                recent_user_chars_budget=config.compact_recent_user_chars_budget,
                max_compaction_retries=config.max_compaction_retries,
            )
        )

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

            self.context.record_items([{"role": "user", "content": user_text}])
            self.run_task()

    def run_task(self) -> None:
        task_started_at = time.perf_counter()
        for loop_index in range(1, self.config.max_loops + 1):
            self._maybe_compact_context()
            llm_started_at = time.perf_counter()
            self._print_progress("Thinking... asking the model how to approach this request.")
            model_output = self.client.create_response(
                system_prompt=self.system_prompt,
                history=self.context.history_for_model(),
                tools=self.tools.openai_tools(),
            )
            llm_elapsed = time.perf_counter() - llm_started_at
            self.context.record_items(model_output.history_items)
            self._report_model_output(loop_index, model_output, llm_elapsed)

            if not model_output.tool_calls:
                if model_output.text:
                    print(model_output.text)
                self._print_progress(
                    f"Done in {time.perf_counter() - task_started_at:.1f}s after {loop_index} loop(s)."
                )
                return

            tool_outputs = self._execute_tool_calls(model_output.tool_calls)
            self.context.record_items(tool_outputs)

        self._print_progress(
            f"Reached the maximum loop count before the task completed ({time.perf_counter() - task_started_at:.1f}s)."
        )

    def _execute_tool_calls(self, tool_calls: list[ToolCall]) -> list[dict[str, Any]]:
        tool_outputs: list[dict[str, Any]] = []
        for call in tool_calls:
            tool_started_at = time.perf_counter()
            if call.name == "ask_user":
                self._print_progress("Model needs one clarification before it can continue.")
                result = self._ask_user(call.arguments)
            else:
                self._print_progress(
                    f"Running {call.name}: {self._summarize_tool_arguments(call.arguments)}."
                )
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    result = self.tools.execute(call.name, call.arguments)
                self._report_tool_result(call.name, result, time.perf_counter() - tool_started_at)

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

    def _maybe_compact_context(self) -> None:
        if not self.config.context_compaction_enabled:
            return
        if not self.context.needs_compaction():
            return

        payload = self.context.compaction_payload()
        if not payload:
            return

        self._print_progress("Compacting context to keep the conversation within budget.")
        try:
            summary = self.client.create_compaction_summary(
                system_prompt=COMPACTION_PROMPT,
                payload=payload,
            )
            self.context.apply_compaction(summary)
            self._print_progress("Context compaction finished.")
        except Exception as exc:  # noqa: BLE001 - compaction must not break the main task
            self.context.note_compaction_failure()
            self._print_progress(f"Context compaction skipped due to error: {exc}")

    def _report_model_output(self, loop_index: int, model_output: ModelOutput, elapsed: float) -> None:
        if model_output.tool_calls:
            suffix = "tool" if len(model_output.tool_calls) == 1 else "tools"
            self._print_progress(
                f"Model responded in {elapsed:.1f}s and wants to use {len(model_output.tool_calls)} {suffix}."
            )
        else:
            self._print_progress(f"Model responded in {elapsed:.1f}s.")

        reasoning = self._format_reasoning(model_output)
        if reasoning:
            print(reasoning)

        if model_output.text and model_output.tool_calls:
            print(model_output.text)

    def _report_tool_result(self, tool_name: str, result: dict[str, Any], elapsed: float) -> None:
        status = str(result.get("status") or "ok")
        result_count = len(result.get("results", [])) if isinstance(result.get("results"), list) else None
        warning_count = len(result.get("warnings", [])) if isinstance(result.get("warnings"), list) else 0
        summary = f"{tool_name} finished in {elapsed:.1f}s with status={status}"
        if result_count is not None:
            summary += f", results={result_count}"
        if warning_count:
            summary += f", warnings={warning_count}"
        self._print_progress(summary + ".")

    def _format_reasoning(self, model_output: ModelOutput) -> str:
        if not self.config.show_model_reasoning:
            return ""
        if _show_raw_reasoning() and model_output.reasoning_text:
            return f"[thinking] {model_output.reasoning_text}"
        if model_output.tool_calls:
            summaries = [
                f"{call.name}({self._summarize_tool_arguments(call.arguments, compact=True)})"
                for call in model_output.tool_calls
            ]
            return f"[thinking] Planning next action: {', '.join(summaries)}"
        if model_output.reasoning_text:
            first_line = model_output.reasoning_text.strip().splitlines()[0]
            return f"[thinking] {first_line}"
        return ""

    def _summarize_tool_arguments(self, arguments: dict[str, Any], compact: bool = False) -> str:
        pieces: list[str] = []
        market = arguments.get("market")
        if market:
            pieces.append(f"market={market}")

        criteria = arguments.get("criteria")
        if isinstance(criteria, dict):
            filters = criteria.get("filters")
            if isinstance(filters, list) and filters:
                formatted_filters = []
                for item in filters[:3]:
                    if isinstance(item, dict):
                        field = item.get("field")
                        op = item.get("op")
                        value = item.get("value")
                        if field and op is not None and value is not None:
                            formatted_filters.append(f"{field} {op} {value}")
                if formatted_filters:
                    pieces.append("filters=" + "; ".join(formatted_filters))

            exclude = criteria.get("exclude")
            if isinstance(exclude, list) and exclude:
                pieces.append("exclude=" + ",".join(str(item) for item in exclude[:3]))

            sort_by = criteria.get("sort_by")
            if sort_by:
                pieces.append(f"sort_by={sort_by}")

        limit = arguments.get("limit")
        if limit:
            pieces.append(f"limit={limit}")

        if not pieces:
            for key, value in list(arguments.items())[:3]:
                pieces.append(f"{key}={value}")

        separator = ", " if compact else " "
        return separator.join(pieces) if pieces else "no arguments"

    def _print_progress(self, message: str) -> None:
        if self.config.show_progress:
            print(f"[status] {message}")


def _show_raw_reasoning() -> bool:
    return os.environ.get("LITTLE_AGENT_SHOW_RAW_REASONING") == "1"
