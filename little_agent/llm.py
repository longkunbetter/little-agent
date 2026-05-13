from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ModelOutput:
    text: str
    reasoning_text: str
    tool_calls: list[ToolCall]
    history_items: list[dict[str, Any]]


class LLMClient(Protocol):
    def create_response(
        self,
        *,
        system_prompt: str,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelOutput:
        ...

    def create_compaction_summary(
        self,
        *,
        system_prompt: str,
        payload: str,
    ) -> str:
        ...


class DeferredLLMClient:
    def __init__(self, factory):
        self._factory = factory
        self._client: LLMClient | None = None

    def create_response(
        self,
        *,
        system_prompt: str,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelOutput:
        client = self._get_client()
        return client.create_response(system_prompt=system_prompt, history=history, tools=tools)

    def create_compaction_summary(
        self,
        *,
        system_prompt: str,
        payload: str,
    ) -> str:
        client = self._get_client()
        return client.create_compaction_summary(system_prompt=system_prompt, payload=payload)

    def _get_client(self) -> LLMClient:
        if self._client is None:
            self._client = self._factory()
        return self._client


class OpenAIChatCompletionsClient:
    def __init__(self, *, model: str, base_url: str, client: Any | None = None) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=_api_key_from_env(), base_url=base_url)
        self.model = model
        self.client = client

    @classmethod
    def from_env(cls) -> "OpenAIChatCompletionsClient":
        if not _api_key_from_env():
            raise RuntimeError(
                "Missing LLM credentials. Set DEEPSEEK_API_KEY or OPENAI_API_KEY before running little-agent."
            )
        return cls(
            model=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
            base_url=os.environ.get("DEEPSEEK_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or DEFAULT_BASE_URL,
        )

    @classmethod
    def deferred_from_env(cls) -> DeferredLLMClient:
        return DeferredLLMClient(cls.from_env)

    def create_response(
        self,
        *,
        system_prompt: str,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelOutput:
        messages = _sanitize_for_json([{"role": "system", "content": system_prompt}, *history])
        safe_tools = _sanitize_for_json(tools)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=safe_tools,
        )

        message = response.choices[0].message
        message_dump = message.model_dump(exclude_none=True)
        history_items: list[dict[str, Any]] = []
        tool_calls: list[ToolCall] = []
        text = message.content or ""
        reasoning_text = _extract_reasoning_text(message_dump)

        history_items.append(_assistant_message_for_history(message_dump))
        for tool_call in message.tool_calls or []:
            function = tool_call.function
            tool_calls.append(
                _parse_tool_call(
                    call_id=tool_call.id,
                    name=function.name,
                    raw_arguments=function.arguments,
                )
            )

        return ModelOutput(
            text=text.strip(),
            reasoning_text=reasoning_text.strip(),
            tool_calls=tool_calls,
            history_items=history_items,
        )

    def create_compaction_summary(
        self,
        *,
        system_prompt: str,
        payload: str,
    ) -> str:
        messages = _sanitize_for_json([{"role": "user", "content": payload}])
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt}, *messages],
        )
        message = response.choices[0].message
        return (message.content or "").strip()


def _api_key_from_env() -> str | None:
    return os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, str):
        return "".join(
            "\ufffd" if 0xD800 <= ord(character) <= 0xDFFF else character
            for character in value
        )
    if isinstance(value, list):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, dict):
        return {
            _sanitize_for_json(key): _sanitize_for_json(item)
            for key, item in value.items()
        }
    return value


def _assistant_message_for_history(message: dict[str, Any]) -> dict[str, Any]:
    history_message: dict[str, Any] = {"role": "assistant"}
    for key in ("content", "tool_calls", "reasoning_content"):
        if key in message:
            history_message[key] = message[key]
    return history_message


def _extract_reasoning_text(message: dict[str, Any]) -> str:
    value = message.get("reasoning_content")
    if isinstance(value, str):
        return value
    return ""


def _parse_tool_call(*, call_id: str, name: str, raw_arguments: str) -> ToolCall:
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError:
        arguments = {"_raw_arguments": raw_arguments}

    return ToolCall(
        call_id=call_id,
        name=name,
        arguments=arguments,
    )
