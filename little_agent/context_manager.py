from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable


SUMMARY_PREFIX = "Context summary for prior conversation:\n"


@dataclass(frozen=True)
class ContextConfig:
    trigger_chars: int = 12000
    recent_user_chars_budget: int = 3000
    max_compaction_retries: int = 2


class ConversationContext:
    def __init__(self, config: ContextConfig) -> None:
        self.config = config
        self.active_history: list[dict[str, Any]] = []
        self.summary_text: str | None = None
        self._compaction_failures = 0

    def record_items(self, items: Iterable[dict[str, Any]]) -> None:
        self.active_history.extend(items)

    def history_for_model(self) -> list[dict[str, Any]]:
        return list(self.active_history)

    def needs_compaction(self) -> bool:
        if self._compaction_failures >= self.config.max_compaction_retries:
            return False
        return _history_char_count(self.active_history) > self.config.trigger_chars

    def compaction_payload(self) -> str | None:
        real_blocks = _real_user_blocks(self.active_history)
        if len(real_blocks) < 2:
            return None

        recent_blocks = _recent_blocks_with_budget(
            real_blocks,
            char_budget=self.config.recent_user_chars_budget,
        )
        older_blocks = real_blocks[: len(real_blocks) - len(recent_blocks)]
        if not older_blocks and self.summary_text is None:
            return None

        payload_parts: list[str] = []
        if self.summary_text:
            payload_parts.append("Existing compacted summary:\n" + self.summary_text.strip())
        if older_blocks:
            payload_parts.append("Older conversation to compress:\n" + _render_blocks(older_blocks))
        payload_parts.append("Recent conversation that will remain verbatim:\n" + _render_blocks(recent_blocks))
        return "\n\n".join(part for part in payload_parts if part.strip())

    def apply_compaction(self, summary_text: str) -> None:
        summary_text = summary_text.strip()
        if not summary_text:
            raise ValueError("Compaction summary must not be empty")

        real_blocks = _real_user_blocks(self.active_history)
        recent_blocks = _recent_blocks_with_budget(
            real_blocks,
            char_budget=self.config.recent_user_chars_budget,
        )
        self.summary_text = summary_text
        self.active_history = [_summary_item(summary_text), *_flatten_blocks(recent_blocks)]
        self._compaction_failures = 0

    def note_compaction_failure(self) -> None:
        self._compaction_failures += 1


def is_summary_item(item: dict[str, Any]) -> bool:
    return item.get("role") == "user" and str(item.get("content", "")).startswith(SUMMARY_PREFIX)


def _summary_item(summary_text: str) -> dict[str, Any]:
    return {"role": "user", "content": SUMMARY_PREFIX + summary_text.strip()}


def _history_char_count(history: list[dict[str, Any]]) -> int:
    total = 0
    for item in history:
        total += len(_item_to_text(item))
    return total


def _item_to_text(item: dict[str, Any]) -> str:
    content = item.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return json.dumps(content, ensure_ascii=False)
    return json.dumps(item, ensure_ascii=False)


def _real_user_blocks(history: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    blocks: list[list[dict[str, Any]]] = []
    current_block: list[dict[str, Any]] = []
    for item in history:
        if is_summary_item(item):
            continue
        if item.get("role") == "user":
            if current_block:
                blocks.append(current_block)
            current_block = [item]
            continue
        if current_block:
            current_block.append(item)
    if current_block:
        blocks.append(current_block)
    return blocks


def _recent_blocks_with_budget(
    blocks: list[list[dict[str, Any]]],
    *,
    char_budget: int,
) -> list[list[dict[str, Any]]]:
    selected: list[list[dict[str, Any]]] = []
    consumed = 0
    for block in reversed(blocks):
        user_chars = len(_item_to_text(block[0]))
        if selected and consumed + user_chars > char_budget:
            break
        selected.append(block)
        consumed += user_chars
        if consumed >= char_budget:
            break
    selected.reverse()
    return selected


def _render_blocks(blocks: list[list[dict[str, Any]]]) -> str:
    rendered: list[str] = []
    for block in blocks:
        for item in block:
            rendered.append(f"{item.get('role', 'unknown')}: {_item_to_text(item)}")
    return "\n".join(rendered)


def _flatten_blocks(blocks: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for block in blocks:
        flattened.extend(block)
    return flattened
