from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def as_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, tools: list[ToolDefinition]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def openai_tools(self) -> list[dict[str, Any]]:
        return [tool.as_openai_tool() for tool in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            return {
                "status": "error",
                "error": f"Unknown tool: {name}",
                "arguments": arguments,
            }
        return tool.handler(arguments)


def build_default_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            _ask_user_tool(),
            _stock_screener_tool(),
            _news_sentiment_tool(),
            _technical_analysis_tool(),
            _financial_statement_tool(),
        ]
    )


def _ask_user_tool() -> ToolDefinition:
    return ToolDefinition(
        name="ask_user",
        description="Ask the user a concise clarification question when required.",
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The clarification question to show the user.",
                }
            },
            "required": ["question"],
            "additionalProperties": False,
        },
        handler=lambda args: {"answer": ""},
    )


def _stock_screener_tool() -> ToolDefinition:
    return ToolDefinition(
        name="stock_screener",
        description="Find A-share or HK stocks that match screening criteria.",
        parameters={
            "type": "object",
            "properties": {
                "market": {
                    "type": "string",
                    "enum": ["A_SHARE", "HK", "BOTH"],
                    "description": "Market universe to screen.",
                },
                "criteria": {
                    "type": "object",
                    "description": "Screening criteria such as dividend yield, P/E, sector, or market cap.",
                    "additionalProperties": True,
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Maximum number of matching stocks to return.",
                },
            },
            "required": ["market", "criteria"],
            "additionalProperties": False,
        },
        handler=_not_implemented("stock_screener"),
    )


def _news_sentiment_tool() -> ToolDefinition:
    return ToolDefinition(
        name="news_sentiment_analysis",
        description="Analyze news sentiment for a listed A-share or HK stock.",
        parameters={
            "type": "object",
            "properties": {
                "stock": {"type": "string", "description": "Ticker, stock code, or company name."},
                "market": {"type": "string", "enum": ["A_SHARE", "HK"]},
                "date_range": {"type": "string", "description": "Natural language or ISO date range."},
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional preferred news sources.",
                },
            },
            "required": ["stock", "market"],
            "additionalProperties": False,
        },
        handler=_not_implemented("news_sentiment_analysis"),
    )


def _technical_analysis_tool() -> ToolDefinition:
    return ToolDefinition(
        name="technical_analysis",
        description="Analyze price and volume data to identify trends and patterns.",
        parameters={
            "type": "object",
            "properties": {
                "stock": {"type": "string", "description": "Ticker, stock code, or company name."},
                "market": {"type": "string", "enum": ["A_SHARE", "HK"]},
                "date_range": {"type": "string", "description": "Natural language or ISO date range."},
                "timeframe": {
                    "type": "string",
                    "description": "Data interval such as daily, weekly, or monthly.",
                },
                "indicators": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Indicators to evaluate, such as MA, RSI, MACD, or volume.",
                },
            },
            "required": ["stock", "market"],
            "additionalProperties": False,
        },
        handler=_not_implemented("technical_analysis"),
    )


def _financial_statement_tool() -> ToolDefinition:
    return ToolDefinition(
        name="financial_statement_analysis",
        description="Analyze financial statements for profitability, leverage, growth, and cash flow health.",
        parameters={
            "type": "object",
            "properties": {
                "stock": {"type": "string", "description": "Ticker, stock code, or company name."},
                "market": {"type": "string", "enum": ["A_SHARE", "HK"]},
                "period_type": {"type": "string", "enum": ["annual", "quarterly", "ttm"]},
                "date_range": {"type": "string", "description": "Natural language or ISO date range."},
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Metrics to inspect, such as ROE, debt ratio, revenue growth, or free cash flow.",
                },
            },
            "required": ["stock", "market"],
            "additionalProperties": False,
        },
        handler=_not_implemented("financial_statement_analysis"),
    )


def _not_implemented(tool_name: str) -> ToolHandler:
    def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "not_implemented",
            "tool": tool_name,
            "arguments": arguments,
            "message": "Tool contract is wired, but market data implementation is pending.",
        }

    return handler
