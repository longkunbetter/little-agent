from __future__ import annotations

from .agent import AgentConfig, StockAgent
from .llm import OpenAIChatCompletionsClient
from .tools import build_default_registry


def main() -> None:
    agent = StockAgent(
        client=OpenAIChatCompletionsClient.deferred_from_env(),
        tools=build_default_registry(),
        config=AgentConfig(),
    )
    agent.run_cli()


if __name__ == "__main__":
    main()
